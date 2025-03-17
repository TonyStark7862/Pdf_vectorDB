import os
import json
import time
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict

import torch
from sentence_transformers import SentenceTransformer

from config import RAGConfig
from logger import RAGLogger
from document_processor import Chunk

class VectorDatabase:
    """Vector database using FAISS for the RAG pipeline."""
    
    def __init__(self, config: RAGConfig, logger: RAGLogger):
        self.config = config
        self.logger = logger
        
        # Initialize the embedding model
        start_time = time.time()
        self.logger.info(f"Loading embedding model: {self.config.embedding_model_path}")
        try:
            self.embedding_model = SentenceTransformer(self.config.embedding_model_path)
            self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()
            self.logger.info(f"Embedding model loaded in {time.time() - start_time:.2f}s")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {str(e)}")
            raise RuntimeError(f"Failed to load embedding model: {str(e)}")
            
        # Optional reranker
        self.reranker = None
        if self.config.enable_reranker and self.config.reranker_model_path:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(self.config.reranker_model_path)
                self.logger.info(f"Reranker model loaded from {self.config.reranker_model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load reranker model: {str(e)}")
                self.config.enable_reranker = False
        
        # Initialize FAISS index
        self.index = None
        self.chunks = {}  # Dictionary to store chunk objects by ID
        self.chunk_ids = []  # Ordered list of chunk IDs in the index
        
        # Create index directory
        os.makedirs(self.config.index_path, exist_ok=True)
        
        # Try to load existing index
        self._try_load_index()
    
    def _try_load_index(self):
        """Try to load an existing index if available."""
        index_file = os.path.join(self.config.index_path, "faiss_index.bin")
        metadata_file = os.path.join(self.config.index_path, "index_metadata.json")
        chunks_file = os.path.join(self.config.index_path, "chunks.pkl")
        
        if os.path.exists(index_file) and os.path.exists(metadata_file) and os.path.exists(chunks_file):
            try:
                # Load FAISS index
                self.index = faiss.read_index(index_file)
                
                # Load metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.chunk_ids = metadata.get('chunk_ids', [])
                
                # Load chunks
                with open(chunks_file, 'rb') as f:
                    self.chunks = pickle.load(f)
                
                self.logger.info(f"Loaded existing index with {len(self.chunk_ids)} chunks")
                return True
            except Exception as e:
                self.logger.error(f"Failed to load existing index: {str(e)}")
                self.index = None
                self.chunks = {}
                self.chunk_ids = []
        
        # Create a new index if loading failed
        self._create_new_index()
        return False
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        # Create a simple flat L2 index for maximum accuracy
        self.index = faiss.IndexFlatL2(self.vector_dim)
        
        # Add an ID mapping layer
        self.index = faiss.IndexIDMap(self.index)
        
        self.logger.info(f"Created new FAISS index with dimension {self.vector_dim}")
    
    def add_chunks(self, chunks: List[Chunk]) -> int:
        """Add chunks to the vector database."""
        if not chunks:
            return 0
        
        start_time = time.time()
        self.logger.info(f"Adding {len(chunks)} chunks to the vector database")
        
        # Get texts for embedding
        texts = [chunk.text for chunk in chunks]
        
        # Create embeddings
        embeddings = self._create_embeddings(texts)
        if embeddings is None:
            self.logger.error("Failed to create embeddings")
            return 0
        
        # Assign IDs to chunks
        chunk_ids = np.array([i + len(self.chunk_ids) for i in range(len(chunks))], dtype=np.int64)
        
        # Add embeddings to the index
        self.index.add_with_ids(embeddings, chunk_ids)
        
        # Store chunks in memory
        for i, chunk in enumerate(chunks):
            chunk_id = int(chunk_ids[i])
            self.chunks[chunk_id] = chunk
            self.chunk_ids.append(chunk_id)
        
        # Save the updated index
        self._save_index()
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Added {len(chunks)} chunks in {elapsed_time:.2f}s")
        
        return len(chunks)
    
    def _create_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Create embeddings for the given texts."""
        if not texts:
            return None
        
        start_time = time.time()
        
        try:
            # Use sentence-transformers to create embeddings
            embeddings = self.embedding_model.encode(
                texts, 
                show_progress_bar=False, 
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            embedding_time = time.time() - start_time
            self.logger.log_embedding_creation(len(texts), embedding_time)
            
            return embeddings
        except Exception as e:
            self.logger.error(f"Error creating embeddings: {str(e)}")
            return None
    
    def search(self, query: str, top_k: Optional[int] = None, cutoff_score: Optional[float] = None) -> List[Tuple[Chunk, float]]:
        """Search for chunks matching the query."""
        if not self.index or self.index.ntotal == 0:
            self.logger.warning("No documents in the index")
            return []
        
        start_time = time.time()
        
        # Use configured values if not explicitly provided
        top_k = top_k or self.config.similarity_top_k
        cutoff_score = cutoff_score or self.config.similarity_cutoff
        
        # Create query embedding
        query_embedding = self._create_embeddings([query])
        if query_embedding is None:
            self.logger.error("Failed to create query embedding")
            return []
        
        # Search the index
        scores, chunk_indices = self.index.search(query_embedding, top_k)
        
        # Process results
        results = []
        for i in range(len(scores[0])):
            score = scores[0][i]
            chunk_idx = chunk_indices[0][i]
            
            # Skip invalid indices and scores below cutoff
            if chunk_idx == -1 or score > cutoff_score:  # Lower L2 distance is better
                continue
            
            # Get the chunk
            chunk = self.chunks.get(int(chunk_idx))
            if chunk is None:
                continue
            
            # Convert L2 distance to similarity score (0-1)
            similarity = 1.0 / (1.0 + score)
            
            results.append((chunk, similarity))
        
        # Apply reranking if enabled
        if self.config.enable_reranker and self.reranker and results:
            results = self._rerank_results(query, results)
        
        elapsed_time = time.time() - start_time
        self.logger.log_query(query, elapsed_time, len(results))
        
        return results
    
    def _rerank_results(self, query: str, results: List[Tuple[Chunk, float]]) -> List[Tuple[Chunk, float]]:
        """Rerank results using cross-encoder."""
        if not self.reranker:
            return results
        
        # Prepare pairs for reranking
        pairs = [(query, chunk.text) for chunk, _ in results]
        
        # Get scores from reranker
        rerank_scores = self.reranker.predict(pairs)
        
        # Create new results with reranked scores
        reranked_results = [(results[i][0], float(rerank_scores[i])) 
                            for i in range(len(results))]
        
        # Sort by score in descending order
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        return reranked_results
    
    def _save_index(self):
        """Save the index and metadata to disk."""
        try:
            # Save FAISS index
            index_file = os.path.join(self.config.index_path, "faiss_index.bin")
            faiss.write_index(self.index, index_file)
            
            # Save metadata
            metadata_file = os.path.join(self.config.index_path, "index_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump({
                    'vector_dim': self.vector_dim,
                    'chunk_ids': self.chunk_ids,
                    'total_chunks': len(self.chunk_ids),
                    'embedding_model': self.config.embedding_model_path,
                }, f)
            
            # Save chunks
            chunks_file = os.path.join(self.config.index_path, "chunks.pkl")
            with open(chunks_file, 'wb') as f:
                pickle.dump(self.chunks, f)
            
            self.logger.info(f"Saved index with {len(self.chunk_ids)} chunks")
        except Exception as e:
            self.logger.error(f"Error saving index: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database."""
        return {
            'total_chunks': len(self.chunk_ids),
            'embedding_model': self.config.embedding_model_path,
            'vector_dimension': self.vector_dim,
            'index_type': type(self.index).__name__,
        }
