"""
Simple RAG system that focuses on reliability over features.
This implementation strips down the RAG system to its essentials to ensure it works correctly.
"""

import os
import time
import uuid
import pickle
import hashlib
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

@dataclass
class Chunk:
    """Simple representation of a text chunk."""
    id: str
    text: str
    metadata: Dict[str, Any]

class SimpleRAG:
    """A simplified RAG system that focuses on reliability."""
    
    def __init__(self, index_dir="simple_rag_index"):
        # Create index directory
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Load embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"Model loaded with dimension {self.vector_dim}")
        
        # Initialize storage
        self.chunks = {}
        self.chunk_ids = []
        
        # Initialize or load index
        self._init_index()
    
    def _init_index(self):
        """Initialize or load the FAISS index."""
        index_path = os.path.join(self.index_dir, "faiss_index.bin")
        chunks_path = os.path.join(self.index_dir, "chunks.pkl")
        
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            print("Loading existing index...")
            self.index = faiss.read_index(index_path)
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
                self.chunk_ids = list(self.chunks.keys())
            print(f"Loaded index with {len(self.chunk_ids)} chunks")
        else:
            print("Creating new index...")
            self.index = faiss.IndexFlatL2(self.vector_dim)
            self.index = faiss.IndexIDMap(self.index)
            print("New index created")
    
    def _save_index(self):
        """Save the index and chunks to disk."""
        index_path = os.path.join(self.index_dir, "faiss_index.bin")
        chunks_path = os.path.join(self.index_dir, "chunks.pkl")
        
        faiss.write_index(self.index, index_path)
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print(f"Index saved with {len(self.chunk_ids)} chunks")
    
    def process_pdf(self, pdf_path: str) -> int:
        """Process a PDF file and add it to the index."""
        print(f"Processing PDF: {pdf_path}")
        start_time = time.time()
        
        try:
            # Extract text from PDF
            with open(pdf_path, 'rb') as f:
                reader = PdfReader(f)
                text_chunks = []
                
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        # Create a chunk per page for simplicity
                        chunk = Chunk(
                            id=str(uuid.uuid4()),
                            text=page_text,
                            metadata={
                                "source": pdf_path,
                                "page": i + 1,
                                "hash": self._get_file_hash(pdf_path)
                            }
                        )
                        text_chunks.append(chunk)
            
            # Add chunks to index
            if text_chunks:
                num_added = self._add_chunks_to_index(text_chunks)
                print(f"Added {num_added} chunks from {pdf_path}")
                self._save_index()
                
                elapsed_time = time.time() - start_time
                print(f"Processing completed in {elapsed_time:.2f}s")
                return num_added
            else:
                print(f"No text extracted from {pdf_path}")
                return 0
                
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return 0
    
    def _add_chunks_to_index(self, chunks: List[Chunk]) -> int:
        """Add chunks to the vector index."""
        if not chunks:
            return 0
        
        # Create embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Assign IDs
        chunk_ids = np.array([len(self.chunk_ids) + i for i in range(len(chunks))], dtype=np.int64)
        
        # Add to index
        self.index.add_with_ids(embeddings, chunk_ids)
        
        # Store chunks
        for i, chunk in enumerate(chunks):
            chunk_id = int(chunk_ids[i])
            self.chunks[chunk_id] = chunk
            self.chunk_ids.append(chunk_id)
        
        return len(chunks)
    
    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Query the index for relevant chunks."""
        print(f"Querying: '{query_text}'")
        start_time = time.time()
        
        if not self.index or self.index.ntotal == 0:
            print("Index is empty, no results")
            return []
        
        # Create query embedding
        query_embedding = self.embedding_model.encode(
            [query_text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Process results
        results = []
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            if idx == -1:
                continue
                
            chunk = self.chunks.get(idx)
            if not chunk:
                continue
                
            # Convert L2 distance to similarity score
            distance = distances[0][i]
            similarity = 1.0 / (1.0 + distance)
            
            results.append((chunk, similarity))
        
        elapsed_time = time.time() - start_time
        print(f"Found {len(results)} results in {elapsed_time:.4f}s")
        
        return results
    
    def generate_answer(self, query_text: str, top_k: int = 3):
        """Generate a simple answer to the query."""
        results = self.query(query_text, top_k)
        
        if not results:
            return "No relevant information found in the documents."
        
        # Build context for the answer
        context_parts = []
        for i, (chunk, score) in enumerate(results):
            context_part = f"\n--- Document {i+1} (Score: {score:.2f}) ---\n"
            context_part += f"Source: {chunk.metadata['source']}, Page: {chunk.metadata['page']}\n"
            context_part += chunk.text[:500] + "..."  # Truncate for display
            context_parts.append(context_part)
        
        context = "\n".join(context_parts)
        
        return f"""Query: {query_text}

Retrieved {len(results)} relevant document chunks:
{context}

To get an actual LLM response, you would pass this context to your LLM.
"""
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate a hash for the file."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


def main():
    """Simple demo of the RAG system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple RAG System")
    parser.add_argument("--pdf", help="Path to PDF file to process")
    parser.add_argument("--query", default="What is this document about?", help="Query to run")
    parser.add_argument("--top_k", type=int, default=3, help="Number of results to retrieve")
    
    args = parser.parse_args()
    
    # Create RAG system
    rag = SimpleRAG()
    
    # Process PDF if provided
    if args.pdf:
        rag.process_pdf(args.pdf)
    
    # Run query
    print("\n" + "="*80)
    answer = rag.generate_answer(args.query, args.top_k)
    print(answer)
    print("="*80)

if __name__ == "__main__":
    main()
