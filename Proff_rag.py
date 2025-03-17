"""
Professional RAG System with accurate retrieval and intelligent chunking.
"""

import os
import time
import uuid
import pickle
import hashlib
import numpy as np
import faiss
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# For PDF processing
import fitz  # PyMuPDF
from PyPDF2 import PdfReader

# For embeddings
from sentence_transformers import SentenceTransformer, util

@dataclass
class Chunk:
    """Representation of a text chunk with metadata."""
    id: str
    text: str
    metadata: Dict[str, Any]

class ProfessionalRAG:
    """Professional RAG system with accurate relevance and intelligent chunking."""
    
    def __init__(self, 
                 index_dir="pro_rag_index",
                 embedding_model="sentence-transformers/all-mpnet-base-v2",
                 chunk_size=500,
                 chunk_overlap=100):
        """Initialize the RAG system with advanced settings."""
        # Create index directory
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Settings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Load embedding model (using a more powerful model for better relevance)
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
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
            # Using IndexFlatIP for inner product (cosine similarity with normalized vectors)
            self.index = faiss.IndexFlatIP(self.vector_dim)
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
        """Process a PDF file with intelligent chunking."""
        print(f"Processing PDF: {pdf_path}")
        start_time = time.time()
        
        try:
            # Use PyMuPDF for better text extraction
            doc = fitz.open(pdf_path)
            
            # Extract structured text
            all_text = []
            for page_num, page in enumerate(doc):
                # Get text with blocks information
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if block["type"] == 0:  # Text block
                        block_text = ""
                        for line in block["lines"]:
                            for span in line["spans"]:
                                block_text += span["text"] + " "
                        
                        # Add paragraph to the collection
                        if block_text.strip():
                            all_text.append({
                                "text": block_text.strip(),
                                "page": page_num + 1
                            })
            
            # Create semantic chunks
            chunks = self._create_semantic_chunks(all_text, pdf_path)
            
            # Add chunks to index
            if chunks:
                num_added = self._add_chunks_to_index(chunks)
                print(f"Added {num_added} semantic chunks from {pdf_path}")
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
    
    def _create_semantic_chunks(self, text_blocks: List[Dict], source_path: str) -> List[Chunk]:
        """Create semantically meaningful chunks from text blocks."""
        chunks = []
        current_text = ""
        current_pages = set()
        
        for block in text_blocks:
            # If adding this block would exceed chunk size and we already have content
            if len(current_text) + len(block["text"]) > self.chunk_size and current_text:
                # Create a chunk
                chunk_id = str(uuid.uuid4())
                chunk = Chunk(
                    id=chunk_id,
                    text=current_text.strip(),
                    metadata={
                        "source": source_path,
                        "pages": sorted(list(current_pages)),
                        "hash": self._get_file_hash(source_path)
                    }
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_size = min(self.chunk_overlap, len(current_text))
                if overlap_size > 0:
                    # Find a sentence boundary for the overlap if possible
                    sentences = re.split(r'(?<=[.!?])\s+', current_text)
                    
                    # Take the last few sentences that fit within overlap_size
                    overlap_text = ""
                    for sentence in reversed(sentences):
                        if len(overlap_text) + len(sentence) + 1 <= overlap_size:
                            overlap_text = sentence + " " + overlap_text
                        else:
                            break
                    
                    current_text = overlap_text
                else:
                    current_text = ""
                
                # Keep the last page in the current pages
                current_pages = {max(current_pages)} if current_pages else set()
            
            # Add the current block
            current_text += block["text"] + " "
            current_pages.add(block["page"])
        
        # Add the final chunk if not empty
        if current_text.strip():
            chunk_id = str(uuid.uuid4())
            chunk = Chunk(
                id=chunk_id,
                text=current_text.strip(),
                metadata={
                    "source": source_path,
                    "pages": sorted(list(current_pages)),
                    "hash": self._get_file_hash(source_path)
                }
            )
            chunks.append(chunk)
        
        return chunks
    
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
            normalize_embeddings=True  # Normalize for cosine similarity
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
        """Query the index for relevant chunks with advanced ranking."""
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
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Process results
        results = []
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            if idx == -1:
                continue
                
            chunk = self.chunks.get(idx)
            if not chunk:
                continue
                
            # Get similarity score (already cosine similarity)
            similarity = scores[0][i]
            
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
            context_part = f"\n--- Document {i+1} (Score: {score:.4f}) ---\n"
            context_part += f"Source: {chunk.metadata['source']}, Pages: {chunk.metadata['pages']}\n"
            context_part += chunk.text
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
    """Demo of the professional RAG system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Professional RAG System")
    parser.add_argument("--pdf", help="Path to PDF file to process")
    parser.add_argument("--query", default="What is this document about?", help="Query to run")
    parser.add_argument("--top_k", type=int, default=3, help="Number of results to retrieve")
    parser.add_argument("--chunk_size", type=int, default=500, help="Size of text chunks")
    parser.add_argument("--chunk_overlap", type=int, default=100, help="Overlap between chunks")
    
    args = parser.parse_args()
    
    # Create RAG system
    rag = ProfessionalRAG(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
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
