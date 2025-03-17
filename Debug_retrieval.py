import os
import sys
import time
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple

# This is a simplified debugging version that focuses solely on the vector search functionality

class SimpleRAGDebugger:
    """Minimal class to debug retrieval issues in the RAG pipeline."""
    
    def __init__(self, index_path="indexes"):
        self.index_path = index_path
        self.embedding_model = None
        self.index = None
        self.chunks = {}
        self.chunk_ids = []
        
        # Initialize embedding model
        print("Loading embedding model...")
        try:
            self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()
            print(f"Embedding model loaded with dimension {self.vector_dim}")
        except Exception as e:
            print(f"ERROR loading embedding model: {str(e)}")
            sys.exit(1)
        
        # Try to load existing index
        self._try_load_index()
    
    def _try_load_index(self):
        """Try to load the existing FAISS index with minimal dependencies."""
        index_file = os.path.join(self.index_path, "faiss_index.bin")
        metadata_file = os.path.join(self.index_path, "index_metadata.json")
        chunks_file = os.path.join(self.index_path, "chunks.pkl")
        
        if not os.path.exists(index_file) or not os.path.exists(chunks_file):
            print(f"WARNING: Index files not found at {index_file}")
            self._create_new_index()
            return False
            
        try:
            # Load FAISS index
            print(f"Loading FAISS index from {index_file}")
            self.index = faiss.read_index(index_file)
            print(f"Index loaded with {self.index.ntotal} vectors")
            
            # Check if index is empty
            if self.index.ntotal == 0:
                print("WARNING: Index is empty (ntotal=0)")
                return False
                
            # Load chunks
            import pickle
            with open(chunks_file, 'rb') as f:
                self.chunks = pickle.load(f)
            print(f"Loaded {len(self.chunks)} chunks from {chunks_file}")
            
            # Load metadata if available
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.chunk_ids = metadata.get('chunk_ids', [])
                print(f"Loaded metadata with {len(self.chunk_ids)} chunk IDs")
            
            print("Index loaded successfully")
            return True
            
        except Exception as e:
            print(f"ERROR loading index: {str(e)}")
            print("Creating a new index instead")
            self._create_new_index()
            return False
    
    def _create_new_index(self):
        """Create a new empty FAISS index."""
        print(f"Creating new FAISS index with dimension {self.vector_dim}")
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.index = faiss.IndexIDMap(self.index)
        print("Created new empty index")
    
    def debug_query(self, query_text: str, top_k: int = 5):
        """Run a debug query and print detailed information about what's happening."""
        print("\n" + "="*80)
        print(f"DEBUG QUERY: '{query_text}'")
        print("="*80)
        
        # Check index state
        if self.index is None:
            print("ERROR: No index loaded")
            return []
            
        if self.index.ntotal == 0:
            print("ERROR: Index is empty (ntotal=0)")
            return []
            
        print(f"Index contains {self.index.ntotal} vectors")
        print(f"Chunks dictionary contains {len(self.chunks)} entries")
        
        # Create query embedding
        print("Creating query embedding...")
        start_time = time.time()
        try:
            query_embedding = self.embedding_model.encode(
                [query_text], 
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            print(f"Query embedding created in {time.time() - start_time:.2f}s")
            print(f"Embedding shape: {query_embedding.shape}")
            print(f"Embedding norm: {np.linalg.norm(query_embedding):.4f}")
            
            # Print a sample of the embedding vector
            print(f"Embedding sample: {query_embedding[0][:5]}...")
            
        except Exception as e:
            print(f"ERROR creating query embedding: {str(e)}")
            return []
        
        # Search the index
        print(f"Searching index for top {top_k} results...")
        start_time = time.time()
        try:
            distances, indices = self.index.search(query_embedding, top_k)
            search_time = time.time() - start_time
            print(f"Search completed in {search_time:.4f}s")
            
            # Print raw search results
            print("\nRaw search results:")
            print(f"Distances: {distances[0]}")
            print(f"Indices: {indices[0]}")
            
        except Exception as e:
            print(f"ERROR during search: {str(e)}")
            return []
        
        # Process results
        results = []
        print("\nProcessed results:")
        
        for i in range(len(indices[0])):
            idx = indices[0][i]
            distance = distances[0][i]
            
            # Skip invalid indices
            if idx == -1:
                print(f"  Result {i+1}: Invalid index (-1)")
                continue
            
            # Get the chunk
            chunk = self.chunks.get(int(idx))
            if chunk is None:
                print(f"  Result {i+1}: Chunk not found for index {idx}")
                continue
            
            # Calculate similarity score (convert L2 distance)
            similarity = 1.0 / (1.0 + distance)
            
            # Print result details
            print(f"  Result {i+1}: ID={idx}, Distance={distance:.4f}, Similarity={similarity:.4f}")
            print(f"    Text: {chunk.text[:100]}...")
            
            if hasattr(chunk, 'metadata') and chunk.metadata:
                print(f"    Metadata: {chunk.metadata}")
            
            results.append((chunk, similarity))
        
        if not results:
            print("\nNO RESULTS FOUND! Possible issues:")
            print("1. The index might be empty or corrupted")
            print("2. The query might be too dissimilar from document content")
            print("3. The similarity threshold might be too high")
            print("4. There might be an issue with the embedding model")
        
        return results

def main():
    """Run a simple debug test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug RAG retrieval issues")
    parser.add_argument("--index_path", default="indexes", help="Path to the index directory")
    parser.add_argument("--query", default="What is the main topic of the document?", 
                        help="Query to test")
    parser.add_argument("--top_k", type=int, default=5, 
                        help="Number of results to retrieve")
    
    args = parser.parse_args()
    
    # Create debugger
    debugger = SimpleRAGDebugger(index_path=args.index_path)
    
    # Run debug query
    results = debugger.debug_query(args.query, args.top_k)
    
    # Print summary
    print("\n" + "="*80)
    print(f"SUMMARY: Found {len(results)} results for query '{args.query}'")
    print("="*80)

if __name__ == "__main__":
    main()
