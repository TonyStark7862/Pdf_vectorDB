import os
import time
from typing import List, Dict, Any, Optional, Tuple
import json

from config import RAGConfig
from logger import RAGLogger
from document_processor import DocumentProcessor, Chunk
from vector_database import VectorDatabase

class RAGPipeline:
    """Main RAG pipeline that coordinates document processing and vector search."""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load config from file if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                self.config = RAGConfig.from_dict(config_dict)
        else:
            self.config = RAGConfig()
        
        # Initialize logger
        self.logger = RAGLogger(self.config.log_path)
        self.logger.info("Initializing RAG pipeline")
        
        # Initialize components
        self.document_processor = DocumentProcessor(self.config, self.logger)
        self.vector_db = VectorDatabase(self.config, self.logger)
        
        self.logger.info("RAG pipeline initialized")
    
    def process_document(self, file_path: str) -> int:
        """Process a document and add it to the vector database.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Number of chunks added to the vector database
        """
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            return 0
        
        start_time = time.time()
        self.logger.info(f"Processing document: {file_path}")
        
        # Process document to create chunks
        chunks = self.document_processor.process_file(file_path)
        if not chunks:
            self.logger.warning(f"No chunks extracted from {file_path}")
            return 0
        
        # Add chunks to vector database
        num_added = self.vector_db.add_chunks(chunks)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Document processing completed in {elapsed_time:.2f}s, added {num_added} chunks")
        
        return num_added
    
    def process_documents(self, file_paths: List[str]) -> int:
        """Process multiple documents and add them to the vector database.
        
        Args:
            file_paths: List of paths to document files
            
        Returns:
            Total number of chunks added to the vector database
        """
        total_chunks = 0
        for file_path in file_paths:
            num_chunks = self.process_document(file_path)
            total_chunks += num_chunks
        
        return total_chunks
    
    def query(self, query_text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Query the RAG pipeline for relevant document chunks.
        
        Args:
            query_text: The query text
            top_k: Maximum number of results to return (uses config value if None)
            
        Returns:
            List of relevant chunks with their metadata and similarity scores
        """
        start_time = time.time()
        self.logger.info(f"Processing query: {query_text}")
        
        # Search vector database
        results = self.vector_db.search(query_text, top_k)
        
        # Format results
        formatted_results = []
        for chunk, score in results:
            formatted_results.append({
                'chunk_id': chunk.id,
                'text': chunk.text,
                'metadata': chunk.metadata,
                'similarity': score
            })
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Query processed in {elapsed_time:.2f}s, found {len(formatted_results)} results")
        
        return formatted_results
    
    def generate_prompt(self, query_text: str, top_k: Optional[int] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate a prompt for an LLM with retrieved context.
        
        Args:
            query_text: The query text
            top_k: Maximum number of context chunks to include
            
        Returns:
            Tuple of (prompt_text, raw_results)
        """
        # Get relevant chunks
        results = self.query(query_text, top_k)
        
        # Build context from chunks
        context_parts = []
        for i, result in enumerate(results):
            context_part = f"[Document {i+1}] {result['text']}"
            context_parts.append(context_part)
        
        context_text = "\n\n".join(context_parts)
        
        # Build the prompt
        prompt = f"""Answer the following question based on the provided context information only.
If you cannot answer the question with the given context, please respond with "I don't have enough information to answer that question."

Context:
{context_text}

Question: {query_text}

Answer:"""
        
        return prompt, results
    
    def answer_question(self, query_text: str, llm_function, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Answer a question using the RAG pipeline and an external LLM.
        
        Args:
            query_text: The query text
            llm_function: A function that takes a prompt and returns an LLM response
            top_k: Maximum number of context chunks to include
            
        Returns:
            Dictionary with the answer and supporting information
        """
        # Generate prompt with context
        prompt, results = self.generate_prompt(query_text, top_k)
        
        # Call the LLM function
        try:
            start_time = time.time()
            llm_response = llm_function(prompt)
            elapsed_time = time.time() - start_time
            
            self.logger.info(f"LLM response generated in {elapsed_time:.2f}s")
            
            # Return answer with supporting information
            return {
                'query': query_text,
                'answer': llm_response,
                'prompt': prompt,
                'supporting_docs': results,
                'response_time': elapsed_time
            }
            
        except Exception as e:
            self.logger.error(f"Error generating LLM response: {str(e)}")
            return {
                'query': query_text,
                'answer': "Error generating answer",
                'error': str(e),
                'supporting_docs': results
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG pipeline."""
        return {
            'vector_db': self.vector_db.get_stats(),
            'config': self.config.to_dict()
        }
