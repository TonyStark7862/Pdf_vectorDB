import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

class RAGLogger:
    """Logger for the RAG pipeline."""
    
    def __init__(self, log_path: str = "logs"):
        self.log_path = log_path
        os.makedirs(self.log_path, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger("rag_pipeline")
        self.logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        if self.logger.handlers:
            for handler in self.logger.handlers:
                self.logger.removeHandler(handler)
        
        # Create a file handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_path, f"rag_pipeline_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        
        # Create a console handler
        console_handler = logging.StreamHandler()
        
        # Create a formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
        
    def log_document_processing(self, doc_path: str, chunk_count: int):
        """Log document processing details."""
        self.logger.info(f"Processed document: {doc_path} - Generated {chunk_count} chunks")
    
    def log_query(self, query: str, retrieval_time: float, matched_chunks: int):
        """Log query processing details."""
        self.logger.info(f"Query: '{query}' - Retrieved {matched_chunks} chunks in {retrieval_time:.4f}s")
    
    def log_embedding_creation(self, num_embeddings: int, time_taken: float):
        """Log embedding creation details."""
        self.logger.info(f"Created {num_embeddings} embeddings in {time_taken:.4f}s")
