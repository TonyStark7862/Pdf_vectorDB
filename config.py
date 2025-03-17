import os
from pathlib import Path
from typing import Dict, Any, Optional, List

class RAGConfig:
    """Configuration for the RAG pipeline."""
    
    def __init__(
        self, 
        embedding_model_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        ocr_model_path: Optional[str] = None,
        reranker_model_path: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        enable_ocr: bool = False,
        enable_reranker: bool = False,
        index_path: str = "indexes",
        log_path: str = "logs",
        max_context_size: int = 4000,
        similarity_top_k: int = 4,
        similarity_cutoff: float = 0.7,
    ):
        self.embedding_model_path = embedding_model_path
        self.ocr_model_path = ocr_model_path
        self.reranker_model_path = reranker_model_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_ocr = enable_ocr
        self.enable_reranker = enable_reranker
        self.index_path = index_path
        self.log_path = log_path
        self.max_context_size = max_context_size
        self.similarity_top_k = similarity_top_k
        self.similarity_cutoff = similarity_cutoff
        
        # Create necessary directories
        os.makedirs(self.index_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "embedding_model_path": self.embedding_model_path,
            "ocr_model_path": self.ocr_model_path,
            "reranker_model_path": self.reranker_model_path,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "enable_ocr": self.enable_ocr,
            "enable_reranker": self.enable_reranker,
            "index_path": self.index_path,
            "log_path": self.log_path,
            "max_context_size": self.max_context_size,
            "similarity_top_k": self.similarity_top_k,
            "similarity_cutoff": self.similarity_cutoff,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
