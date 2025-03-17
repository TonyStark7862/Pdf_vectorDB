import os
import re
import time
import uuid
from typing import List, Dict, Any, Tuple, Optional, Iterator, Union
from dataclasses import dataclass
import json
import csv
import io

import pandas as pd
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import hashlib
import textwrap

from config import RAGConfig
from logger import RAGLogger

@dataclass
class Chunk:
    """Representation of a text chunk for RAG."""
    id: str
    text: str
    metadata: Dict[str, Any]

class DocumentProcessor:
    """Processor for documents in the RAG pipeline."""
    
    def __init__(self, config: RAGConfig, logger: RAGLogger):
        self.config = config
        self.logger = logger
        
        # OCR model is optional
        self.ocr_model = None
        if self.config.enable_ocr and self.config.ocr_model_path:
            try:
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel
                self.ocr_processor = TrOCRProcessor.from_pretrained(self.config.ocr_model_path)
                self.ocr_model = VisionEncoderDecoderModel.from_pretrained(self.config.ocr_model_path)
                self.logger.info(f"OCR model loaded from {self.config.ocr_model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load OCR model: {str(e)}")
                self.config.enable_ocr = False
    
    def process_file(self, file_path: str) -> List[Chunk]:
        """Process a file and return chunks."""
        start_time = time.time()
        self.logger.info(f"Processing file: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            chunks = self.process_pdf(file_path)
        elif file_ext == '.csv':
            chunks = self.process_csv(file_path)
        else:
            self.logger.warning(f"Unsupported file type: {file_ext}")
            return []
        
        processing_time = time.time() - start_time
        self.logger.info(f"File processed in {processing_time:.2f}s, generated {len(chunks)} chunks")
        return chunks
    
    def process_pdf(self, pdf_path: str) -> List[Chunk]:
        """Process a PDF file and return chunks with context preservation."""
        self.logger.info(f"Processing PDF: {pdf_path}")
        
        chunks = []
        doc_hash = self._get_file_hash(pdf_path)
        
        # Use PyMuPDF (fitz) for better handling of PDF structure
        doc = fitz.open(pdf_path)
        
        # Extract text and structure information
        structured_content = []
        page_images = {}
        
        for page_num, page in enumerate(doc):
            # Extract text with its layout info
            blocks = page.get_text("dict")["blocks"]
            page_text = ""
            
            # Track tables separately to preserve them
            tables = []
            current_table = None
            
            for block in blocks:
                # Handle text blocks
                if block["type"] == 0:  # Text block
                    lines = []
                    for line in block["lines"]:
                        line_text = " ".join([span["text"] for span in line["spans"]])
                        lines.append(line_text)
                    
                    block_text = "\n".join(lines)
                    page_text += block_text + "\n"
                
                # Handle image blocks
                elif block["type"] == 1:  # Image block
                    img_rect = fitz.Rect(block["bbox"])
                    img_index = len(page_images.get(page_num, []))
                    
                    # Create a reference to the image
                    img_ref = f"[IMAGE:{page_num}:{img_index}]"
                    page_text += f"\n{img_ref}\n"
                    
                    # Store image information but don't process with OCR yet
                    if page_num not in page_images:
                        page_images[page_num] = []
                    
                    page_images[page_num].append({
                        "rect": img_rect,
                        "ref": img_ref,
                        "surrounding_text": self._get_surrounding_text(page_text)
                    })
            
            # Append page content
            structured_content.append({
                "page_num": page_num + 1,
                "text": page_text.strip(),
                "tables": tables
            })
        
        # Process with OCR if enabled
        if self.config.enable_ocr and self.ocr_model:
            structured_content = self._process_with_ocr(doc, structured_content, page_images)
        
        # Create chunks with proper context preservation
        chunks = self._create_chunks_from_structured_content(structured_content, doc_hash, pdf_path)
        
        doc.close()
        return chunks
    
    def _get_surrounding_text(self, text: str, window_size: int = 200) -> str:
        """Get the surrounding text for context."""
        if not text:
            return ""
        
        # Get the last window_size characters
        return text[-window_size:] if len(text) > window_size else text
    
    def _process_with_ocr(self, doc, structured_content, page_images):
        """Process images with OCR if available."""
        if not self.ocr_model:
            return structured_content
            
        import PIL.Image
        import torch
        
        for page_num, images in page_images.items():
            page = doc[page_num]
            for img_info in images:
                try:
                    # Extract the image
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=img_info["rect"])
                    img = PIL.Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Process with OCR
                    pixel_values = self.ocr_processor(img, return_tensors="pt").pixel_values
                    generated_ids = self.ocr_model.generate(pixel_values)
                    ocr_text = self.ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    # Replace image reference with OCR text
                    for i, content in enumerate(structured_content):
                        if content["page_num"] - 1 == page_num:
                            content["text"] = content["text"].replace(
                                img_info["ref"], 
                                f"{img_info['ref']}\nOCR Text: {ocr_text}\n"
                            )
                            break
                except Exception as e:
                    self.logger.error(f"OCR processing failed: {str(e)}")
        
        return structured_content
    
    def _create_chunks_from_structured_content(
        self, structured_content, doc_hash, file_path
    ) -> List[Chunk]:
        """Create chunks from structured content with context preservation."""
        chunks = []
        full_text = "\n\n".join([page["text"] for page in structured_content])
        
        # Create chunks with overlap
        chunk_texts = self._split_text_with_overlap(full_text)
        
        for i, chunk_text in enumerate(chunk_texts):
            # Create metadata
            pages_in_chunk = self._identify_pages_in_chunk(chunk_text, structured_content)
            
            metadata = {
                "source": file_path,
                "doc_hash": doc_hash,
                "chunk_index": i,
                "total_chunks": len(chunk_texts),
                "pages": pages_in_chunk,
            }
            
            # Create chunk
            chunk_id = str(uuid.uuid4())
            chunk = Chunk(id=chunk_id, text=chunk_text, metadata=metadata)
            chunks.append(chunk)
        
        return chunks
    
    def _identify_pages_in_chunk(self, chunk_text, structured_content):
        """Identify which pages are included in a chunk."""
        pages = []
        for page in structured_content:
            page_text = page["text"]
            if any(sentence in chunk_text for sentence in page_text.split(".")):
                pages.append(page["page_num"])
        
        return sorted(list(set(pages)))
    
    def process_csv(self, csv_path: str) -> List[Chunk]:
        """Process a CSV file and return chunks with context preservation."""
        self.logger.info(f"Processing CSV: {csv_path}")
        
        chunks = []
        doc_hash = self._get_file_hash(csv_path)
        
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)
            
            # Process the CSV intelligently
            chunks.extend(self._process_csv_tabular(df, doc_hash, csv_path))
            chunks.extend(self._process_csv_semantic(df, doc_hash, csv_path))
            
            self.logger.info(f"Generated {len(chunks)} chunks from CSV")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error processing CSV file: {str(e)}")
            return []
    
    def _process_csv_tabular(self, df, doc_hash, file_path) -> List[Chunk]:
        """Process CSV in a tabular format."""
        chunks = []
        
        # Get column descriptions
        column_descriptions = {col: self._generate_column_description(df, col) for col in df.columns}
        column_description_text = "CSV Column Information:\n" + "\n".join([
            f"- {col}: {desc}" for col, desc in column_descriptions.items()
        ])
        
        # Create a summary chunk
        summary_text = f"CSV Summary for {os.path.basename(file_path)}:\n"
        summary_text += f"Total rows: {len(df)}\n"
        summary_text += f"Columns: {', '.join(df.columns)}\n\n"
        summary_text += column_description_text
        
        summary_chunk = Chunk(
            id=str(uuid.uuid4()),
            text=summary_text,
            metadata={
                "source": file_path,
                "doc_hash": doc_hash,
                "chunk_type": "csv_summary",
                "row_range": [0, len(df)],
            }
        )
        chunks.append(summary_chunk)
        
        # Create row-based chunks for large CSVs
        if len(df) > 100:
            chunk_size = 100
            for i in range(0, len(df), chunk_size):
                end_idx = min(i + chunk_size, len(df))
                
                # Create a tabular view of this chunk
                chunk_df = df.iloc[i:end_idx]
                chunk_text = f"CSV Rows {i+1} to {end_idx} from {os.path.basename(file_path)}:\n\n"
                chunk_text += chunk_df.to_string(index=False, max_rows=chunk_size, max_cols=12)
                
                chunk = Chunk(
                    id=str(uuid.uuid4()),
                    text=chunk_text,
                    metadata={
                        "source": file_path,
                        "doc_hash": doc_hash,
                        "chunk_type": "csv_rows",
                        "row_range": [i, end_idx],
                    }
                )
                chunks.append(chunk)
        else:
            # For small CSVs, include the full table
            full_text = f"Full CSV content of {os.path.basename(file_path)}:\n\n"
            full_text += df.to_string(index=False)
            
            chunk = Chunk(
                id=str(uuid.uuid4()),
                text=full_text,
                metadata={
                    "source": file_path,
                    "doc_hash": doc_hash,
                    "chunk_type": "csv_full",
                    "row_range": [0, len(df)],
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _generate_column_description(self, df, column):
        """Generate a description for a column."""
        try:
            # Get column data type
            dtype = df[column].dtype
            
            # Get unique values if categorical or few distinct values
            unique_count = df[column].nunique()
            sample = None
            
            if unique_count <= 10 and len(df) > 10:
                sample = ", ".join([str(x) for x in df[column].unique() if pd.notna(x)][:10])
            elif len(df) > 0:
                sample = str(df[column].iloc[0])
            
            # Calculate null percentage
            null_pct = (df[column].isna().sum() / len(df)) * 100
            
            description = f"Type: {dtype}"
            if sample:
                description += f", Examples: {sample}"
            if null_pct > 0:
                description += f", Missing: {null_pct:.1f}%"
                
            return description
            
        except:
            return "Unknown"
    
    def _process_csv_semantic(self, df, doc_hash, file_path) -> List[Chunk]:
        """Process CSV with semantic understanding of the data."""
        chunks = []
        
        # Try to identify key columns
        key_columns = self._identify_key_columns(df)
        
        # Process by logical groups (if size allows)
        if len(df.columns) <= 15 and len(df) <= 1000:
            # Group by key columns if possible
            if key_columns and len(df) > 20:
                for group_val, group_df in df.groupby(key_columns[0]):
                    # Create a description of this group
                    group_desc = f"Group: {key_columns[0]}={group_val}\n"
                    group_desc += f"Rows: {len(group_df)}\n\n"
                    
                    # Create a textual view of this group
                    group_desc += group_df.head(20).to_string(index=False)
                    if len(group_df) > 20:
                        group_desc += f"\n... and {len(group_df) - 20} more rows"
                    
                    chunk = Chunk(
                        id=str(uuid.uuid4()),
                        text=group_desc,
                        metadata={
                            "source": file_path,
                            "doc_hash": doc_hash,
                            "chunk_type": "csv_group",
                            "group_column": key_columns[0],
                            "group_value": str(group_val),
                        }
                    )
                    chunks.append(chunk)
        
        # Create narrative descriptions of the data
        try:
            # Basic statistics
            stats_text = f"Statistical summary of {os.path.basename(file_path)}:\n\n"
            
            # Use describe for numerical columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if not numeric_cols.empty:
                stats_df = df[numeric_cols].describe().round(2)
                stats_text += stats_df.to_string() + "\n\n"
            
            # Add categorical summaries if relevant
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if not categorical_cols.empty and len(categorical_cols) <= 5:
                stats_text += "Categorical distributions:\n"
                for col in categorical_cols:
                    if df[col].nunique() <= 10:
                        stats_text += f"\n{col} value counts:\n"
                        stats_text += df[col].value_counts().head(10).to_string()
                        stats_text += "\n"
            
            chunk = Chunk(
                id=str(uuid.uuid4()),
                text=stats_text,
                metadata={
                    "source": file_path,
                    "doc_hash": doc_hash,
                    "chunk_type": "csv_stats",
                }
            )
            chunks.append(chunk)
            
        except Exception as e:
            self.logger.warning(f"Could not generate statistics: {str(e)}")
        
        return chunks
    
    def _identify_key_columns(self, df):
        """Try to identify key/index columns in a dataframe."""
        potential_keys = []
        
        for col in df.columns:
            if df[col].nunique() == len(df) and df[col].notna().all():
                # Perfect unique key
                return [col]
            
            # Look for ID-like column names
            if re.search(r'id$|^id|_id|uuid|key', col.lower()):
                potential_keys.append(col)
                
            # Look for date/time columns that might define logical groups
            elif df[col].dtype == 'datetime64[ns]' or 'date' in col.lower():
                potential_keys.append(col)
        
        # If we found potential keys, return the first one
        if potential_keys:
            return [potential_keys[0]]
        
        # Otherwise return a column with relatively low cardinality for grouping
        cardinality = [(col, df[col].nunique()) for col in df.columns 
                        if df[col].nunique() > 1 and df[col].nunique() <= len(df) / 10]
        
        if cardinality:
            # Sort by cardinality and return the best option
            sorted_cols = sorted(cardinality, key=lambda x: x[1])
            return [sorted_cols[0][0]]
        
        return []
    
    def _split_text_with_overlap(self, text: str) -> List[str]:
        """Split text into chunks with overlap, preserving context."""
        if not text:
            return []
        
        chunk_size = self.config.chunk_size
        chunk_overlap = self.config.chunk_overlap
        
        # If text is shorter than chunk size, return as single chunk
        if len(text) <= chunk_size:
            return [text]
        
        # Split text by paragraphs first
        paragraphs = text.split("\n\n")
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size,
            # finalize the current chunk and start a new one
            if len(current_chunk) + len(paragraph) + 2 > chunk_size and current_chunk:
                chunks.append(current_chunk)
                
                # Start new chunk with overlap from the previous chunk
                overlap_size = min(chunk_overlap, len(current_chunk))
                current_chunk = current_chunk[-overlap_size:] if overlap_size > 0 else ""
                
                # If overlap contains incomplete paragraph, find paragraph boundary
                if "\n\n" in current_chunk:
                    current_chunk = current_chunk.split("\n\n", 1)[1]
            
            # Add paragraph separator if needed
            if current_chunk and not current_chunk.endswith("\n"):
                current_chunk += "\n\n"
                
            # Add the paragraph
            current_chunk += paragraph
        
        # Add the final chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate a hash for the file."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
