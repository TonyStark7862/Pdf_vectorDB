import os
import time
import tempfile
import json
import streamlit as st
import pandas as pd
from typing import List, Optional
import uuid

from rag_pipeline_fix import RAGPipeline  # Use the fixed RAG pipeline
from config import RAGConfig

# Set page configuration
st.set_page_config(
    page_title="Production RAG System",
    page_icon="🔍",
    layout="wide",
)

# Function to get a unique session ID
def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

# Initialize session state variables
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "config" not in st.session_state:
    st.session_state.config = None

# Function to initialize or update the RAG pipeline
def initialize_rag_pipeline(config_dict=None):
    if config_dict:
        # Save config to a temporary file
        temp_dir = tempfile.gettempdir()
        config_path = os.path.join(temp_dir, f"rag_config_{get_session_id()}.json")
        
        with open(config_path, "w") as f:
            json.dump(config_dict, f)
            
        st.session_state.rag_pipeline = RAGPipeline(config_path)
    else:
        st.session_state.rag_pipeline = RAGPipeline()
    
    st.session_state.config = st.session_state.rag_pipeline.config.to_dict()

# Initialize RAG pipeline if not already done
if st.session_state.rag_pipeline is None:
    initialize_rag_pipeline()

# Main interface
st.title("Production RAG System")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    with st.expander("Embedding Model", expanded=False):
        embedding_model = st.text_input(
            "Embedding Model Path",
            value=st.session_state.config["embedding_model_path"],
            help="Path to the sentence transformer model for embeddings"
        )
    
    with st.expander("Chunking Settings", expanded=False):
        chunk_size = st.slider(
            "Chunk Size", 
            min_value=100, 
            max_value=2000, 
            value=st.session_state.config["chunk_size"],
            step=100,
            help="Size of text chunks in characters"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=st.session_state.config["chunk_overlap"],
            step=50,
            help="Overlap between consecutive chunks in characters"
        )
    
    with st.expander("Retrieval Settings", expanded=False):
        top_k = st.slider(
            "Top K Results",
            min_value=1,
            max_value=10,
            value=st.session_state.config["similarity_top_k"],
            help="Number of results to retrieve"
        )
        
        similarity_cutoff = st.slider(
            "Similarity Cutoff",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.config["similarity_cutoff"],
            step=0.05,
            help="Minimum similarity score for results"
        )
    
    with st.expander("OCR Settings", expanded=False):
        enable_ocr = st.checkbox(
            "Enable OCR",
            value=st.session_state.config["enable_ocr"],
            help="Enable OCR for images in PDFs"
        )
        
        ocr_model_path = st.text_input(
            "OCR Model Path",
            value=st.session_state.config["ocr_model_path"] or "",
            help="Path to the OCR model"
        )
    
    with st.expander("Reranker Settings", expanded=False):
        enable_reranker = st.checkbox(
            "Enable Reranker",
            value=st.session_state.config["enable_reranker"],
            help="Enable reranker for improved results"
        )
        
        reranker_model_path = st.text_input(
            "Reranker Model Path",
            value=st.session_state.config["reranker_model_path"] or "",
            help="Path to the reranker model"
        )
    
    # Button to apply configuration changes
    if st.button("Apply Configuration"):
        new_config = {
            "embedding_model_path": embedding_model,
            "ocr_model_path": ocr_model_path if enable_ocr else None,
            "reranker_model_path": reranker_model_path if enable_reranker else None,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "enable_ocr": enable_ocr,
            "enable_reranker": enable_reranker,
            "similarity_top_k": top_k,
            "similarity_cutoff": similarity_cutoff,
            "index_path": st.session_state.config["index_path"],
            "log_path": st.session_state.config["log_path"],
            "max_context_size": st.session_state.config["max_context_size"],
        }
        initialize_rag_pipeline(new_config)
        st.success("Configuration updated!")

# Main area with tabs
tab1, tab2, tab3 = st.tabs(["Document Upload", "Query", "System Info"])

# Document Upload Tab
with tab1:
    st.header("Document Upload")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, CSV)",
        type=["pdf", "csv"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Process new files
        new_files = []
        for uploaded_file in uploaded_files:
            # Check if file is already processed
            if uploaded_file.name not in [f["name"] for f in st.session_state.uploaded_files]:
                new_files.append(uploaded_file)
        
        if new_files:
            st.info(f"Processing {len(new_files)} new document(s)...")
            
            # Save files to temp directory and process
            temp_dir = tempfile.gettempdir()
            processed_files = []
            
            progress_bar = st.progress(0)
            
            for i, uploaded_file in enumerate(new_files):
                # Save to temp file
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process the file
                try:
                    start_time = time.time()
                    num_chunks = st.session_state.rag_pipeline.process_document(file_path)
                    elapsed_time = time.time() - start_time
                    
                    file_info = {
                        "name": uploaded_file.name,
                        "path": file_path,
                        "chunks": num_chunks,
                        "processing_time": elapsed_time,
                        "processed_at": time.time()
                    }
                    
                    st.session_state.uploaded_files.append(file_info)
                    processed_files.append(file_info)
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                
                # Update progress
                progress_bar.progress((i + 1) / len(new_files))
            
            if processed_files:
                st.success(f"Successfully processed {len(processed_files)} document(s)!")
    
    # Display uploaded files table
    if st.session_state.uploaded_files:
        st.subheader("Processed Documents")
        
        # Create a table of uploaded files
        file_data = []
        for file_info in st.session_state.uploaded_files:
            file_data.append({
                "Document": file_info["name"],
                "Chunks": file_info["chunks"],
                "Processing Time": f"{file_info['processing_time']:.2f}s"
            })
        
        st.dataframe(pd.DataFrame(file_data))

# Query Tab
with tab2:
    st.header("Ask Questions")
    
    # Query input
    query = st.text_input("Enter your question:")
    
    # Add a test query button
    if st.button("Run Test Query"):
        query = "What is the main topic of the document?"
    
    # Add an optional checkbox for showing the prompt
    show_context = st.checkbox("Show retrieved context", value=False)
    
    submitted = st.button("Submit")
    
    if (submitted or query) and query:
        if st.session_state.uploaded_files:
            with st.spinner("Generating answer..."):
                try:
                    # We'll use a simple passthrough of the llm_function
                    def abc_response(prompt):
                        # This is a placeholder - in real use, this would be imported
                        # as specified in the requirements
                        return "This is where your LLM response would be shown. The context has been successfully retrieved from the document and passed to the LLM function."
                    
                    # Get answer from RAG pipeline
                    result = st.session_state.rag_pipeline.answer_question(query, abc_response)
                    
                    # Add to query history
                    st.session_state.query_history.append({
                        "query": query,
                        "result": result,
                        "timestamp": time.time()
                    })
                    
                    # Display answer
                    st.subheader("Answer")
                    st.write(result["answer"])
                    
                    # Display context if requested
                    if show_context and result.get("supporting_docs"):
                        st.subheader("Retrieved Context")
                        for i, doc in enumerate(result["supporting_docs"]):
                            with st.expander(f"Document {i+1} - Score: {doc['similarity']:.4f}"):
                                st.text(f"Source: {doc['metadata'].get('source', 'Unknown')}")
                                if 'pages' in doc['metadata']:
                                    st.text(f"Pages: {doc['metadata']['pages']}")
                                st.markdown(doc["text"])
                    elif show_context:
                        st.info("No context was retrieved for this query.")
                                
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
                    st.error("Stack trace:", exc_info=True)
        else:
            st.warning("Please upload at least one document first.")

# System Info Tab
with tab3:
    st.header("System Information")
    
    if st.session_state.rag_pipeline:
        stats = st.session_state.rag_pipeline.get_stats()
        
        # Vector DB Stats
        st.subheader("Vector Database")
        st.json(stats["vector_db"])
        
        # Config
        st.subheader("Current Configuration")
        st.json(stats["config"])
        
        # Query History
        if st.session_state.query_history:
            st.subheader("Recent Queries")
            
            for query_item in reversed(st.session_state.query_history[-5:]):
                with st.expander(f"Query: {query_item['query']}"):
                    st.text(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(query_item['timestamp']))}")
                    st.text(f"Found {len(query_item['result'].get('supporting_docs', []))} relevant chunks")
                    if 'response_time' in query_item['result']:
                        st.text(f"Response time: {query_item['result']['response_time']:.2f}s")
