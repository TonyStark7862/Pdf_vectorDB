"""
Professional Streamlit UI for RAG Q&A System with accurate retrieval.
"""

import os
import time
import tempfile
import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import plotly.express as px

# Import the professional RAG system
from professional_rag import ProfessionalRAG, Chunk

# Page configuration
st.set_page_config(
    page_title="Professional RAG Q&A System",
    page_icon="🔍",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        color: #1E88E5;
        font-size: 38px;
        font-weight: bold;
    }
    .sub-header {
        color: #0D47A1;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .result-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #1E88E5;
    }
    .score-high {
        color: #2E7D32;
        font-weight: bold;
    }
    .score-medium {
        color: #FF8F00;
        font-weight: bold;
    }
    .score-low {
        color: #C62828;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the RAG system with caching
@st.cache_resource
def get_rag_system(chunk_size=500, chunk_overlap=100):
    return ProfessionalRAG(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

# Initialize session state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 500
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = 100
if "rag" not in st.session_state:
    st.session_state.rag = get_rag_system(
        st.session_state.chunk_size,
        st.session_state.chunk_overlap
    )

# Main page header
st.markdown('<p class="main-header">Professional RAG Q&A System</p>', unsafe_allow_html=True)
st.markdown("High-accuracy retrieval for document question answering")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Q&A", "Documents", "Settings"])

# Q&A Tab
with tab1:
    st.markdown('<p class="sub-header">Ask a Question</p>', unsafe_allow_html=True)
    
    # Two-column layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Query input
        query = st.text_input("Enter your question:", placeholder="What information are you looking for?")
    
    with col2:
        # Settings for this query
        top_k = st.slider("Results to retrieve", min_value=1, max_value=10, value=3)
        similarity_threshold = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    
    # Full width for the submit button
    submit = st.button("Search Documents", use_container_width=True)
    
    # Process the query when submitted
    if submit and query:
        if not st.session_state.uploaded_files:
            st.warning("📚 Please upload at least one document first (go to Settings tab)")
        else:
            with st.spinner("🔍 Searching through documents..."):
                # Get results
                start_time = time.time()
                results = st.session_state.rag.query(query, top_k)
                search_time = time.time() - start_time
                
                # Filter by similarity threshold
                results = [(chunk, score) for chunk, score in results if score >= similarity_threshold]
                
                # Add to history
                st.session_state.query_history.append({
                    "query": query,
                    "results": results,
                    "time": search_time,
                    "timestamp": time.time()
                })
                
                # Display results
                if results:
                    st.success(f"✅ Found {len(results)} relevant passages in {search_time:.2f} seconds")
                    
                    # Function that would be imported in a real system
                    def abc_response(prompt):
                        return "This is where your LLM response would appear. In a real system, you would import the abc_response function that generates answers based on the retrieved context."
                    
                    # Prepare context for LLM
                    context = ""
                    for i, (chunk, score) in enumerate(results):
                        context += f"[Document {i+1}] Source: {os.path.basename(chunk.metadata['source'])}, Pages: {chunk.metadata['pages']}\n{chunk.text}\n\n"
                    
                    # Build prompt
                    prompt = f"""Answer the following question based on the provided context:

Context:
{context}

Question: {query}

Answer:"""

                    # Get LLM response
                    with st.container(border=True):
                        st.subheader("📝 Answer")
                        llm_answer = abc_response(prompt)
                        st.markdown(llm_answer)
                    
                    # Display results
                    st.markdown('<p class="sub-header">Retrieved Passages</p>', unsafe_allow_html=True)
                    
                    for i, (chunk, score) in enumerate(results):
                        # Determine score class
                        score_class = "score-high" if score >= 0.8 else "score-medium" if score >= 0.6 else "score-low"
                        
                        # Create an expander for each result
                        with st.expander(f"Passage {i+1} - Relevance: {score:.4f}"):
                            st.markdown(f'<div class="result-box">', unsafe_allow_html=True)
                            
                            # Source info
                            st.markdown(f"**Source**: {os.path.basename(chunk.metadata['source'])}")
                            st.markdown(f"**Pages**: {chunk.metadata['pages']}")
                            st.markdown(f"**Relevance**: <span class='{score_class}'>{score:.4f}</span>", unsafe_allow_html=True)
                            
                            # Content with highlighting
                            st.markdown("**Content**:")
                            st.markdown(chunk.text)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No relevant information found that meets the similarity threshold.")

# Documents Tab
with tab2:
    st.markdown('<p class="sub-header">Document Analysis</p>', unsafe_allow_html=True)
    
    if st.session_state.uploaded_files:
        # Create dataframe
        docs_data = []
        for doc in st.session_state.uploaded_files:
            docs_data.append({
                "Document": os.path.basename(doc["path"]),
                "Chunks": doc["chunks"],
                "Processing Time": doc["time"]
            })
        
        # Two columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Show table
            st.dataframe(
                pd.DataFrame(docs_data),
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            # Charts
            if len(docs_data) > 0:
                # Create chart data
                chart_df = pd.DataFrame(docs_data)
                
                # Chunks per document
                fig = px.bar(
                    chart_df, 
                    x="Document", 
                    y="Chunks", 
                    title="Chunks per Document",
                    color="Chunks",
                    color_continuous_scale="blues"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Query history visualization
        if st.session_state.query_history:
            st.markdown('<p class="sub-header">Query History</p>', unsafe_allow_html=True)
            
            # Create dataframe for history
            history_data = []
            for item in st.session_state.query_history:
                history_data.append({
                    "Query": item["query"],
                    "Results": len(item["results"]),
                    "Search Time (s)": item["time"],
                    "Timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(item["timestamp"]))
                })
            
            # Display as table
            st.dataframe(
                pd.DataFrame(history_data),
                use_container_width=True,
                hide_index=True
            )
            
            # Show recent queries
            st.markdown('<p class="sub-header">Recent Query Details</p>', unsafe_allow_html=True)
            
            for i, query_item in enumerate(reversed(st.session_state.query_history[-5:])):
                with st.expander(f"Query: {query_item['query']}"):
                    st.text(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(query_item['timestamp']))}")
                    st.text(f"Results: {len(query_item['results'])} passages")
                    st.text(f"Search time: {query_item['time']:.4f}s")
                    
                    # Show results scores
                    if query_item['results']:
                        scores = [score for _, score in query_item['results']]
                        score_df = pd.DataFrame({
                            "Passage": [f"Passage {i+1}" for i in range(len(scores))],
                            "Relevance Score": scores
                        })
                        
                        # Bar chart of scores
                        fig = px.bar(
                            score_df, 
                            x="Passage", 
                            y="Relevance Score",
                            color="Relevance Score",
                            color_continuous_scale="RdYlGn",
                            range_y=[0, 1]
                        )
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📚 No documents have been processed yet. Go to the Settings tab to upload documents.")

# Settings Tab
with tab3:
    st.markdown('<p class="sub-header">System Settings</p>', unsafe_allow_html=True)
    
    # Two columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Chunking settings
        st.markdown("### Chunking Settings")
        
        new_chunk_size = st.slider(
            "Chunk Size", 
            min_value=100, 
            max_value=1000, 
            value=st.session_state.chunk_size,
            step=50,
            help="Size of text chunks in characters"
        )
        
        new_chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=300,
            value=st.session_state.chunk_overlap,
            step=25,
            help="Overlap between chunks in characters"
        )
        
        # Apply settings button
        if st.button("Apply Settings"):
            # Check if settings changed
            if (new_chunk_size != st.session_state.chunk_size or 
                new_chunk_overlap != st.session_state.chunk_overlap):
                
                st.session_state.chunk_size = new_chunk_size
                st.session_state.chunk_overlap = new_chunk_overlap
                
                # Recreate RAG system with new settings
                st.session_state.rag = get_rag_system(
                    st.session_state.chunk_size,
                    st.session_state.chunk_overlap
                )
                
                # Clear processed files since they need to be reprocessed
                st.session_state.uploaded_files = []
                
                st.success("✅ Settings updated! Please reupload your documents.")
            else:
                st.info("ℹ️ No changes in settings.")
    
    with col2:
        # Document upload
        st.markdown("### Upload Documents")
        
        uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
        
        if uploaded_file:
            if uploaded_file.name not in [os.path.basename(f["path"]) for f in st.session_state.uploaded_files]:
                with st.spinner("Processing document..."):
                    # Save to temp file
                    temp_dir = tempfile.gettempdir()
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process the document
                    start_time = time.time()
                    num_chunks = st.session_state.rag.process_pdf(file_path)
                    process_time = time.time() - start_time
                    
                    if num_chunks > 0:
                        # Add to session state
                        st.session_state.uploaded_files.append({
                            "name": uploaded_file.name,
                            "path": file_path,
                            "chunks": num_chunks,
                            "time": process_time
                        })
                        st.success(f"✅ Processed {uploaded_file.name} into {num_chunks} chunks")
                    else:
                        st.error(f"❌ Failed to process {uploaded_file.name}")
            else:
                st.info(f"ℹ️ {uploaded_file.name} is already processed.")
    
    # System information
    st.markdown('<p class="sub-header">System Information</p>', unsafe_allow_html=True)
    
    # Three columns for system info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Documents Processed",
            value=len(st.session_state.uploaded_files)
        )
    
    with col2:
        total_chunks = sum(doc["chunks"] for doc in st.session_state.uploaded_files) if st.session_state.uploaded_files else 0
        st.metric(
            label="Total Chunks",
            value=total_chunks
        )
    
    with col3:
        st.metric(
            label="Queries Run",
            value=len(st.session_state.query_history)
        )

# Footer
st.markdown("---")
st.caption("Professional RAG Q&A System - Built with FAISS and Sentence Transformers")
