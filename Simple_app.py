import os
import time
import tempfile
import streamlit as st
import pandas as pd
from typing import List, Dict

# Import the simplified RAG system
from simple_rag import SimpleRAG, Chunk

# Page configuration
st.set_page_config(
    page_title="Simple RAG Q&A System",
    page_icon="🔍",
    layout="wide"
)

# Initialize the RAG system
@st.cache_resource
def get_rag_system():
    return SimpleRAG()

rag = get_rag_system()

# Initialize session state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# Main page
st.title("Simple RAG Q&A System")

# Sidebar
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of results to retrieve", min_value=1, max_value=10, value=3)
    show_context = st.checkbox("Show context from documents", value=True)
    
    st.header("Document Processor")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file:
        if uploaded_file.name not in [f["name"] for f in st.session_state.uploaded_files]:
            with st.spinner("Processing document..."):
                # Save to temp file
                temp_dir = tempfile.gettempdir()
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process the document
                start_time = time.time()
                num_chunks = rag.process_pdf(file_path)
                process_time = time.time() - start_time
                
                if num_chunks > 0:
                    # Add to session state
                    st.session_state.uploaded_files.append({
                        "name": uploaded_file.name,
                        "path": file_path,
                        "chunks": num_chunks,
                        "time": process_time
                    })
                    st.success(f"Processed {uploaded_file.name} into {num_chunks} chunks")
                else:
                    st.error(f"Failed to process {uploaded_file.name}")

# Main content area
tab1, tab2 = st.tabs(["Q&A", "Documents"])

# Q&A Tab
with tab1:
    st.header("Ask a Question")
    
    # Query input
    query = st.text_input("Enter your question:")
    submit = st.button("Submit")
    
    if submit and query:
        if not st.session_state.uploaded_files:
            st.warning("Please upload at least one document first.")
        else:
            with st.spinner("Searching for answers..."):
                # Get results
                start_time = time.time()
                results = rag.query(query, top_k)
                search_time = time.time() - start_time
                
                # Add to history
                st.session_state.query_history.append({
                    "query": query,
                    "results": results,
                    "time": search_time,
                    "timestamp": time.time()
                })
                
                # Display results
                st.subheader("Answer")
                
                # Basic answer 
                if results:
                    st.success(f"Found {len(results)} relevant passages in {search_time:.2f} seconds")
                    
                    # Function that would be imported in a real system
                    def abc_response(prompt):
                        return "This is where your LLM response would appear. In a real system, you would import the abc_response function that generates answers based on the retrieved context."
                    
                    # Prepare context for LLM
                    context = ""
                    for i, (chunk, score) in enumerate(results):
                        context += f"[Document {i+1}] {chunk.text}\n\n"
                    
                    # Build prompt
                    prompt = f"""Answer the following question based on the provided context:

Context:
{context}

Question: {query}

Answer:"""

                    # Get LLM response
                    llm_answer = abc_response(prompt)
                    st.markdown(llm_answer)
                    
                    # Show context if requested
                    if show_context:
                        st.subheader("Retrieved Context")
                        for i, (chunk, score) in enumerate(results):
                            with st.expander(f"Document {i+1} - Score: {score:.4f}"):
                                st.text(f"Source: {chunk.metadata.get('source', 'Unknown')}")
                                st.text(f"Page: {chunk.metadata.get('page', 'Unknown')}")
                                st.markdown(chunk.text)
                else:
                    st.warning("No relevant information found in the documents.")

# Documents Tab
with tab2:
    st.header("Processed Documents")
    
    if st.session_state.uploaded_files:
        # Create dataframe
        docs_data = []
        for doc in st.session_state.uploaded_files:
            docs_data.append({
                "Document": doc["name"],
                "Chunks": doc["chunks"],
                "Processing Time": f"{doc['time']:.2f}s"
            })
        
        st.dataframe(pd.DataFrame(docs_data))
        
        # Show recent queries
        if st.session_state.query_history:
            st.subheader("Recent Queries")
            
            for i, query_item in enumerate(reversed(st.session_state.query_history[-5:])):
                with st.expander(f"Query: {query_item['query']}"):
                    st.text(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(query_item['timestamp']))}")
                    st.text(f"Results: {len(query_item['results'])} passages")
                    st.text(f"Search time: {query_item['time']:.4f}s")
    else:
        st.info("No documents have been processed yet. Upload a PDF in the sidebar.")

# Footer
st.markdown("---")
st.caption("Simple RAG Q&A System - Built with Streamlit and FAISS")
