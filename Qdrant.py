import streamlit as st
import numpy as np
import os
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

# Configuration for local paths
LOCAL_QDRANT_PATH = "./qdrant_data"
LOCAL_MODEL_PATH = "./models/all-MiniLM-L6-v2"

# Ensure directories exist
os.makedirs(LOCAL_QDRANT_PATH, exist_ok=True)
os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

def initialize_embedding_model():
    """Initialize the embedding model from local path or download if not present."""
    try:
        st.info(f"Loading embedding model from {LOCAL_MODEL_PATH}...")
        model = SentenceTransformer(LOCAL_MODEL_PATH)
        st.success("✅ Model loaded from local path")
    except:
        st.warning("Model not found locally. Downloading model (this may take a moment)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Save the model for future use
        model.save(LOCAL_MODEL_PATH)
        st.success("✅ Model downloaded and saved locally")
    return model

def setup_qdrant_client():
    """Setup Qdrant client with local persistence."""
    client = QdrantClient(path=LOCAL_QDRANT_PATH)
    st.success("✅ Connected to local Qdrant database")
    return client

def create_collection(client, collection_name, vector_size=384):
    """Create a new collection if it doesn't exist."""
    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if collection_name not in collection_names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        st.success(f"✅ Collection '{collection_name}' created")
    else:
        st.info(f"Collection '{collection_name}' already exists")

def generate_embeddings(model, texts):
    """Generate embeddings using the provided model."""
    with st.spinner("Generating embeddings..."):
        embeddings = model.encode(texts)
    return embeddings

def add_products(client, collection_name, model, products):
    """Add products to the collection."""
    # Extract product descriptions
    descriptions = [product["description"] for product in products]
    
    # Generate embeddings
    embeddings = generate_embeddings(model, descriptions)
    
    # Get collection info to check if points already exist
    collection_info = client.get_collection(collection_name)
    existing_count = collection_info.points_count
    
    # Generate IDs starting after existing points
    starting_id = existing_count
    
    # Prepare points for upload
    points = [
        models.PointStruct(
            id=starting_id + idx,
            vector=embedding.tolist(),
            payload=product
        )
        for idx, (embedding, product) in enumerate(zip(embeddings, products))
    ]
    
    # Upload to collection
    with st.spinner(f"Adding {len(points)} products to collection..."):
        client.upsert(
            collection_name=collection_name,
            points=points
        )
    
    st.success(f"✅ Added {len(points)} products to collection")

def search_products(client, collection_name, model, query_text, category=None, limit=5):
    """Search for products with optional category filtering."""
    # Generate embedding for query
    query_embedding = generate_embeddings(model, [query_text])[0]
    
    # Prepare filter if category is provided
    if category and category != "All Categories":
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value=category)
                )
            ]
        )
    else:
        query_filter = None
    
    # Search in collection
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        query_filter=query_filter,
        limit=limit
    )
    
    return search_results

# Sample product data
def load_sample_products():
    return [
        {
            "name": "Ergonomic Office Chair",
            "description": "High-quality ergonomic office chair with lumbar support and adjustable height",
            "category": "Furniture",
            "price": 299.99,
            "in_stock": True
        },
        {
            "name": "Mechanical Keyboard",
            "description": "Mechanical gaming keyboard with RGB lighting and programmable keys",
            "category": "Electronics",
            "price": 129.99,
            "in_stock": True
        },
        {
            "name": "Wireless Mouse",
            "description": "Wireless optical mouse with ergonomic design and long battery life",
            "category": "Electronics",
            "price": 49.99,
            "in_stock": True
        },
        {
            "name": "Standing Desk",
            "description": "Adjustable height standing desk for ergonomic work setup",
            "category": "Furniture",
            "price": 399.99,
            "in_stock": False
        },
        {
            "name": "Laptop Stand",
            "description": "Portable laptop stand for better ergonomics and cooling",
            "category": "Accessories",
            "price": 39.99,
            "in_stock": True
        },
        {
            "name": "Ultrawide Monitor",
            "description": "34-inch curved ultrawide monitor with high resolution for immersive productivity",
            "category": "Electronics",
            "price": 449.99,
            "in_stock": True
        },
        {
            "name": "Noise-Cancelling Headphones",
            "description": "Premium wireless headphones with active noise cancellation for focused work",
            "category": "Electronics",
            "price": 249.99,
            "in_stock": True
        },
        {
            "name": "Ergonomic Footrest",
            "description": "Adjustable footrest for improved posture and comfort during long work sessions",
            "category": "Accessories",
            "price": 59.99,
            "in_stock": True
        },
        {
            "name": "Cable Management System",
            "description": "Complete desk cable management solution to organize and hide cables",
            "category": "Accessories",
            "price": 29.99,
            "in_stock": True
        },
        {
            "name": "Smart Desk Lamp",
            "description": "LED desk lamp with adjustable brightness and color temperature, with smart home integration",
            "category": "Lighting",
            "price": 79.99,
            "in_stock": True
        }
    ]

def main():
    st.set_page_config(
        page_title="Qdrant Vector Database Demo",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Qdrant Vector Database Demo")
    st.markdown("""
    This application demonstrates how to use **Qdrant** vector database with **Streamlit** for semantic product search.
    The database and embedding model are stored locally for persistence.
    
    ### Features:
    - Local storage of vector database and embedding model
    - Semantic search based on natural language queries
    - Category filtering
    - Product management (add sample products or custom ones)
    """)
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.collection_name = "product_catalog"
    
    # Initialize model and client
    if not st.session_state.initialized:
        with st.spinner("Initializing application..."):
            st.session_state.model = initialize_embedding_model()
            st.session_state.client = setup_qdrant_client()
            create_collection(
                st.session_state.client, 
                st.session_state.collection_name,
                vector_size=st.session_state.model.get_sentence_embedding_dimension()
            )
            st.session_state.initialized = True
    
    st.divider()
    
    # Sidebar for adding products
    with st.sidebar:
        st.header("Manage Products")
        
        # Option to add sample products
        if st.button("Add Sample Products"):
            sample_products = load_sample_products()
            add_products(
                st.session_state.client,
                st.session_state.collection_name,
                st.session_state.model,
                sample_products
            )
        
        st.divider()
        
        # Add custom product
        st.subheader("Add Custom Product")
        with st.form("add_product_form"):
            name = st.text_input("Product Name")
            description = st.text_area("Product Description")
            category = st.text_input("Category")
            price = st.number_input("Price", min_value=0.0, value=99.99, step=0.01)
            in_stock = st.checkbox("In Stock", value=True)
            
            submit_button = st.form_submit_button("Add Product")
            
            if submit_button and name and description and category:
                new_product = {
                    "name": name,
                    "description": description,
                    "category": category,
                    "price": price,
                    "in_stock": in_stock
                }
                add_products(
                    st.session_state.client,
                    st.session_state.collection_name,
                    st.session_state.model,
                    [new_product]
                )
        
        st.divider()
        
        # Display collection info
        st.subheader("Collection Info")
        if st.button("Refresh Collection Info"):
            try:
                collection_info = st.session_state.client.get_collection(st.session_state.collection_name)
                st.write(f"Collection Name: {st.session_state.collection_name}")
                st.write(f"Number of Products: {collection_info.points_count}")
                st.write(f"Vector Size: {collection_info.config.params.vectors.size}")
            except Exception as e:
                st.error(f"Error fetching collection info: {e}")
    
    # Main area - Search interface
    st.header("Semantic Product Search")
    
    # Query inputs
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Search Query",
            placeholder="Describe what you're looking for...",
            help="Use natural language to describe the product you want to find"
        )
    
    with col2:
        # Get available categories
        categories = ["All Categories"]
        try:
            # Get some sample points to extract categories
            points = st.session_state.client.scroll(
                collection_name=st.session_state.collection_name,
                limit=100
            )[0]
            
            if points:
                # Extract unique categories
                unique_categories = set()
                for point in points:
                    if "category" in point.payload:
                        unique_categories.add(point.payload["category"])
                
                categories.extend(sorted(unique_categories))
        except:
            pass
        
        category_filter = st.selectbox("Category Filter", categories)
    
    # Search button
    search_button = st.button("Search", type="primary")
    
    # Execute search when button is clicked
    if search_button and query:
        results = search_products(
            st.session_state.client,
            st.session_state.collection_name,
            st.session_state.model,
            query,
            category=category_filter
        )
        
        # Display results
        if results:
            st.success(f"Found {len(results)} matching products")
            
            for i, result in enumerate(results):
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.subheader(result.payload["name"])
                        st.write(result.payload["description"])
                    with col2:
                        st.metric("Price", f"${result.payload['price']:.2f}")
                        st.text(f"Category: {result.payload['category']}")
                    with col3:
                        st.metric("Match Score", f"{result.score:.2f}")
                        in_stock = "✅ In Stock" if result.payload.get("in_stock", False) else "❌ Out of Stock"
                        st.text(in_stock)
                st.divider()
        else:
            st.warning("No matching products found. Try a different search query or category.")
    
    # Sample queries section
    st.divider()
    st.header("Sample Queries to Try")
    
    sample_queries = [
        "comfortable seating for my office",
        "devices for typing and computer input",
        "something to help with better posture",
        "audio equipment for focus",
        "desk organization solutions",
        "lighting for my workspace"
    ]
    
    query_cols = st.columns(3)
    for i, sample_query in enumerate(sample_queries):
        with query_cols[i % 3]:
            if st.button(f"🔍 {sample_query}", key=f"sample_{i}"):
                # Pre-fill the search box with this query
                st.experimental_set_query_params(query=sample_query)
                st.experimental_rerun()

if __name__ == "__main__":
    main()
