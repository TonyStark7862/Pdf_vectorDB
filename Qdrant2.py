import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
import os

# Define local paths
LOCAL_QDRANT_PATH = "./simple_qdrant_data"
LOCAL_EMBEDDINGS_PATH = "./simple_qdrant_data/embeddings.npy"

# Ensure directory exists
os.makedirs(LOCAL_QDRANT_PATH, exist_ok=True)

def check_qdrant_installation():
    """Verify Qdrant client is working properly"""
    print("Testing Qdrant client installation...")
    try:
        # Create in-memory client for testing
        client = QdrantClient(":memory:")
        print("✅ Qdrant client initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Qdrant client initialization failed: {e}")
        return False

def test_vector_operations():
    """Test basic vector operations with Qdrant"""
    print("\nTesting vector operations...")
    
    try:
        # Create client with persistent storage
        client = QdrantClient(path=LOCAL_QDRANT_PATH)
        print("✅ Connected to local Qdrant database")
        
        # Create test collection
        collection_name = "test_collection"
        
        # Check if collection exists and recreate
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name in collection_names:
            client.delete_collection(collection_name)
            print(f"✅ Deleted existing collection '{collection_name}'")
        
        # Create new collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )
        print(f"✅ Created new collection '{collection_name}'")
        
        # Add vectors
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=1,
                    vector=[0.1, 0.2, 0.3, 0.4],
                    payload={"name": "item1", "category": "A"}
                ),
                models.PointStruct(
                    id=2,
                    vector=[0.2, 0.3, 0.4, 0.5],
                    payload={"name": "item2", "category": "B"}
                ),
                models.PointStruct(
                    id=3,
                    vector=[0.3, 0.4, 0.5, 0.6],
                    payload={"name": "item3", "category": "A"}
                ),
            ]
        )
        print(f"✅ Added 3 vectors to collection")
        
        # Search for similar vectors
        results = client.search(
            collection_name=collection_name,
            query_vector=[0.1, 0.2, 0.3, 0.4],
            limit=3
        )
        
        print(f"✅ Successfully searched for similar vectors")
        print("\nSearch results:")
        for result in results:
            print(f"  ID: {result.id}, Score: {result.score:.4f}, Name: {result.payload['name']}")
        
        # Test filtered search
        filtered_results = client.search(
            collection_name=collection_name,
            query_vector=[0.1, 0.2, 0.3, 0.4],
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="category",
                        match=models.MatchValue(value="A")
                    )
                ]
            ),
            limit=3
        )
        
        print("\nFiltered search results (category = A):")
        for result in filtered_results:
            print(f"  ID: {result.id}, Score: {result.score:.4f}, Name: {result.payload['name']}")
            
        return True
        
    except Exception as e:
        print(f"❌ Vector operations test failed: {e}")
        return False

def test_real_embeddings():
    """Test with real text embeddings using SentenceTransformer"""
    print("\nTesting with real text embeddings...")
    
    try:
        # Load or download model
        print("Loading embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Model loaded successfully")
        
        # Create sample texts
        texts = [
            "Ergonomic office chair with lumbar support",
            "Mechanical keyboard with RGB lighting",
            "Wireless mouse with long battery life",
            "Standing desk for better posture",
            "Noise-cancelling headphones for focus"
        ]
        
        # Generate embeddings
        print("Generating embeddings for sample texts...")
        embeddings = model.encode(texts)
        print(f"✅ Generated embeddings with shape: {embeddings.shape}")
        
        # Save embeddings for later use
        np.save(LOCAL_EMBEDDINGS_PATH, embeddings)
        print(f"✅ Saved embeddings to {LOCAL_EMBEDDINGS_PATH}")
        
        # Create client with persistent storage
        client = QdrantClient(path=LOCAL_QDRANT_PATH)
        
        # Create real embedding collection
        collection_name = "product_embeddings"
        
        # Check if collection exists and recreate
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name in collection_names:
            client.delete_collection(collection_name)
        
        # Create new collection with appropriate vector size
        vector_size = embeddings.shape[1]  # Get dimension from actual embeddings
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"✅ Created collection '{collection_name}' with vector size {vector_size}")
        
        # Add vectors with product information
        products = [
            {"name": "Ergonomic Chair", "category": "Furniture", "price": 299.99},
            {"name": "Mechanical Keyboard", "category": "Electronics", "price": 129.99},
            {"name": "Wireless Mouse", "category": "Electronics", "price": 49.99},
            {"name": "Standing Desk", "category": "Furniture", "price": 399.99},
            {"name": "Noise-Cancelling Headphones", "category": "Electronics", "price": 249.99}
        ]
        
        points = [
            models.PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload=product
            )
            for idx, (embedding, product) in enumerate(zip(embeddings, products))
        ]
        
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"✅ Added {len(points)} product embeddings to collection")
        
        # Test semantic search
        print("\nTesting semantic search...")
        
        # Generate query embedding
        query_text = "comfortable seating for my office"
        query_embedding = model.encode([query_text])[0]
        
        # Search
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=3
        )
        
        print(f"Search results for: '{query_text}'")
        for result in search_results:
            print(f"  Score: {result.score:.4f} - {result.payload['name']} - ${result.payload['price']}")
        
        # Try another query
        print()
        query_text = "computer input devices"
        query_embedding = model.encode([query_text])[0]
        
        # Search
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=3
        )
        
        print(f"Search results for: '{query_text}'")
        for result in search_results:
            print(f"  Score: {result.score:.4f} - {result.payload['name']} - ${result.payload['price']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Real embeddings test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests to verify Qdrant functionality"""
    print("=== QDRANT FUNCTIONALITY TEST ===\n")
    
    # Check if Qdrant is installed properly
    if not check_qdrant_installation():
        print("\n❌ Qdrant installation test failed. Please check your installation.")
        return
    
    # Test basic vector operations
    if not test_vector_operations():
        print("\n❌ Basic vector operations test failed.")
        return
    
    # Test with real embeddings
    if not test_real_embeddings():
        print("\n❌ Real embeddings test failed.")
        return
    
    print("\n✅ All tests passed! Qdrant is working properly on your system.")
    print(f"\nData is stored in: {os.path.abspath(LOCAL_QDRANT_PATH)}")

if __name__ == "__main__":
    main()
