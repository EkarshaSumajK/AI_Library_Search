"""
Quick test script to verify RAG system is working
"""
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

def test_rag():
    print("🧪 Testing RAG System...")
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False)
    )
    
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    try:
        collection = client.get_collection(
            name="nu_library_books",
            embedding_function=embedding_function
        )
        
        # Get collection stats
        count = collection.count()
        print(f"✅ Collection found with {count} documents")
        
        # Test search
        print("\n🔍 Testing search for 'research methodology'...")
        results = collection.query(
            query_texts=["research methodology"],
            n_results=5,
            include=['documents', 'metadatas', 'distances']
        )
        
        print(f"✅ Found {len(results['documents'][0])} results")
        
        # Display first result
        if results['documents'][0]:
            print("\n📖 First result:")
            print(f"   Title: {results['metadatas'][0][0].get('title', 'N/A')}")
            print(f"   Author: {results['metadatas'][0][0].get('author', 'N/A')}")
            print(f"   Chunk Type: {results['metadatas'][0][0].get('chunk_type', 'N/A')}")
            print(f"   Relevance Score: {1 - results['distances'][0][0]:.3f}")
        
        print("\n✅ RAG system is working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_rag()
