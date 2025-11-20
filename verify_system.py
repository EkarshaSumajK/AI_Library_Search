"""
Comprehensive system verification script
Checks all components of the RAG system
"""
import os
import sys
from pathlib import Path

def check_files():
    """Check if all required files exist"""
    print("📁 Checking required files...")
    
    required_files = [
        'app.py',
        'data_ingestion.py',
        'new_data.csv',
        'requirements.txt',
        'test_rag.py'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - MISSING")
            missing_files.append(file)
    
    return len(missing_files) == 0

def check_csv_data():
    """Check CSV data structure"""
    print("\n📊 Checking CSV data...")
    
    try:
        import csv
        with open('new_data.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            books = list(reader)
            
        print(f"   ✅ Found {len(books)} books in CSV")
        
        # Check required columns
        required_columns = ['Title', 'Author(s)', 'Description', 'Link', 'Keywords']
        first_book = books[0]
        
        missing_columns = []
        for col in required_columns:
            if col in first_book:
                print(f"   ✅ Column '{col}' exists")
            else:
                print(f"   ❌ Column '{col}' - MISSING")
                missing_columns.append(col)
        
        return len(missing_columns) == 0
        
    except Exception as e:
        print(f"   ❌ Error reading CSV: {e}")
        return False

def check_chromadb():
    """Check if ChromaDB exists and has data"""
    print("\n💾 Checking ChromaDB...")
    
    if not os.path.exists('chroma_db'):
        print("   ❌ chroma_db directory not found")
        print("   💡 Run: python data_ingestion.py")
        return False
    
    try:
        import chromadb
        from chromadb.config import Settings
        from chromadb.utils import embedding_functions
        
        client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        collection = client.get_collection(
            name="nu_library_books",
            embedding_function=embedding_function
        )
        
        count = collection.count()
        print(f"   ✅ ChromaDB found with {count} documents")
        
        if count < 50:
            print(f"   ⚠️  Warning: Expected ~82 documents, found {count}")
            print("   💡 Consider re-running: python data_ingestion.py")
        
        return count > 0
        
    except Exception as e:
        print(f"   ❌ Error accessing ChromaDB: {e}")
        print("   💡 Run: python data_ingestion.py")
        return False

def check_secrets():
    """Check if secrets are configured"""
    print("\n🔐 Checking secrets configuration...")
    
    secrets_path = Path('.streamlit/secrets.toml')
    
    if not secrets_path.exists():
        print("   ❌ .streamlit/secrets.toml not found")
        print("   💡 Create file with: GOOGLE_API_KEY = \"your-key-here\"")
        return False
    
    try:
        with open(secrets_path, 'r') as f:
            content = f.read()
            
        if 'GOOGLE_API_KEY' in content:
            print("   ✅ GOOGLE_API_KEY found in secrets.toml")
            return True
        else:
            print("   ❌ GOOGLE_API_KEY not found in secrets.toml")
            print("   💡 Add: GOOGLE_API_KEY = \"your-key-here\"")
            return False
            
    except Exception as e:
        print(f"   ❌ Error reading secrets: {e}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'streamlit',
        'chromadb',
        'google.generativeai',
        'sentence_transformers'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print("\n   💡 Install missing packages:")
        print("   pip install -r requirements.txt")
    
    return len(missing_packages) == 0

def run_quick_test():
    """Run a quick search test"""
    print("\n🧪 Running quick search test...")
    
    try:
        import chromadb
        from chromadb.config import Settings
        from chromadb.utils import embedding_functions
        
        client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        collection = client.get_collection(
            name="nu_library_books",
            embedding_function=embedding_function
        )
        
        # Test search
        results = collection.query(
            query_texts=["research methodology"],
            n_results=3,
            include=['documents', 'metadatas', 'distances']
        )
        
        if results['documents'][0]:
            print(f"   ✅ Search working - found {len(results['documents'][0])} results")
            print(f"   📖 Top result: {results['metadatas'][0][0].get('title', 'N/A')}")
            return True
        else:
            print("   ❌ Search returned no results")
            return False
            
    except Exception as e:
        print(f"   ❌ Search test failed: {e}")
        return False

def main():
    """Run all verification checks"""
    print("=" * 60)
    print("🔍 NU eLibrary RAG System Verification")
    print("=" * 60)
    
    checks = {
        'Files': check_files(),
        'CSV Data': check_csv_data(),
        'Dependencies': check_dependencies(),
        'ChromaDB': check_chromadb(),
        'Secrets': check_secrets(),
        'Search Test': run_quick_test()
    }
    
    print("\n" + "=" * 60)
    print("📊 Verification Summary")
    print("=" * 60)
    
    for check_name, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:.<40} {status}")
    
    all_passed = all(checks.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All checks passed! System is ready to use.")
        print("\n🚀 Next steps:")
        print("   1. Run: streamlit run app.py")
        print("   2. Open browser to http://localhost:8501")
        print("   3. Try keyword search: 'research methodology'")
        print("   4. Try chatbot: 'I need books about dissertation writing'")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\n💡 Common fixes:")
        print("   - Missing files: Check you're in the correct directory")
        print("   - Missing ChromaDB: Run 'python data_ingestion.py'")
        print("   - Missing secrets: Create '.streamlit/secrets.toml'")
        print("   - Missing packages: Run 'pip install -r requirements.txt'")
    
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
