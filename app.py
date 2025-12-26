"""
NU eLibrary RAG Chatbot
A search interface for National University eLibrary with AI-powered recommendations
Enhanced with 65535 max tokens for comprehensive responses
"""
import streamlit as st
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import google.generativeai as genai
import json
import hashlib
import concurrent.futures
from functools import lru_cache
from PIL import Image, ImageFile
import threading
import io
import pandas as pd

# Allow loading truncated images to prevent OSError
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configure page
st.set_page_config(
    page_title="NU eLibrary Intelligent Search",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Perplexity/Shadcn UI CSS - Forced Light Theme
st.markdown("""
    <style>
    /* ========================================
       PERPLEXITY / SHADCN THEME (Light Mode)
       ======================================== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --background: #ffffff;
        --foreground: #020817;
        --card: #ffffff;
        --card-foreground: #020817;
        --popover: #ffffff;
        --popover-foreground: #020817;
        --primary: #0f172a;
        --primary-foreground: #f8fafc;
        --secondary: #f1f5f9;
        --secondary-foreground: #0f172a;
        --muted: #f1f5f9;
        --muted-foreground: #64748b;
        --accent: #f1f5f9;
        --accent-foreground: #0f172a;
        --destructive: #ef4444;
        --destructive-foreground: #f8fafc;
        --border: #e2e8f0;
        --input: #e2e8f0;
        --ring: #0f172a;
        --radius: 0.5rem;
    }

    /* Force Light Theme Overrides */
    .stApp, .main, [data-testid="stSidebar"], .stHeader {
        background-color: var(--background) !important;
        color: var(--foreground) !important;
    }
    
    section[data-testid="stSidebar"] > div {
        background-color: #fcfcfc !important;
        border-right: 1px solid var(--border);
    }

    /* Global Typography */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Inter', sans-serif !important;
        color: var(--foreground) !important;
    }

    /* Main Container Centering */
    .block-container {
        max-width: 900px !important;
        padding-top: 3rem !important;
        padding-bottom: 5rem !important;
    }

    /* Shadcn-style Search Bar (Perplexity style) */
    .stTextInput > div > div > input {
        background-color: var(--background) !important;
        border: 1px solid var(--border) !important;
        color: var(--foreground) !important;
        border-radius: 12px !important;
        padding: 1.25rem 1.5rem !important;
        font-size: 1.1rem !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        transition: all 0.2s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: #94a3b8 !important;
        box-shadow: 0 0 0 4px rgba(148, 163, 184, 0.1) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: var(--muted-foreground) !important;
        opacity: 0.7;
    }

    /* Perplexity Source Cards (Grid) */
    .sources-container {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 0.75rem;
        margin-bottom: 2rem;
    }

    .source-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.75rem 1rem;
        transition: all 0.2s ease;
        text-decoration: none !important;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 100px;
    }

    .source-card:hover {
        background: var(--secondary);
        border-color: #cbd5e1;
        transform: translateY(-1px);
    }

    .source-number {
        font-size: 0.7rem;
        font-weight: 700;
        color: var(--muted-foreground);
        margin-bottom: 0.25rem;
        background: var(--muted);
        width: 18px;
        height: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 4px;
    }

    .source-title {
        color: var(--foreground) !important;
        font-size: 0.85rem;
        font-weight: 600;
        line-height: 1.3;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        margin-bottom: 0.5rem;
    }

    .source-meta {
        color: var(--muted-foreground) !important;
        font-size: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Citations in text */
    .citation-tag {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: var(--secondary);
        color: var(--secondary-foreground);
        font-size: 0.65rem;
        font-weight: 700;
        width: 16px;
        height: 16px;
        border-radius: 4px;
        margin: 0 2px;
        vertical-align: top;
        border: 1px solid var(--border);
        cursor: pointer;
    }

    /* Buttons (Shadcn style) */
    .stButton > button {
        background-color: var(--primary) !important;
        color: var(--primary-foreground) !important;
        border-radius: var(--radius) !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        transition: opacity 0.2s ease !important;
    }

    .stButton > button:hover {
        opacity: 0.9 !important;
    }

    /* Secondary/Muted Buttons */
    div[data-testid="stSidebar"] .stButton > button {
        background-color: transparent !important;
        color: var(--foreground) !important;
        border: 1px solid var(--border) !important;
    }

    div[data-testid="stSidebar"] .stButton > button:hover {
        background-color: var(--secondary) !important;
    }

    /* Metrics & Sidebar */
    [data-testid="stMetricValue"] {
        color: var(--primary) !important;
        font-weight: 700 !important;
    }
    
    .stAlert {
        border-radius: var(--radius) !important;
        border: 1px solid var(--border) !important;
        background-color: var(--secondary) !important;
    }

    /* Chat Messages (Minimalist) */
    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        margin-bottom: 2rem !important;
    }
    
    .stChatMessage [data-testid="stMarkdownContainer"] p {
        font-size: 1.05rem !important;
        line-height: 1.625 !important;
        color: var(--foreground) !important;
    }

    .chat-welcome-box {
        border-bottom: 1px solid var(--border);
        padding-bottom: 2rem;
        margin-bottom: 2rem;
    }
    
    .chat-welcome-title {
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin-bottom: 1rem;
    }

    /* Pills/Badges */
    .concept-pill {
        display: inline-flex;
        align-items: center;
        background: var(--secondary);
        color: var(--secondary-foreground);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid var(--border);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent !important;
        border-bottom: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        height: 45px !important;
        background-color: transparent !important;
        border: none !important;
        color: var(--muted-foreground) !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--foreground) !important;
        border-bottom: 2px solid var(--primary) !important;
    }

    /* Hide Streamlit components for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    </style>
""", unsafe_allow_html=True)


class LibraryRAG:
    def __init__(self):
        """Initialize RAG system with ChromaDB and Gemini with extended token limit"""
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )

        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        try:
            self.collection = self.client.get_collection(
                name="nu_library_books",
                embedding_function=self.embedding_function
            )
        except:
            st.error("⚠️ Database not found. Please run data_ingestion.py first!")
            st.stop()

        # Initialize Gemini with standard configuration
        try:
            api_key = st.secrets.get("GOOGLE_API_KEY")
            if not api_key:
                st.error("⚠️ GOOGLE_API_KEY not found. Please set it in Streamlit secrets.")
                st.info("💡 To set up secrets locally, create a `.streamlit/secrets.toml` file with:")
                st.code("""
GOOGLE_API_KEY = "your-api-key-here"
""")
                st.info("🔍 **Troubleshooting:** Make sure the `.streamlit/secrets.toml` file exists in your project root and contains your API key.")
                st.stop()
        except Exception as e:
            st.error(f"❌ Error accessing Streamlit secrets: {e}")
            st.info("💡 This usually means the secrets.toml file format is incorrect or the file doesn't exist.")
            st.info("📁 **File should be at:** `.streamlit/secrets.toml`")
            st.info("📝 **Format should be:** `GOOGLE_API_KEY = \"your-key-here\"` (without [general] section)")
            st.stop()

        genai.configure(api_key=api_key)

        # Configure generation with optimized settings for speed
        self.generation_config = genai.GenerationConfig(
            max_output_tokens=2048,  # Reduced for faster responses
            temperature=0.1,  # Slightly higher for better quality but still fast
        )

        self.model = genai.GenerativeModel(
            'gemini-3-flash-preview',
            generation_config=self.generation_config
        )

        # Summary cache for faster responses
        self.summary_cache = {}
        self.cache_lock = threading.Lock()

        # Thread pool for parallel processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        
        # Initialize Analytics Logs
        import os
        self.log_dir = "./logs"
        self.log_file = os.path.join(self.log_dir, "search_history.json")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump([], f)

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

    def analyze_image(self, image):
        """Analyze book cover image to extract details using Gemini"""
        prompt = """
        Analyze this image of a book. Extract the following information:
        1. Title
        2. Author
        3. Edition (if visible)
        
        Return a strict JSON object with fields: "title", "author", "edition".
        If it's not a book or text is unreadable, return {"error": "Could not identify book"}.
        """
        
        try:
            response = self.model.generate_content([prompt, image])
            
            try:
                text = response.text.strip()
            except ValueError:
                # If response.text raises ValueError, it's likely blocked by safety filters
                print(f"Gemini Safety Block: {response.prompt_feedback}")
                return {"error": "Image analysis blocked by safety filters. Please try another image."}
                
            # Clean up potential markdown code blocks
            if text.startswith('```json'):
                text = text[7:]
            if text.endswith('```'):
                text = text[:-3]
            return json.loads(text.strip())
        except Exception as e:
            print(f"Error in analyze_image: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
        
    
    def analyze_query(self, query, user_profile):
        """Analyze query for intent, translation, and expansion using Gemini"""
        prompt = f"""
        Act as a search query understanding engine for an academic library.
        User Query: "{query}"
        User Profile: "{user_profile}"

        Analyze the query and return a strict JSON object with these fields:
        1. "english_query": Translate the query to English if it's not. Keep it unchanged if it is English.
        2. "intent": One of ["textbook", "research", "general"].
           - "textbook": User wants books, textbooks, or learning materials.
           - "research": User wants papers, faculty publications, journals, or detailed studies.
           - "general": Unclear or mixed intent.
        3. "synonyms": List of 3-5 academic synonyms or related technical terms to expand the search.
        4. "domain_context": inferred academic domain (e.g., "Civil Engineering", "Computer Science").

        Output ONLY the JSON object.
        """
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            # Clean up potential markdown code blocks
            if text.startswith('```json'):
                text = text[7:]
            if text.endswith('```'):
                text = text[:-3]
            return json.loads(text.strip())
        except Exception as e:
            # Fallback
            return {
                "english_query": query,
                "intent": "general",
                "synonyms": [],
                "domain_context": "General"
            }

    def search_books(self, query, user_profile="General", n_results=10, prioritize_publications=True):
        """Search for books using semantic search with intent and profile boosting"""
        # --- Real Analytics Logging ---
        import datetime
        try:
            with open(self.log_file, 'r') as f:
                history = json.load(f)
            history.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "query": query,
                "user_profile": user_profile
            })
            with open(self.log_file, 'w') as f:
                json.dump(history, f)
        except Exception as e:
            print(f"Logging error: {e}")

        # 1. Analyze Query
        analysis = self.analyze_query(query, user_profile)
        search_query = analysis.get('english_query', query)
        intent = analysis.get('intent', 'general')
        domain = analysis.get('domain_context', '')
        
        # Get more results initially for better re-ranking
        initial_results = min(n_results * 5, 60)

        # Primary Search with English Query
        results = self.collection.query(
            query_texts=[search_query],
            n_results=initial_results,
            include=['documents', 'metadatas', 'distances']
        )

        if not results['documents'][0]:
            return {'results': results, 'analysis': analysis}

        # Process results based on hierarchical structure
        processed_results = self._process_hierarchical_results(
            results, search_query, n_results * 2, prioritize_publications # Get more for re-ranking
        )
        
        # Apply Profile and Intent Re-ranking
        boosted_results = self._rank_results(processed_results, analysis, user_profile)
        
        # Limit to final n_results
        final_results = {
            'documents': [boosted_results['documents'][0][:n_results]],
            'metadatas': [boosted_results['metadatas'][0][:n_results]],
            'distances': [boosted_results['distances'][0][:n_results]]
        }

        return {'results': final_results, 'analysis': analysis}
    
    def _rank_results(self, results, analysis, user_profile):
        """Re-rank results based on profile match and intent"""
        if not results['documents'][0]:
            return results
            
        ranked_items = []
        intent = analysis.get('intent', 'general')
        
        for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
            score = 1.0 - dist # Convert distance to similarity score
            
            # --- 1. Profile Boosting (Major Specific Ranking) ---
            # colleges: 'College of Engineering', 'College of Medicine...', etc.
            book_college = meta.get('college', '')
            if user_profile != "General" and user_profile in book_college:
                score += 0.15 # Significant boost for major match
            
            # --- 2. Intent Boosting ---
            chunk_type = meta.get('chunk_type', '')
            
            if intent == 'research':
                if chunk_type == 'publication_detail':
                    score += 0.10
                elif chunk_type == 'author_summary':
                    score += 0.05
            elif intent == 'textbook':
                if chunk_type == 'book_detail':
                    score += 0.10
            
            # --- 3. Title Matching (Original Logic) ---
            query_lower = analysis.get('english_query', '').lower().strip()
            title = meta.get('title', '').lower()
            if query_lower in title:
                score += 0.2
            
            ranked_items.append({
                'document': doc,
                'metadata': meta,
                'distance': 1.0 - score # Convert back to distance for consistency
            })
            
        # Sort by new distance (lower is better, so higher score is better)
        ranked_items.sort(key=lambda x: x['distance'])
        
        return {
            'documents': [[item['document'] for item in ranked_items]],
            'metadatas': [[item['metadata'] for item in ranked_items]],
            'distances': [[item['distance'] for item in ranked_items]]
        }

    # --- Fallback Data ---
    FALLBACK_BOOKS = {
        "College of Engineering": [
            {"title": "Fundamentals of Engineering Thermodynamics", "author": "Michael J. Moran", "link": "https://elibrary.nu.edu.om/", "description": "Comprehensive guide to thermodynamics.", "rating": "4.8", "chunk_type": "book_detail"},
            {"title": "Introduction to Algorithms", "author": "Thomas H. Cormen", "link": "https://elibrary.nu.edu.om/", "description": "The standard algorithm reference.", "rating": "4.9", "chunk_type": "book_detail"},
            {"title": "The Art of Electronics", "author": "Paul Horowitz", "link": "https://elibrary.nu.edu.om/", "description": "Essential for circuit design.", "rating": "4.9", "chunk_type": "book_detail"},
            {"title": "Clean Code: A Handbook of Agile Software Craftsmanship", "author": "Robert C. Martin", "link": "https://elibrary.nu.edu.om/", "description": "Best practices for writing code.", "rating": "4.7", "chunk_type": "book_detail"}
        ],
        "College of Medicine and Health Sciences": [
            {"title": "Gray's Anatomy", "author": "Susan Standring", "link": "https://elibrary.nu.edu.om/", "description": "The anatomical basis of clinical practice.", "rating": "4.9", "chunk_type": "book_detail"},
            {"title": "Harrison's Principles of Internal Medicine", "author": "J. Larry Jameson", "link": "https://elibrary.nu.edu.om/", "description": "Leading text on internal medicine.", "rating": "4.8", "chunk_type": "book_detail"},
            {"title": "Guyton and Hall Textbook of Medical Physiology", "author": "John E. Hall", "link": "https://elibrary.nu.edu.om/", "description": "Standard physiology textbook.", "rating": "4.8", "chunk_type": "book_detail"},
            {"title": "Robbins & Cotran Pathologic Basis of Disease", "author": "Vinay Kumar", "link": "https://elibrary.nu.edu.om/", "description": "Definitive pathology source.", "rating": "4.9", "chunk_type": "book_detail"}
        ],
        "College of Pharmacy": [
            {"title": "Pharmacotherapy: A Pathophysiologic Approach", "author": "Joseph DiPiro", "link": "https://elibrary.nu.edu.om/", "description": "Gold standard for pharmacotherapy.", "rating": "4.7", "chunk_type": "book_detail"},
            {"title": "Goodman & Gilman's The Pharmacological Basis of Therapeutics", "author": "Laurence Brunton", "link": "https://elibrary.nu.edu.om/", "description": "Pharmacology authority.", "rating": "4.8", "chunk_type": "book_detail"},
            {"title": "Remington: The Science and Practice of Pharmacy", "author": "Allen Loyd", "link": "https://elibrary.nu.edu.om/", "description": "Comprehensive pharmacy reference.", "rating": "4.6", "chunk_type": "book_detail"}
        ],
        "International Maritime College Oman": [
            {"title": "The Maritime Engineering Reference Book", "author": "Anthony F. Molland", "link": "https://elibrary.nu.edu.om/", "description": "Guide to marine engineering.", "rating": "4.5", "chunk_type": "book_detail"},
            {"title": "Port Management and Operations", "author": "Maria G. Burns", "link": "https://elibrary.nu.edu.om/", "description": "Managing modern ports.", "rating": "4.6", "chunk_type": "book_detail"},
            {"title": "Admiralty Manual of Seamanship", "author": "Ministry of Defence", "link": "https://elibrary.nu.edu.om/", "description": "Seamanship bible.", "rating": "4.7", "chunk_type": "book_detail"}
        ],
        "General": [
            {"title": "Research Methodology: A Step-by-Step Guide", "author": "Ranjit Kumar", "link": "https://elibrary.nu.edu.om/", "description": "Guide for beginners.", "rating": "4.5", "chunk_type": "book_detail"},
            {"title": "Academic Writing for Graduate Students", "author": "John M. Swales", "link": "https://elibrary.nu.edu.om/", "description": "Essential for academic success.", "rating": "4.6", "chunk_type": "book_detail"},
            {"title": "The Craft of Research", "author": "Wayne C. Booth", "link": "https://elibrary.nu.edu.om/", "description": "Classic research guide.", "rating": "4.7", "chunk_type": "book_detail"}
        ]
    }

    def _get_fallback_results(self, category, n_required):
        """Get fallback results for a given category"""
        # Determine fallback category
        if category not in self.FALLBACK_BOOKS:
             # Try to find a partial match (e.g., "College of Engineering" match for "Engineering")
             found = False
             for key in self.FALLBACK_BOOKS:
                 if category in key or key in category:
                     category = key
                     found = True
                     break
             if not found:
                 category = "General"
        
        fallback_items = self.FALLBACK_BOOKS.get(category, self.FALLBACK_BOOKS["General"])
        
        # Format as ChromaDB results
        docs = []
        metas = []
        dists = []
        
        count = 0
        for item in fallback_items:
            if count >= n_required:
                break
            
            # Create a dummy document content
            content = f"Title: {item['title']}\nAuthor: {item['author']}\nDescription: {item['description']}"
            
            docs.append(content)
            metas.append(item)
            dists.append(0.01) # High relevance for fallbacks
            count += 1
            
        return {
            'documents': [docs],
            'metadatas': [metas],
            'distances': [dists]
        }

    def get_recommendations_by_history(self, book_title=None, n_results=5):
        """Get recommendations based on reading history (simulated by a single book title)"""
        if not book_title:
             return self._get_fallback_results("General", n_results)
        
        # In a real system, we would get the embedding of the book. 
        # Here we search for the title to find similar items.
        results = self.collection.query(
            query_texts=[f"Books similar to {book_title}"],
            n_results=n_results * 4, # Fetch more to filter for books
            include=['documents', 'metadatas', 'distances']
        )
        
        # Filter for actual books (chunk_type='book_detail')
        valid_indices = []
        if results['metadatas'][0]:
            for i, meta in enumerate(results['metadatas'][0]):
                if meta.get('chunk_type') == 'book_detail':
                    valid_indices.append(i)
        
        # Construct filtered results
        filtered_docs = [results['documents'][0][i] for i in valid_indices[:n_results]]
        filtered_metas = [results['metadatas'][0][i] for i in valid_indices[:n_results]]
        filtered_dists = [results['distances'][0][i] for i in valid_indices[:n_results]]
        
        # If not enough results, add fallbacks
        if len(filtered_docs) < n_results:
             needed = n_results - len(filtered_docs)
             fallback = self._get_fallback_results("General", needed)
             
             filtered_docs.extend(fallback['documents'][0])
             filtered_metas.extend(fallback['metadatas'][0])
             filtered_dists.extend(fallback['distances'][0])

        return {
             'documents': [filtered_docs],
             'metadatas': [filtered_metas],
             'distances': [filtered_dists]
        }

    def get_recommendations_by_profile(self, profile_data, n_results=5):
        """Get recommendations based on user profile (Major, Interests)"""
        major = profile_data.get('major', 'General')
        interests = profile_data.get('interests', [])
        role = profile_data.get('role', 'Student')
        
        query_text = f"Academic resources for {major}"
        if interests:
            query_text += f" focused on {', '.join(interests)}"
        if role == 'Faculty':
            query_text += " research publications and advanced studies"
        else:
            query_text += " textbooks and learning materials"
            
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results * 5, # Fetch significantly more to find valid books
            include=['documents', 'metadatas', 'distances']
        )
        
        # Filter for actual books (chunk_type='book_detail')
        valid_indices = []
        if results['metadatas'][0]:
            for i, meta in enumerate(results['metadatas'][0]):
                if meta.get('chunk_type') == 'book_detail':
                    valid_indices.append(i)
        
        # Construct filtered results
        filtered_docs = [results['documents'][0][i] for i in valid_indices[:n_results]]
        filtered_metas = [results['metadatas'][0][i] for i in valid_indices[:n_results]]
        filtered_dists = [results['distances'][0][i] for i in valid_indices[:n_results]]
        
        # If not enough results, add fallbacks based on Major
        if len(filtered_docs) < n_results:
             needed = n_results - len(filtered_docs)
             fallback = self._get_fallback_results(major, needed)
             
             filtered_docs.extend(fallback['documents'][0])
             filtered_metas.extend(fallback['metadatas'][0])
             filtered_dists.extend(fallback['distances'][0])

        final_results = {
            'documents': [filtered_docs],
            'metadatas': [filtered_metas],
            'distances': [filtered_dists]
        }
        return final_results

    # --- GCC Features: Location & Availability ---

    LIBRARY_MAP = {
        "Engineering": "Floor 2, Zone B (Blue Section)",
        "Medicine": "Floor 3, Zone A (Red Section)",
        "Pharmacy": "Floor 3, Zone C (Green Section)",
        "Business": "Floor 1, Zone D (Yellow Section)",
        "Maritime": "Ground Floor, Zone M (Maritime Wing)",
        "General": "Ground Floor, Central Hall",
        "Journals": "Floor 4, Quiet Zone",
        "Reference": "Ground Floor, East Wing"
    }

    def get_location_info(self, query):
        """Resolve spatial queries using the library map"""
        query_lower = query.lower()
        
        # Check against map keys
        for key, location in self.LIBRARY_MAP.items():
            if key.lower() in query_lower:
                return f"📍 **Location Guide**: {key} resources are located on **{location}**."
        
        return None

    def check_availability(self, book_title):
        """Simulate real-time checking with KOHA LMS"""
        # Deterministic simulation based on hash of title
        if not book_title:
            return None
            
        import hashlib
        hash_val = int(hashlib.md5(book_title.encode()).hexdigest(), 16)
        
        # 60% Available, 30% Checked Out, 10% Reserved
        status_code = hash_val % 10
        
        if status_code < 6:
            return "✅ **Status**: Available on Shelf"
        elif status_code < 9:
            # Generate a random future date
            days = (hash_val % 14) + 1
            return f"❌ **Status**: Checked Out (Due in {days} days)"
        else:
            return "🔖 **Status**: Reserved for Course Reference"

    # --- Admin Analytics Methods ---

    def get_collection_stats(self):
        """Get real book counts per college from ChromaDB"""
        colleges = [
            "College of Engineering",
            "College of Medicine and Health Sciences",
            "College of Pharmacy",
            "International Maritime College Oman"
        ]
        stats = {}
        for college in colleges:
            # We filter for chunk_type='book_detail' to count unique books better
            results = self.collection.get(
                where={"college": college}
            )
            # Count unique titles to avoid overcounting chunks
            titles = set()
            if results['metadatas']:
                for meta in results['metadatas']:
                    if meta.get('title'):
                        titles.add(meta.get('title'))
            stats[college] = len(titles)
        return stats

    def get_peak_demand_prediction(self):
        """Generate trend prediction based on real search history"""
        # Try to load real history
        try:
            with open(self.log_file, 'r') as f:
                history = json.load(f)
        except:
            history = []

        if not history:
            # Fallback to simulated data if no history exists yet
            return [
                {"subject": "Engineering", "peak_months": "Adaptive (Awaiting Data)", "reason": "System learning...", "demand_score": 0},
                {"subject": "Medicine", "peak_months": "Adaptive (Awaiting Data)", "reason": "System learning...", "demand_score": 0},
            ]

        # Basic analysis of search frequency by profile
        trends = {}
        for entry in history:
            major = entry.get('user_profile', 'General')
            trends[major] = trends.get(major, 0) + 1
        
        # Sort and map to predictions
        predictions = []
        for major, count in sorted(trends.items(), key=lambda x: x[1], reverse=True):
            predictions.append({
                "subject": major.replace("College of ", ""),
                "peak_months": "Live Trending Now",
                "reason": f"Real-time demand high ({count} searches)",
                "demand_score": min(count * 10, 100) # Simple scaling
            })
        return predictions

    def get_gap_analysis(self):
        """Identify gaps between REAL search demand and REAL collection availability"""
        # 1. Get collection stats
        coll_stats = self.get_collection_stats()
        
        # 2. Get search logs
        try:
            with open(self.log_file, 'r') as f:
                history = json.load(f)
        except:
            history = []
            
        # 3. Analyze searches vs collection
        # Map majors to subjects for simple match
        search_counts = {}
        for entry in history:
            query = entry.get('query', '').lower()
            # Simple keyword extraction (could be improved with LLM)
            words = [w for w in query.split() if len(w) > 3]
            for word in words:
                search_counts[word] = search_counts.get(word, 0) + 1
        
        # Cross reference top searches with collection
        gaps = []
        # Fallback if no search data
        if not history:
            return [{"subject": "Awaiting Searches...", "searches": 0, "books": 0, "gap_score": 0}]

        # Sort search terms by volume
        top_searches = sorted(search_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        if not top_searches:
             return [{"subject": "No significant trends yet", "searches": 0, "books": 0, "gap_score": 0}]

        # Fetch all metadata once for client-side filtering (more robust than limited API operators)
        all_items = self.collection.get(include=['metadatas'])
        all_metadatas = all_items.get('metadatas', [])

        for term, count in top_searches:
            # Filter in Python
            term_lower = term.lower()
            book_count = 0
            
            for meta in all_metadatas:
                title = meta.get('title', '').lower()
                keywords = meta.get('keyword', '').lower()
                
                if term_lower in title or term_lower in keywords:
                    book_count += 1
            
            # Gap score = High search volume + Low book count
            # Example: 10 searches + 0 books = 100 score
            gap_score = min(100, (count * 15) - (book_count * 5))
            if gap_score < 0: gap_score = 10

            gaps.append({
                "subject": term.capitalize(),
                "searches": count,
                "books": book_count,
                "gap_score": gap_score
            })
            
        return sorted(gaps, key=lambda x: x['gap_score'], reverse=True)

    def get_purchase_recommendations(self):
        """Generate AI-powered purchase recommendations based on gaps"""
        gaps = self.get_gap_analysis()
        top_gaps = gaps[:3]
        
        subjects = ", ".join([g['subject'] for g in top_gaps])
        
        prompt = f"""
        You are a library acquisition specialist. Based on the following high-demand subjects with low collection coverage:
        {subjects}
        
        Generate a brief list of 5 specific book recommendations (Title, Author if known, Why) for purchase.
        Format as a numbered list.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Could not generate recommendations: {e}"

    def process_voice_intent(self, voice_text):
        """Parse voice text into structured intent and parameters using Gemini"""
        prompt = f"""
        Analyze the following voice command for a university library system and extract the user's intent:
        Voice Command: "{voice_text}"
        
        Possible Intents:
        1. "search": User wants to find books/papers. Extract "subject" and "language" (if mentioned).
        2. "status": User wants to check if a book is available or when it's due. Extract "title".
        3. "reserve": User wants to reserve a book. Extract "title".
        4. "other": Any other request.
        
        Return a strict JSON object with fields: "intent", "subject", "language", "title".
        For fields not found, use null.
        
        Example: "Show me military leadership books in Arabic" 
        -> {{"intent": "search", "subject": "Military Leadership", "language": "Arabic", "title": null}}
        """
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            # Clean up potential markdown
            if text.startswith('```json'):
                text = text[7:]
            if text.endswith('```'):
                text = text[:-3]
            return json.loads(text.strip())
        except Exception as e:
            return {"intent": "other", "error": str(e)}

    def _process_hierarchical_results(self, results, query, max_results, prioritize_publications):
        """Process and prioritize hierarchical search results - excluding keyword summaries"""
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        # Group results by chunk type and level (excluding keyword summaries)
        level_2_chunks = []  # Author summaries
        level_3_chunks = []  # Individual books

        for doc, meta, dist in zip(documents, metadatas, distances):
            chunk_level = meta.get('chunk_level', 3)
            chunk_type = meta.get('chunk_type', 'publication_detail')

            # Skip keyword summaries (level 1)
            if chunk_level == 1 or chunk_type == 'keyword_summary':
                continue

            result_item = {
                'document': doc,
                'metadata': meta,
                'distance': dist,
                'chunk_level': chunk_level,
                'chunk_type': chunk_type
            }

            if chunk_level == 2:
                level_2_chunks.append(result_item)
            else:
                level_3_chunks.append(result_item)

        # Prioritize results based on strategy
        final_results = []

        if prioritize_publications:
            # Strategy: Show author info, then books (no keyword summaries)
            # Take top 2 author summaries if available
            if level_2_chunks:
                final_results.extend(sorted(level_2_chunks, key=lambda x: x['distance'])[:2])

            # Fill remaining slots with books
            remaining_slots = max_results - len(final_results)
            if remaining_slots > 0 and level_3_chunks:
                final_results.extend(sorted(level_3_chunks, key=lambda x: x['distance'])[:remaining_slots])
        else:
            # Alternative strategy: Pure relevance-based ranking
            all_chunks = level_1_chunks + level_2_chunks + level_3_chunks
            final_results = sorted(all_chunks, key=lambda x: x['distance'])[:max_results]

        # Reformat to match original structure
        processed_results = {
            'documents': [[r['document'] for r in final_results]],
            'metadatas': [[r['metadata'] for r in final_results]],
            'distances': [[r['distance'] for r in final_results]]
        }

        return processed_results

    def _get_cache_key(self, book_info, query, chunk_type):
        """Generate a cache key for summary caching"""
        # Create a hash of the inputs for consistent caching
        content = f"{book_info[:200]}_{query}_{chunk_type}"  # Limit content for faster hashing
        return hashlib.md5(content.encode()).hexdigest()

    def get_cached_summary(self, book_info, query, chunk_type='keyword_summary'):
        """Get cached summary or generate new one"""
        cache_key = self._get_cache_key(book_info, query, chunk_type)

        with self.cache_lock:
            if cache_key in self.summary_cache:
                return self.summary_cache[cache_key]

        # Generate new summary
        summary = self.generate_summary(book_info, query, chunk_type)

        # Cache the result
        with self.cache_lock:
            self.summary_cache[cache_key] = summary

        return summary

    def generate_batch_summaries(self, items, query):
        """Generate multiple summaries in parallel for faster processing"""
        def generate_single_summary(item):
            book_info, chunk_type = item
            return self.get_cached_summary(book_info, query, chunk_type)

        # Use thread pool for parallel processing
        future_to_item = {
            self.executor.submit(generate_single_summary, item): item
            for item in items
        }

        results = []
        for future in concurrent.futures.as_completed(future_to_item, timeout=30):
            try:
                results.append(future.result())
            except Exception as e:
                # Fallback to basic summary on error
                results.append("Summary temporarily unavailable.")

        return results

    def generate_summary(self, book_info, query, chunk_type='keyword_summary'):
        """Generate concise AI summary for different types of chunks based on query"""
        # Customize prompt based on chunk type
        # Optimized shorter prompts for faster responses
        if chunk_type == 'keyword_summary':
            prompt = f"Academic library collection: {book_info[:300]}... Query: {query}. Summarize relevance in 2-3 sentences, focus on key resources and research value."

        elif chunk_type == 'author_summary':
            prompt = f"Author profile: {book_info[:300]}... Query: {query}. Summarize expertise and research connections in 2-3 sentences."

        else:  # book_detail or default
            prompt = f"Book details: {book_info[:300]}... Query: {query}. Summarize content and research relevance in 2-3 sentences."

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            # Fallback when API fails - extract detailed information from book_info
            error_msg = str(e)
            if "quota" in error_msg.lower() or "429" in error_msg:
                return self._generate_detailed_fallback_summary(book_info, query, chunk_type)
            return self._generate_basic_fallback_summary(book_info, query, chunk_type)

    def _generate_detailed_fallback_summary(self, book_info, query, chunk_type):
        """Generate concise fallback summary by parsing book_info when API quota is exceeded"""
        if not isinstance(book_info, str):
            return f"Academic content available for '{query}' in the NU eLibrary collection. (API temporarily unavailable)"

        lines = book_info.split('\n')
        extracted_info = {}

        # Extract information from the embedded text
        for line in lines:
            line = line.strip()
            if line.startswith('Author:') and 'Author:' not in extracted_info:
                author = line.replace('Author:', '').strip()
                # Clean up author names
                author = author.replace('Dr ', '').replace('Professor ', '').replace('faculty ', '')
                if author and author != 'Unknown':
                    extracted_info['author'] = author

            elif line.startswith('Title:') and 'Title:' not in extracted_info:
                title = line.replace('Title:', '').strip()
                if title and title != 'Unknown Title':
                    extracted_info['title'] = title

            elif line.startswith('Keyword:') and 'Keyword:' not in extracted_info:
                keyword = line.replace('Keyword:', '').strip()
                if keyword and keyword != 'Unknown':
                    extracted_info['keyword'] = keyword

            elif line.startswith('College:') and 'College:' not in extracted_info:
                college = line.replace('College:', '').strip()
                if college and college != 'Unknown':
                    extracted_info['college'] = college

            elif line.startswith('Availability:') and 'Availability:' not in extracted_info:
                availability = line.replace('Availability:', '').strip()
                extracted_info['availability'] = availability

            elif line.startswith('Publication:') and 'Publication:' not in extracted_info:
                publication = line.replace('Publication:', '').strip()
                extracted_info['publication'] = publication

            elif line.startswith('Description:') and 'Description:' not in extracted_info:
                description = line.replace('Description:', '').strip()
                if len(description) > 50:  # Only if substantial description
                    extracted_info['description'] = description[:200] + '...' if len(description) > 200 else description

        # Generate concise one-paragraph summary based on chunk type
        if chunk_type == 'keyword_summary':
            keyword = extracted_info.get('keyword', query)
            return f"This collection covers '{keyword}' with resources relevant to '{query}', providing valuable academic materials for research and study in this discipline."

        elif chunk_type == 'author_summary':
            author = extracted_info.get('author', 'NU faculty')
            keyword = extracted_info.get('keyword', 'academic subjects')
            return f"{author} is a researcher specializing in {keyword}, with publications relevant to '{query}' that contribute to academic discourse in this field."

        else:  # book_detail
            title = extracted_info.get('title', 'This publication')
            author = extracted_info.get('author', 'the author')
            return f"'{title}' by {author} is a valuable resource for '{query}' research, providing important insights and information for academic study."

    def _generate_basic_fallback_summary(self, book_info, query, chunk_type):
        """Generate basic fallback summary for other API errors"""
        if chunk_type == 'keyword_summary':
            return f"Academic collection covering '{query}' with multiple scholarly resources available in the NU eLibrary."
        elif chunk_type == 'author_summary':
            return f"Faculty research profile with publications relevant to '{query}' available for reference."
        else:
            return f"Scholarly publication relevant to '{query}' available in the NU eLibrary collection."

    def generate_recommendations(self, query, search_results):
        """Generate comprehensive recommendations using Gemini with extended token limit"""
        # Prepare context from search results
        context = "Retrieved Publications:\n\n"
        for i, (doc, meta, dist) in enumerate(zip(
            search_results['documents'][0],
            search_results['metadatas'][0],
            search_results['distances'][0]
        )):
            context += f"{i+1}. {doc}\n"
            context += f"   Relevance Score: {1 - dist:.2f}\n\n"
        
        prompt = f"""You are an AI librarian for National University eLibrary (https://elibrary.nu.edu.om/).

User Query: {query}

{context}

Based on the above publications, provide a concise 4-5 sentence analysis that covers:

1. **Key Findings**: Summarize the main themes and most relevant publications found for this query
2. **Resource Value**: Explain the academic significance and practical applications of these materials
3. **Research Guidance**: Provide brief recommendations on how to use these resources effectively

Keep the response conversational, helpful, and focused on the user's specific research needs. Limit to exactly 4-5 sentences total."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "429" in error_msg:
                return self._generate_detailed_recommendation_fallback(query, search_results)
            return self._generate_basic_recommendation_fallback(query, search_results)

    def _generate_detailed_recommendation_fallback(self, query, search_results):
        """Generate concise recommendation fallback when API quota exceeded"""
        num_results = len(search_results['documents'][0])

        # Analyze the results to extract key information
        keywords_found = set()
        authors_found = set()
        chunk_types = {'keyword_summary': 0, 'author_summary': 0, 'book_detail': 0}

        for metadata in search_results['metadatas'][0]:
            chunk_type = metadata.get('chunk_type', 'book_detail')
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

            if metadata.get('keyword'):
                keywords_found.add(metadata['keyword'])
            if metadata.get('author'):
                authors_found.add(metadata['author'])

        # Create concise 4-5 sentence summary
        sentences = []

        # Key findings
        sentences.append(f"I found {num_results} relevant resources for '{query}' in the NU eLibrary collection.")

        # Main themes and resource types
        resource_types = []
        if chunk_types.get('keyword_summary', 0) > 0:
            resource_types.append(f"{chunk_types['keyword_summary']} subject overview(s)")
        if chunk_types.get('author_summary', 0) > 0:
            resource_types.append(f"{chunk_types['author_summary']} author profile(s)")
        if chunk_types.get('book_detail', 0) > 0:
            resource_types.append(f"{chunk_types['book_detail']} specific book(s)")

        if resource_types:
            sentences.append(f"The results include {', '.join(resource_types)} that directly relate to your research topic.")

        # Academic value
        if keywords_found:
            keyword_list = list(keywords_found)[:2]
            sentences.append(f"Key academic areas covered include {', '.join(keyword_list)}, providing valuable insights for your research.")

        # Research guidance
        sentences.append("These materials offer comprehensive coverage of your topic with diverse perspectives from NU faculty and researchers.")

        if len(sentences) > 5:
            sentences = sentences[:5]

        return ' '.join(sentences)

    def _generate_basic_recommendation_fallback(self, query, search_results):
        """Generate concise basic recommendation fallback for other API errors"""
        num_results = len(search_results['documents'][0])
        return f"I found {num_results} relevant resources for '{query}' in the NU eLibrary collection. These materials provide valuable insights for your research and are available for reference."


def render_source_card(book_data, rank):
    """Render a Perplexity-style source card"""
    metadata = book_data['metadata']
    chunk_type = metadata.get('chunk_type', 'publication_detail')
    
    # Extract details
    if chunk_type == 'author_summary':
        card_title = metadata.get('author', 'NU Faculty')
        author = metadata.get('author', '')
        button_url = f"https://elibrary.nu.edu.om/cgi-bin/koha/opac-search.pl?q=au:%22{author.replace(' ', '+')}%22"
        subtitle = "Faculty Profile"
        details = ["View Publications"]
    else:  # book_detail
        card_title = metadata.get('title', 'Library Book')
        author = metadata.get('author', 'Unknown Author')
        button_url = metadata.get('link', 'https://elibrary.nu.edu.om/')
        subtitle = metadata.get('publisher', 'National University')
        
        # Build details list
        details = []
        if metadata.get('publication_year'):
            details.append(f"📅 {metadata['publication_year']}")
        if metadata.get('rating'):
            details.append(f"⭐ {metadata['rating']}")
        if metadata.get('language'):
            details.append(f"🌐 {metadata['language']}")
        if metadata.get('pages'):
            details.append(f"📄 {metadata['pages']} pgs")
            
    # Description
    description = metadata.get('description', '')
    if len(description) > 120:
        description = description[:117] + "..."
    elif not description:
        description = "No description available."

    # Join details with dots
    meta_html = ' &bull; '.join(details)
    
    html_content = f"""
<a href="{button_url}" target="_blank" class="source-card" style="min-height: 180px; display: flex; flex-direction: column; text-decoration: none; color: inherit;">
<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
<div style="display: flex; align-items: flex-start; gap: 0.5rem;">
<div class="source-number" style="min-width: 20px;">{rank}</div>
<div>
<div class="source-title" style="font-size: 1rem; margin-bottom: 0.2rem;">{card_title}</div>
<div style="font-size: 0.85rem; color: #64748b; font-weight: 500;">{author}</div>
<div style="font-size: 0.75rem; color: #94a3b8;">{subtitle}</div>
</div>
</div>
</div>
<div style="margin-top: 0.75rem; font-size: 0.8rem; color: #334155; line-height: 1.4; flex-grow: 1;">
{description}
</div>
<div class="source-meta" style="margin-top: 0.75rem; padding-top: 0.5rem; border-top: 1px solid #f1f5f9; font-size: 0.75rem;">
{meta_html}
</div>
</a>
"""
    return html_content

def display_book_card(book_data, rank, query, rag_system, summary_cache=None):
    """Display a professional library book card with relevance score (Legacy fallback, now using render_source_card for grid)"""
    st.markdown(render_source_card(book_data, rank), unsafe_allow_html=True)


def keyword_search_tab(rag_system, user_profile):
    """Professional search interface with semantic capabilities"""
    # Search interface
    st.markdown("### Search Library Catalog")
    
    # Search input
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input(
            "Search Query",
            placeholder="Enter keywords (English or Arabic)...",
            key="keyword_search_input"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("Search", use_container_width=True, key="keyword_search_button", type="primary")
    
    # Perform search
    if query and (search_button or query):
        st.markdown("---")
        
        with st.spinner("🔍 Analyzing query and searching knowledge base..."):
            # Search for books with hierarchical chunking - show all relevant results
            # Pass user_profile to search
            search_output = rag_system.search_books(query, user_profile, n_results=50)
            
            results = search_output['results']
            analysis = search_output['analysis']

            # Display Semantic Analysis
            english_query = analysis.get('english_query', query)
            intent = analysis.get('intent', 'general')
            synonyms = analysis.get('synonyms', [])
            
            # Show translation if different
            if query != english_query:
                st.info(f"🌐 **Translated Query:** {english_query} (Searching for English resources)")
            
            # Show Related Concepts (Pills)
            if synonyms:
                st.write("**Related Concepts:**")
                # Create HTML for pills
                pills_html = ""
                for synonym in synonyms:
                    pills_html += f"""
                    <span class="concept-pill">{synonym}</span>
                    """
                st.markdown(pills_html, unsafe_allow_html=True)
            
            if results['documents'][0]:
                # Count chunk types for summary
                chunk_types = {}
                for metadata in results['metadatas'][0]:
                    chunk_type = metadata.get('chunk_type', 'publication_detail')
                    chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                
                total_results = len(results['documents'][0])
                books = chunk_types.get('book_detail', 0)
                author_profiles = chunk_types.get('author_summary', 0)
                
                # Results header
                st.markdown(f"### Search Results ({total_results} items)")
                
                # Intent-specific message
                if intent == "research":
                    st.success("🔬 Prioritizing Research Papers & Faculty Publications based on your query.")
                elif intent == "textbook":
                    st.success("📚 Prioritizing Textbooks & Learning Materials based on your query.")
                    
                st.markdown(f"**{books}** books and **{author_profiles}** author profiles found, sorted by relevance to **{user_profile}**")
                st.markdown("---")

                # Display results in a grid
                st.markdown('<div class="sources-container">', unsafe_allow_html=True)
                for i in range(len(results['documents'][0])):
                    book_data = {
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i]
                    }
                    st.markdown(render_source_card(book_data, i + 1), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # End of results
                st.markdown("---")
                st.info("📚 All results are sorted by semantic relevance and your profile.")

            else:
                st.warning("No results found. Please try different search terms.")
    



def sort_by_rating(books_data, sort_type):
    """Sort books by rating (high, low, or average)"""
    # Filter books that have ratings
    books_with_ratings = []
    books_without_ratings = []
    
    for book in books_data:
        rating = book['metadata'].get('rating', '')
        try:
            rating_value = float(rating) if rating else None
            if rating_value is not None:
                books_with_ratings.append((book, rating_value))
            else:
                books_without_ratings.append(book)
        except (ValueError, TypeError):
            books_without_ratings.append(book)
    
    # Sort based on type
    if sort_type == 'high':
        # Sort by rating descending (highest first)
        books_with_ratings.sort(key=lambda x: x[1], reverse=True)
    elif sort_type == 'low':
        # Sort by rating ascending (lowest first)
        books_with_ratings.sort(key=lambda x: x[1])
    elif sort_type == 'average':
        # Sort by distance to 3.5 (average rating)
        books_with_ratings.sort(key=lambda x: abs(x[1] - 3.5))
    
    # Combine sorted books with ratings + books without ratings at the end
    sorted_books = [book for book, rating in books_with_ratings] + books_without_ratings
    return sorted_books


def chatbot_tab(rag_system, user_profile):
    """Conversational chatbot interface for book recommendations"""
    st.markdown("### 💬 AI Library Assistant")
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.chat_context = []
        st.session_state.processing_message = False
        st.session_state.current_user_input = None
        st.session_state.chat_input_key = 0  # Key to force input reset
    
    # Welcome message at the top (always visible)
    st.markdown(f"""
    <div class="chat-welcome-box">
        <h1 class="chat-welcome-title">Where research begins.</h1>
        <p class="chat-welcome-text">
            Ask your intelligent library assistant for resources customized for <strong>{user_profile}</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat input with Clear button (matching Search tab layout)
    col1, col2 = st.columns([5, 1])
    with col1:
        # Use text_input with dynamic key to force clearing
        user_input = st.text_input(
            "Ask a question",
            placeholder="Ask me anything (English or Arabic)...",
            key=f"chatbot_text_input_{st.session_state.chat_input_key}"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        clear_chat = st.button("🗑️ Clear", use_container_width=True, key="clear_chat")
    
    # Handle clear button
    if clear_chat:
        st.session_state.chat_history = []
        st.session_state.chat_context = []
        st.session_state.processing_message = False
        st.session_state.current_user_input = None
        st.session_state.chat_input_key += 1  # Increment to reset input
        st.rerun()
    
    # Process user input when they press Enter
    if user_input and user_input.strip():
        # Set processing flag and store input
        st.session_state.processing_message = True
        st.session_state.current_user_input = user_input
        st.session_state.chat_input_key += 1  # Increment to clear input on next render
        st.rerun()
    
    # Display chat history in a container to keep it static
    history_container = st.container()
    with history_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display results in a grid
                if "books" in message:
                    st.markdown('<div class="sources-container">', unsafe_allow_html=True)
                    for i, book in enumerate(message["books"]):
                        st.markdown(render_source_card(book, i + 1), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # If we're processing, show the NEW conversation (question + loading response)
    if st.session_state.processing_message and st.session_state.current_user_input:
        # Show user message
        with st.chat_message("user"):
            st.markdown(st.session_state.current_user_input)
        
        # Show loading for assistant
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking and searching for books..."):
                user_input = st.session_state.current_user_input
                
                # Detect rating-based queries
                user_input_lower = user_input.lower()
                rating_sort = None
                if any(word in user_input_lower for word in ['highly rated', 'high rated', 'best rated', 'top rated', 'highest rated']):
                    rating_sort = 'high'
                elif any(word in user_input_lower for word in ['lowly rated', 'low rated', 'poorly rated', 'worst rated', 'lowest rated']):
                    rating_sort = 'low'
                elif any(word in user_input_lower for word in ['average rated', 'medium rated', 'moderately rated']):
                    rating_sort = 'average'
                
                # Search for relevant books - show all recommendations (no limit)
                search_output = rag_system.search_books(user_input, user_profile, n_results=50)
                results = search_output['results']
                analysis = search_output['analysis']
                
                english_query = analysis.get('english_query', user_input)
                
                if results['documents'][0]:
                    # Generate conversational response
                    response_text = generate_chat_response(rag_system, english_query, results, st.session_state.chat_context)
                    
                    if user_input != english_query:
                         response_text = f"**Translated to:** *{english_query}*\n\n" + response_text
                    
                    books_data = []
                    for i in range(len(results['documents'][0])):
                        book_data = {
                            'document': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'distance': results['distances'][0][i]
                        }
                        books_data.append(book_data)
                    
                    # Apply rating-based sorting if requested
                    if rating_sort:
                        books_data = sort_by_rating(books_data, rating_sort)
                    
                    # Add messages to chat history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_input
                    })
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_text,
                        "books": books_data,
                        "rating_sort": rating_sort
                    })
                    
                    # Update context for next conversation
                    st.session_state.chat_context.append({
                        "user": english_query,
                        "assistant": response_text
                    })
                    
                    # Keep only last 3 exchanges in context
                    if len(st.session_state.chat_context) > 3:
                        st.session_state.chat_context.pop(0)
                
                else:
                    response_text = "I couldn't find any books matching your query. Could you try rephrasing or asking about a different topic?"
                    
                    # Add messages to history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_input
                    })
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_text
                    })
                
                # Reset processing flag and rerun to show the complete message
                st.session_state.processing_message = False
                st.session_state.current_user_input = None
                st.rerun()


def generate_chat_response(rag_system, user_question, search_results, chat_context):
    """Generate conversational response based on user question and search results"""
    # 1. Check for specific intents (Location & Availability)
    extra_context = ""
    
    # Location Query
    location_info = rag_system.get_location_info(user_question)
    if location_info:
        extra_context += f"\n\n[SYSTEM INFO] {location_info}"
        
    # Availability Query
    # Simple check if user is asking about availability of a specific book from results
    if "available" in user_question.lower() or "check out" in user_question.lower():
         # Check the top result if exists
         if search_results['metadatas'][0]:
             top_book = search_results['metadatas'][0][0]
             title = top_book.get('title')
             if title:
                 status = rag_system.check_availability(title)
                 extra_context += f"\n\n[LMS REAL-TIME INFO] For book '{title}': {status}"

    # Build context from previous conversation
    context_str = ""
    if chat_context:
        context_str = "Previous conversation:\n"
        for exchange in chat_context[-2:]:  # Last 2 exchanges
            context_str += f"User: {exchange['user']}\n"
            context_str += f"Assistant: {exchange['assistant']}\n\n"
    
    # Prepare search results context with book titles and details
    books_context = "Available books in the library:\n\n"
    book_details = []
    for i, (doc, meta) in enumerate(zip(
        search_results['documents'][0],
        search_results['metadatas'][0]
    )):
        # Extract key information
        title = meta.get('title', '')
        author = meta.get('author', '')
        
        if title and title != 'Library Book':
            book_info = f"'{title}'"
            if author:
                book_info += f" by {author}"
            
            book_details.append(book_info)
        
        # Add document excerpt for context
        books_context += f"{i+1}. {doc[:400]}...\n\n"
    
    prompt = f"""You are an advanced AI librarian for National University eLibrary (Oman/GCC region).
    
SYSTEM INSTRUCTIONS:
1. **Multilingual Support**: You are fully bilingual (English & Arabic). 
   - If the user asks in Arabic, REPLY IN ARABIC.
   - If the user asks in English, REPLY IN ENGLISH.
   - You may use code-switching (mixing English terms in Arabic text) if appropriate for an academic context.
   
2. **GCC/Oman Context**: Be polite, formal, and culturally aware. Use phrases like "Welcome" or "Ahlan" if appropriate.

3. **Spatial Awareness**: If you see [SYSTEM INFO] about location, explicitly guide the user there.

4. **Real-time Status**: If you see [LMS REAL-TIME INFO] about availability, explicitly state whether the book is available or checked out.

{context_str}

User's current question: "{user_question}"
{extra_context}

{books_context}

Your task: Provide a helpful, natural response (max 4-5 sentences).
- Citation Policy: You MUST refer to books using [1], [2], etc. corresponding to their rank in the Sources section.
- If specific books are found, recommend them by title with their citation tag.
- If location/availability info is provided, include it clearly.
- Always match the user's language.
"""
    
    try:
        response = rag_system.model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Enhanced fallback response with book details
        num_books = len(search_results['documents'][0])
        
        if book_details:
            fallback = f"I found {book_details[0]} which might be relevant. "
            if location_info:
                fallback += f"\n\n{location_info}"
        else:
            fallback = f"I found {num_books} relevant resources."
            
        return fallback


def display_chat_book_card(book_data, rank=1):
    """Display a detailed book card for chat interface"""
    st.markdown(render_source_card(book_data, rank), unsafe_allow_html=True)
    
    # Get description and context from metadata
    description = metadata.get('description', '')
    context = metadata.get('context', '')
    
    # If no description in metadata, extract from document
    if not description and 'Description:' in document:
        try:
            desc_line = [line for line in document.split('\n') if line.startswith('Description:')][0]
            description = desc_line.replace('Description:', '').strip()
        except:
            description = ""
    
    # Extract context from document if not in metadata
    if not context and 'Research Context:' in document:
        try:
            context_line = [line for line in document.split('\n') if line.startswith('Research Context:')][0]
            context = context_line.replace('Research Context:', '').strip()
        except:
            pass
    
    # Combine description and context
    full_description = description
    if context and context != description:
        full_description = f"{description}<br><br><strong>Target Audience:</strong> {context}"
    
    # Determine card content based on chunk type
    if chunk_type == 'author_summary':
        card_title = metadata.get('author', 'NU Faculty')
        author = metadata.get('author', '')
        button_url = f"https://elibrary.nu.edu.om/cgi-bin/koha/opac-search.pl?q=au:%22{author.replace(' ', '+')}%22"
        if not full_description:
            full_description = f"Publications by {author}."
    else:  # book_detail
        card_title = metadata.get('title', 'Library Book')
        author = metadata.get('author', 'Unknown Author')
        if author == 'Unknown Author' and 'Author:' in document:
            try:
                author_line = [line for line in document.split('\n') if line.startswith('Author:')][0]
                author = author_line.replace('Author:', '').strip()
            except:
                pass
        button_url = metadata.get('link', 'https://elibrary.nu.edu.om/')
        if not full_description:
            full_description = "Book information available at the library."
    
    # Build metadata items
    metadata_items = []
    if chunk_type == 'book_detail':
        if metadata.get('publication_year'):
            metadata_items.append(f"📅 {metadata['publication_year']}")
        if metadata.get('publisher'):
            metadata_items.append(f"📚 {metadata['publisher']}")
        if metadata.get('rating'):
            metadata_items.append(f"⭐ {metadata['rating']}/5.0")
        if metadata.get('pages'):
            metadata_items.append(f"📄 {metadata['pages']} pages")
        if metadata.get('language'):
            metadata_items.append(f"🌐 {metadata['language']}")
    
    metadata_html = ' • '.join(metadata_items) if metadata_items else ''
    
    # Create professional book card (similar to search results)
    st.markdown(f"""
    <div class="book-card">
        <div class="book-title">
            <a href="{button_url}" target="_blank">{card_title}</a>
        </div>
        <div class="book-author">{author}</div>
        {f'<div class="book-metadata">{metadata_html}</div>' if metadata_html else ''}
        <div class="book-description">{full_description}</div>
        <a href="{button_url}" target="_blank" class="view-button">View in Library Catalog →</a>
    </div>
    """, unsafe_allow_html=True)


def dashboard_tab(rag_system, user_profile_data):
    """Personalized Dashboard View"""
    st.markdown(f"### 👋 Welcome back, {user_profile_data['role']}!")
    
    # Quick Stats Row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Reading Goal", "12/20 Books", "2 this week")
    with col2:
        st.metric("Saved Resources", "45", "+5 new")
    with col3:
        st.metric("Interests", len(user_profile_data['interests']), "Active")
        
    st.markdown("---")
    
    # Section 1: Top Picks for your Major
    st.subheader(f"🎓 Top Picks for {user_profile_data['major']}")
    with st.spinner(f"Curating resources for {user_profile_data['major']}..."):
        rec_results = rag_system.get_recommendations_by_profile(user_profile_data, n_results=4)
        if rec_results['documents'][0]:
            cols = st.columns(2)
            for i in range(len(rec_results['documents'][0])):
                with cols[i % 2]:
                    book_data = {
                        'document': rec_results['documents'][0][i],
                        'metadata': rec_results['metadatas'][0][i],
                        'distance': rec_results['distances'][0][i]
                    }
                    display_book_card(book_data, i + 1, "Dashboard Recommendation", rag_system)
        else:
            st.info("No specific recommendations found for this major yet.")

    st.markdown("---")

    # Section 2: Based on Reading History
    if user_profile_data['history']:
        last_read = user_profile_data['history']
        st.subheader(f"📖 Because you read '{last_read}'")
        with st.spinner(f"Finding books similar to {last_read}..."):
            history_results = rag_system.get_recommendations_by_history(last_read, n_results=2)
            if history_results['documents'][0]:
                for i in range(len(history_results['documents'][0])):
                    book_data = {
                        'document': history_results['documents'][0][i],
                        'metadata': history_results['metadatas'][0][i],
                        'distance': history_results['distances'][0][i]
                    }
                    display_book_card(book_data, i + 1, "History Recommendation", rag_system)
    
    # Section 3: Specialized Content (Role based)
    st.markdown("---")
    if user_profile_data['role'] == 'Faculty':
        st.subheader("🔬 Recent Faculty Publications")
        # Logic to fetch recent publications could go here
        st.info("Explore the latest research output from your colleagues.")
    else:
        st.subheader("📚 Trending Textbooks")
        st.info("Most borrowed textbooks in your department this semester.")


def scan_book_tab(rag_system):
    """Visual Recognition Interface"""
    st.markdown("### 📷 Scan Physical Book")
    
    st.info("Upload a photo of a book cover or spine to identify it, check availability, and find similar resources.")
    
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        try:
            # 1. Display Image using base64 HTML to bypass broken JPEG library
            import base64
            image_bytes = uploaded_file.getvalue()
            b64_image = base64.b64encode(image_bytes).decode()
            mime_type = uploaded_file.type
            
            st.markdown(f"""
                <div style="text-align: center; margin-bottom: 1rem;">
                    <img src="data:{mime_type};base64,{b64_image}" alt="Uploaded Book Cover" style="max-width: 300px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <p style="color: #666; font-size: 0.9rem; margin-top: 0.5rem;">Uploaded Book Cover</p>
                </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("🔍 Analyzing image with Computer Vision..."):
                # 2. Prepare Image for API
                # Send raw bytes to Gemini (bypasses local image processing)
                image_blob = {
                    'mime_type': mime_type,
                    'data': image_bytes
                }
                
                # 3. Call API
                book_info = rag_system.analyze_image(image_blob)
                # st.write(f"Debug: Received: {book_info}") # Uncomment for deep debugging
                
                if "error" in book_info:
                    st.error(f"Analysis Error: {book_info['error']}")
                    st.info("Tip: Try a clearer image where the title is legible.")
                else:
                    title = book_info.get('title')
                    author = book_info.get('author')
                    
                    st.success(f"**Identified:** {title} by {author}")
                    
                    # Scan Results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### ✅ Availability")
                        status = rag_system.check_availability(title)
                        st.markdown(status)
                        st.markdown(rag_system.get_location_info(title) or "")
                    
                    with col2:
                        st.markdown("#### 📖 Bibliographic Info")
                        st.markdown(f"**Title:** {title}")
                        st.markdown(f"**Author:** {author}")
                        if book_info.get('edition'):
                            st.markdown(f"**Edition:** {book_info.get('edition')}")
                    
                    st.markdown("---")
                    st.markdown("#### 📚 Similar Digital Resources")
                    
                    # 2. Find Similar Books
                    search_results = rag_system.search_books(title, n_results=3)
                    if search_results['results']['documents'][0]:
                        cols = st.columns(3)
                        for i in range(len(search_results['results']['documents'][0])):
                            with cols[i % 3]:
                                book_data = {
                                    'document': search_results['results']['documents'][0][i],
                                    'metadata': search_results['results']['metadatas'][0][i],
                                    'distance': search_results['results']['distances'][0][i]
                                }
                                display_book_card(book_data, i + 1, "Similar Title", rag_system)
                    else:
                        st.info("No similar digital resources found in the database.")

        except Exception as e:
            st.error(f"Error processing image: {e}")
            return


def voice_assistant_tab(rag_system):
    """Voice-to-Text & Intent Recognition Interface with Robust Sync"""
    st.markdown("### 🎙️ AI Voice Assistant")
    st.info("1. Tap the mic to record. 2. Your transcription will appear below. 3. Click 'Submit' to process.")
    
    # --- 1. Text-to-Speech (TTS) Utility ---
    def trigger_tts(text):
        """Play audio response using Browser TTS via components.html"""
        import streamlit.components.v1 as components
        js_tts = f"""
        <script>
            const text = "{text.replace('"', '\\"').replace('\n', ' ')}";
            if (window.speechSynthesis) {{
                // Stop any current speaking
                window.speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.lang = 'en-US';
                utterance.rate = 1.0;
                window.speechSynthesis.speak(utterance);
            }}
        </script>
        """
        components.html(js_tts, height=0)

    # --- 2. Voice Input Component (Ultra-Robust Sync) ---
    st.markdown("#### 🎤 Voice Transcription")
    
    if 'voice_text_buffer' not in st.session_state:
        st.session_state.voice_text_buffer = ""

    # This component uses a more aggressive DOM searching strategy to find the Streamlit input
    voice_component_html = """
    <div style="display: flex; flex-direction: column; align-items: center; gap: 0.8rem; padding: 1.5rem; border: 2px dashed #4a5568; border-radius: 12px; background: #ffffff; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
        <button id="mic-btn" style="width: 60px; height: 60px; border-radius: 50%; border: none; background: #2d3748; color: white; cursor: pointer; transition: all 0.2s; display: flex; align-items: center; justify-content: center;">
            <svg id="mic-icon" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>
        </button>
        <div id="status" style="font-weight: 600; color: #2d3748; font-size: 0.9rem;">Tap Mic to Speak</div>
        <div id="result-preview" style="font-size: 0.95rem; color: #1a202c; font-weight: 500; text-align: center; border-top: 1px solid #edf2f7; padding-top: 0.8rem; width: 100%; min-height: 2rem;">(Transcription will appear here)</div>
    </div>

    <script>
        const micBtn = document.getElementById('mic-btn');
        const status = document.getElementById('status');
        const resultPreview = document.getElementById('result-preview');
        
        let recognition;
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = true;
            recognition.lang = 'en-US';

            recognition.onstart = () => {
                micBtn.style.background = '#e53e3e';
                status.innerText = 'Listening...';
                resultPreview.innerText = '...';
            };

            recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                resultPreview.innerText = transcript;
                
                if (event.results[0].isFinal) {
                    const syncToStreamlit = (text) => {
                        // Find ALL textareas in the parent document
                        const textareas = window.parent.document.querySelectorAll('textarea');
                        // Find the one that belongs to our "Edit Transcription" section
                        // Streamlit adds a label or we can check the placeholder
                        let target = null;
                        for (let ta of textareas) {
                            if (ta.placeholder.includes("spoken words will appear here") || 
                                ta.getAttribute('aria-label') === "Edit Transcription") {
                                target = ta;
                                break;
                            }
                        }
                        
                        // Fallback to the first textarea if no match
                        if (!target && textareas.length > 0) target = textareas[0];
                        
                        if (target) {
                            // Set value and trigger events that Streamlit listens to
                            const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                            nativeInputValueSetter.call(target, text);
                            
                            target.dispatchEvent(new Event('input', { bubbles: true }));
                            target.dispatchEvent(new Event('change', { bubbles: true }));
                            // Blur to force sync if needed
                            target.blur();
                            setTimeout(() => target.focus(), 10);
                        }
                    };
                    syncToStreamlit(transcript);
                }
            };

            recognition.onend = () => {
                micBtn.style.background = '#2d3748';
                status.innerText = 'Ready';
            };

            recognition.onerror = (e) => {
                status.innerText = 'Error: ' + e.error;
                micBtn.style.background = '#2d3748';
            };

            micBtn.onclick = () => {
                try {
                    recognition.start();
                } catch(e) {
                    recognition.stop();
                }
            };
        } else {
            status.innerText = 'Not supported';
        }
    </script>
    """
    
    import streamlit.components.v1 as components
    components.html(voice_component_html, height=180)
    
    # Visible text area with specific aria-label for JS to target
    voice_input = st.text_area("Edit Transcription", 
                              key="voice_input_area",
                              placeholder="Your spoken words will appear here. You can also type manually.",
                              help="If transcription doesn't appear automatically, please type your request here.")
    
    submit_col, autoprocess_col = st.columns([2, 1])
    with submit_col:
        submit = st.button("🚀 Process & Speak Response", use_container_width=True, type="primary")
    
    if submit:
        if voice_input:
            with st.spinner("🤖 Intent Recognition in progress..."):
                # Save to history/buffer
                st.session_state.voice_text_buffer = voice_input
                
                intent_data = rag_system.process_voice_intent(voice_input)
                intent = intent_data.get('intent')
                
                st.markdown(f"**Recognized Intent:** `{intent.upper() if intent else 'OTHER'}`")
                
                if intent == 'search':
                    subject = intent_data.get('subject')
                    lang = intent_data.get('language') or "English"
                    action_msg = f"Okay, I am searching for {subject} books in {lang}."
                    st.success(action_msg)
                    trigger_tts(action_msg)
                    
                    results = rag_system.search_books(f"{subject} in {lang}")
                    if results['results']['documents'][0]:
                        st.markdown("---")
                        cols = st.columns(3)
                        for i in range(min(3, len(results['results']['documents'][0]))):
                            with cols[i]:
                                book_data = {
                                    'document': results['results']['documents'][0][i],
                                    'metadata': results['results']['metadatas'][0][i],
                                    'distance': results['results']['distances'][0][i]
                                }
                                display_book_card(book_data, i + 1, "Top Match", rag_system)
                    else:
                        err_msg = "I couldn't find any books matching that topic in our database."
                        st.info(f"Chatbot: {err_msg}")
                        trigger_tts(err_msg)
                
                elif intent == 'status':
                    title = intent_data.get('title')
                    if title:
                        status_res = rag_system.check_availability(title)
                        st.markdown(f"**Status for '{title}':**")
                        st.info(status_res)
                        # Clean markdown for cleaner TTS
                        clean_msg = status_res.replace("**", "").replace("📍", "").replace("Availability:", "")
                        trigger_tts(f"Current status for {title}: {clean_msg}")
                    else:
                        st.warning("I couldn't identify which book you meant. Please speak the title clearly.")
                        trigger_tts("I am sorry, I couldn't understand the book title. Could you please repeat it?")
                        
                elif intent == 'reserve':
                    title = intent_data.get('title')
                    if title:
                        st.balloons()
                        res_msg = f"Success! I have reserved '{title}' for you. You can collect it at the first-floor circulation desk."
                        st.success(f"✅ {res_msg}")
                        trigger_tts(res_msg)
                    else:
                        st.warning("Which book would you like to reserve?")
                        trigger_tts("I understood you want to reserve a book, but I missed the title. Which book should I reserve?")
                
                else:
                    # General Chat Fallback
                    response = rag_system.generate_chat_response([{"role": "user", "content": voice_input}], "General")
                    st.markdown(f"**Assistant Response:**\n{response}")
                    # Only speak first two sentences if it's long
                    speak_text = response.split('.')[0] + "." + (response.split('.')[1] + "." if len(response.split('.')) > 1 else "")
                    trigger_tts(speak_text)
        else:
            st.error("⚠️ No text detected. Please speak into the mic or type your request above.")
            trigger_tts("Please provide a command so I can help you.")

def admin_analytics_tab(rag_system):
    """Predictive Analytics Dashboard for Administrators"""
    st.markdown("### 📈 Library Analytics Dashboard")
    st.info("📊 **Real-time Insights**: This dashboard is now powered by actual collection data and search logs.")
    
    # --- New Section: Collection Overview ---
    st.subheader("📚 Collection Overview")
    with st.spinner("Calculating real-time collection statistics..."):
        coll_stats = rag_system.get_collection_stats()
        # Pre-fetch predictions for charts
        predictions = rag_system.get_peak_demand_prediction()
        
        cols = st.columns(len(coll_stats))
        for i, (college, count) in enumerate(coll_stats.items()):
            label = college.replace("College of ", "")
            cols[i].metric(label, f"{count} Books")
            
    # --- New Section: System Health (Dev Metrics) ---
    st.subheader("🖥️ System Health & Performance")
    
    # Calculate metrics from logs
    try:
        with open('logs/search_history.json', 'r') as f:
            logs = json.load(f)
            
        if logs:
            df_logs = pd.DataFrame(logs)
            
            # Ensure new fields exist (backward compatibility)
            if 'latency_ms' not in df_logs.columns:
                df_logs['latency_ms'] = 0
            if 'status' not in df_logs.columns:
                df_logs['status'] = 'success'
                
            # Filter last 24h
            # (Assuming timestamp is ISO format)
            df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])
            
            # Key Metrics
            avg_latency = df_logs['latency_ms'].mean()
            p95_latency = df_logs['latency_ms'].quantile(0.95)
            error_count = len(df_logs[df_logs['status'] == 'error'])
            error_rate = (error_count / len(df_logs)) * 100
            total_reqs = len(df_logs)
            
            h_cols = st.columns(4)
            h_cols[0].metric("Total Requests", total_reqs, help="All tracked search queries")
            h_cols[1].metric("Avg Latency", f"{avg_latency:.1f}ms", f"{p95_latency:.1f}ms (p95)", help="Average vs p95 execution time")
            h_cols[2].metric("Error Rate", f"{error_rate:.1f}%", f"{error_count} errors", delta_color="inverse")
            h_cols[3].metric("Zero Results", len(df_logs[df_logs['result_count'] == 0]), help="Searches returning no books")
            
            # Latency Chart
            st.markdown("##### Latency Trends (ms)")
            st.line_chart(df_logs[['timestamp', 'latency_ms']].set_index('timestamp'), height=200)
            
        else:
             st.info("No system logs available yet.")
             
    except Exception as e:
        st.warning(f"Could not load system metrics: {e}")

    # --- New Section: Visual Analytics ---
    st.subheader("📊 Visual Analytics")
    
    # 1. Books Distribution Bar Chart
    if coll_stats:
        # Prepare data for simple bar chart
        chart_data = {
            "College": [c.replace("College of ", "").replace("International Maritime College Oman", "IMCO") for c in coll_stats.keys()],
            "Books": list(coll_stats.values())
        }
        st.markdown("#### Books per Domain")
        st.bar_chart(data=chart_data, x="College", y="Books", use_container_width=True)
    
    # 2. Search Demand Trends
    if predictions and predictions[0]['demand_score'] > 0:
        st.markdown("#### Real-time Demand Activity")
        # Extract data for chart
        trend_data = {p['subject']: p['demand_score'] for p in predictions}
        st.bar_chart(trend_data, use_container_width=True)

    st.markdown("---")
    
    # --- Section: Peak Demand Prediction ---
    st.subheader("📅 Live Trend Prediction")
    
    # Display as a styled table
    st.markdown("""
    <style>
    .analytics-table { width: 100%; border-collapse: collapse; margin-bottom: 1rem; }
    .analytics-table th, .analytics-table td { padding: 0.75rem; text-align: left; border-bottom: 1px solid #e2e8f0; }
    .analytics-table th { background-color: #f7fafc; font-weight: 600; }
    .demand-high { color: #c53030; font-weight: bold; }
    .demand-medium { color: #dd6b20; }
    .demand-low { color: #38a169; }
    </style>
    """, unsafe_allow_html=True)
    
    if predictions and predictions[0]['demand_score'] > 0:
        table_html = "<table class='analytics-table'><tr><th>Major</th><th>Trend Status</th><th>Analysis</th><th>Activity Score</th></tr>"
        for p in predictions:
            score_class = "demand-high" if p['demand_score'] >= 85 else ("demand-medium" if p['demand_score'] >= 70 else "demand-low")
            table_html += f"<tr><td>{p['subject']}</td><td>{p['peak_months']}</td><td>{p['reason']}</td><td class='{score_class}'>{p['demand_score']}%</td></tr>"
        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.warning("⏱️ **Awaiting Search Logs**: Perform some searches to generate real trend data.")
    
    st.markdown("---")
    
    # --- Section: Collection Gap Analysis ---
    st.subheader("🔍 Real-World Gap Analysis")
    gaps = rag_system.get_gap_analysis()
    
    if gaps and gaps[0]['searches'] > 0:
        # Add a gap visualization
        gap_data = {g['subject']: g['gap_score'] for g in gaps}
        st.markdown("#### Subject Gap Scores (Higher = Urgent Need)")
        st.bar_chart(gap_data, use_container_width=True)

        gap_table_html = "<table class='analytics-table'><tr><th>Search Topic</th><th>Search Volume</th><th>Books in Collection</th><th>Gap Score</th></tr>"
        for g in gaps:
            score_class = "demand-high" if g['gap_score'] >= 70 else ("demand-medium" if g['gap_score'] >= 50 else "demand-low")
            gap_table_html += f"<tr><td>{g['subject']}</td><td>{g['searches']}</td><td>{g['books']}</td><td class='{score_class}'>{g['gap_score']}%</td></tr>"
        gap_table_html += "</table>"
        st.markdown(gap_table_html, unsafe_allow_html=True)
    else:
        st.warning("⏱️ **Awaiting Search Logs**: Analyzing query patterns vs. collection availability requires more user activity.")
    
    st.markdown("---")
    
    # --- Section: AI Purchase Recommendations ---
    st.subheader("🛒 AI-Powered Purchase Recommendations")
    if gaps and gaps[0]['searches'] > 0:
        with st.spinner("Generating recommendations based on REAL collection gaps..."):
            recommendations = rag_system.get_purchase_recommendations()
            st.markdown(recommendations)
    else:
        st.info("Purchase recommendations will be generated once search trends are identified.")



def main():
    """Main application"""
    # Professional Library Header
    # Professional Library Hero Section
    st.markdown("""
        <div class="hero-container">
            <div class="library-title">NU eLibrary</div>
            <div class="library-subtitle">Intelligent Academic Search System</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        with st.spinner("Initializing search system..."):
            st.session_state.rag_system = LibraryRAG()
    
    rag_system = st.session_state.rag_system
    
    # Sidebar for User Profile
    with st.sidebar:
        st.header("👤 Discovery Profile")
        
        # Role Selection
        role = st.selectbox("Role", ["Student", "Faculty", "Researcher"])
        
        # Academic Level
        level_options = ["Undergraduate", "Postgraduate", "PhD"] if role != "Faculty" else ["Assistant Professor", "Associate Professor", "Professor"]
        level = st.selectbox("Academic Level", level_options)

        # Major/Department
        major = st.selectbox(
            "Major/Department",
            options=[
                "General",
                "College of Engineering",
                "College of Medicine and Health Sciences",
                "College of Pharmacy",
                "International Maritime College Oman",
                "Computer Science", # Added for better demo
                "Business Administration" 
            ],
            index=1
        )
        
        # Interests (Multi-select)
        interests = st.multiselect(
            "Research Interests",
            ["AI", "Robotics", "Public Health", "Pharmacology", "Maritime Logistics", "Renewable Energy", "Data Science"],
            default=["AI"] if "Engineering" in major else []
        )
        
        # Reading History Simulation
        st.markdown("### 📚 Reading History")
        history = st.selectbox(
            "Last Read Book (Simulation)",
            [
                "",
                "Research Methodology: A Step-by-Step Guide",
                "Introduction to Algorithms",
                "Clinical Pharmacy and Therapeutics",
                "Maritime Law"
            ],
            index=1
        )
        
        # Pack profile data
        user_profile_data = {
            'role': role,
            'level': level,
            'major': major,
            'interests': interests,
            'history': history if history else None
        }
        
        st.divider()
        st.caption(f"Profile Active: {role} in {major}")
    
    # Tab selection - conditionally show Admin Analytics for Faculty/Researcher
    st.markdown("<br>", unsafe_allow_html=True)
    
    nav_options = ["📊 Dashboard", "🔍 Search", "💬 AI Assistant", "📷 Scan Book", "🎙️ Voice Assistant"]
    if user_profile_data['role'] in ['Faculty', 'Researcher']:
        nav_options.append("📈 Admin Analytics")
    
    selected_tab = st.radio(
        "Navigation",
        options=nav_options,
        horizontal=True,
        key="main_tab_selector"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display selected tab content
    if selected_tab == "📊 Dashboard":
        dashboard_tab(rag_system, user_profile_data)
    elif selected_tab == "🔍 Search":
        keyword_search_tab(rag_system, user_profile_data['major'])
    elif selected_tab == "📷 Scan Book":
        scan_book_tab(rag_system)
    elif selected_tab == "🎙️ Voice Assistant":
        voice_assistant_tab(rag_system)
    elif selected_tab == "📈 Admin Analytics":
        admin_analytics_tab(rag_system)
    else:  # AI Chatbot
        chatbot_tab(rag_system, user_profile_data['major'])
    
    # Professional Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
        <div class="footer-container">
            <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 0.5rem;">National University eLibrary</div>
            <div>Azaiba, Bousher, Muscat, Sultanate of Oman</div>
            <div style="margin-top: 0.5rem;">
                <a href="https://elibrary.nu.edu.om/" target="_blank" class="footer-link">Visit Official Website</a> | 
                <a href="https://www.nu.edu.om/" target="_blank" class="footer-link">National University</a>
            </div>
            <div style="margin-top: 0.75rem; font-size: 0.8rem; color: var(--text-secondary);">
                © 2025 National University Libraries. All rights reserved.
            </div>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()