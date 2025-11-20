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
import hashlib
import threading
import concurrent.futures
from functools import lru_cache

# Configure page
st.set_page_config(
    page_title="NU eLibrary Intelligent Search",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Library UI CSS - Light Mode Only
st.markdown("""
    <style>
    /* ========================================
       LIGHT MODE STYLING
       ======================================== */
    
    /* Main Layout */
    .main {
        background-color: #ffffff !important;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Header */
    .library-header {
        background: linear-gradient(135deg, #c99a2c 0%, #b8892a 100%);
        color: white;
        padding: 1.5rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
        border-bottom: 3px solid #2c5282;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .library-title {
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0;
        color: white;
    }
    
    .library-subtitle {
        font-size: 0.95rem;
        color: rgba(255, 255, 255, 0.95);
        margin-top: 0.25rem;
    }
    
    /* Search Box */
    .stTextInput > div > div > input {
        border: 2px solid #e2e8f0 !important;
        border-radius: 4px;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        background-color: #ffffff !important;
        color: #2d3748 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #2c5282 !important;
        box-shadow: 0 0 0 1px #2c5282 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #2c5282 !important;
        color: white !important;
        font-weight: 500;
        border-radius: 4px;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #1a365d !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Book Cards */
    .book-card {
        background-color: #ffffff !important;
        padding: 1.5rem;
        border-radius: 4px;
        border: 1px solid #e2e8f0 !important;
        margin-bottom: 1rem;
        transition: all 0.2s;
    }
    
    .book-card:hover {
        border-color: #cbd5e0 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .book-title {
        color: #1a365d !important;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        line-height: 1.4;
    }
    
    .book-title a {
        color: #1a365d !important;
        text-decoration: none;
    }
    
    .book-title a:hover {
        color: #2c5282 !important;
        text-decoration: underline;
    }
    
    .book-author {
        color: #4a5568 !important;
        font-size: 0.95rem;
        margin-bottom: 0.75rem;
    }
    
    .book-metadata {
        color: #718096 !important;
        font-size: 0.875rem;
        margin-bottom: 0.75rem;
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
    }
    
    .book-metadata-item {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
    }
    
    .book-description {
        color: #2d3748 !important;
        line-height: 1.6;
        margin-top: 0.75rem;
        font-size: 0.95rem;
    }
    
    .relevance-badge {
        display: inline-block;
        background-color: #edf2f7 !important;
        color: #2d3748 !important;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .view-button {
        display: inline-block;
        background-color: #2c5282 !important;
        color: white !important;
        padding: 0.5rem 1.25rem;
        border-radius: 4px;
        text-decoration: none;
        font-size: 0.9rem;
        font-weight: 500;
        margin-top: 0.75rem;
        transition: all 0.2s;
    }
    
    .view-button:hover {
        background-color: #1a365d !important;
        text-decoration: none;
        color: white !important;
    }
    
    /* Info Boxes */
    .stAlert {
        border-radius: 4px;
        border-left: 4px solid #2c5282 !important;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #1a365d !important;
    }
    
    /* Tabs/Radio */
    div[data-testid="stRadio"] > div {
        background-color: #f7fafc !important;
        padding: 0.5rem;
        border-radius: 4px;
        border: 1px solid #e2e8f0 !important;
    }
    
    div[data-testid="stRadio"] > div > label {
        font-weight: 500;
        color: #2d3748 !important;
    }
    
    /* Chat Messages */
    .stChatMessage {
        background-color: #f7fafc !important;
        border-radius: 4px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Dividers */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #e2e8f0 !important;
    }
    
    /* Welcome Section */
    .welcome-section {
        background-color: #f7fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 4px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    
    .welcome-title {
        color: #1a365d !important;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .welcome-text {
        color: #4a5568 !important;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Chat Welcome Box */
    .chat-welcome-box {
        background-color: #e8f4f8 !important;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .chat-welcome-title {
        color: #192f59 !important;
        margin-top: 0;
    }
    
    .chat-welcome-text {
        color: #666 !important;
    }
    
    /* Footer */
    .footer-container {
        text-align: center;
        padding: 1.5rem;
        color: #718096 !important;
        font-size: 0.875rem;
        background-color: #f7fafc !important;
        border-radius: 4px;
        margin-top: 2rem;
    }
    
    .footer-title {
        font-weight: 600;
        color: #2d3748 !important;
        margin-bottom: 0.5rem;
    }
    
    .footer-link {
        color: #2c5282 !important;
        text-decoration: none;
    }
    
    .footer-copyright {
        margin-top: 0.75rem;
        font-size: 0.8rem;
        color: #a0aec0 !important;
    }
    
    /* Force override any dark mode attempts */
    body, .stApp, [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
        color: #2d3748 !important;
    }
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
            'gemini-2.5-flash',
            generation_config=self.generation_config
        )

        # Summary cache for faster responses
        self.summary_cache = {}
        self.cache_lock = threading.Lock()

        # Thread pool for parallel processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
    
    def search_books(self, query, n_results=10, prioritize_publications=True):
        """Search for books using semantic search with title matching boost"""
        # Get more results initially for better re-ranking
        initial_results = min(n_results * 4, 50)

        results = self.collection.query(
            query_texts=[query],
            n_results=initial_results,
            include=['documents', 'metadatas', 'distances']
        )

        if not results['documents'][0]:
            return results

        # Process results based on hierarchical structure
        processed_results = self._process_hierarchical_results(
            results, query, n_results, prioritize_publications
        )
        
        # Apply title matching boost for better exact matches
        boosted_results = self._boost_title_matches(processed_results, query)

        return boosted_results
    
    def _boost_title_matches(self, results, query):
        """Boost results where title closely matches the query"""
        if not results['documents'][0]:
            return results
        
        query_lower = query.lower().strip()
        query_words = set(query_lower.split())
        
        # Create list of results with boosted scores
        boosted_items = []
        for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
            title = meta.get('title', '').lower()
            author = meta.get('author', '').lower()
            keywords = meta.get('keyword', '').lower()
            
            # Calculate title match score
            title_words = set(title.split())
            keyword_words = set(keywords.split(',')) if keywords else set()
            
            # Exact title match (highest boost)
            if query_lower in title:
                boost = -0.3  # Reduce distance significantly
            # Title starts with query
            elif title.startswith(query_lower):
                boost = -0.25
            # High word overlap in title
            elif len(query_words & title_words) >= len(query_words) * 0.8:
                boost = -0.2
            # Partial word overlap in title
            elif len(query_words & title_words) >= len(query_words) * 0.5:
                boost = -0.1
            # Author name match
            elif query_lower in author:
                boost = -0.15
            # Keyword match
            elif any(query_lower in kw.strip() for kw in keyword_words):
                boost = -0.05
            else:
                boost = 0
            
            boosted_items.append({
                'document': doc,
                'metadata': meta,
                'distance': max(0, dist + boost),  # Apply boost (lower distance = better match)
                'original_distance': dist
            })
        
        # Sort by boosted distance
        boosted_items.sort(key=lambda x: x['distance'])
        
        # Reformat to match original structure
        return {
            'documents': [[item['document'] for item in boosted_items]],
            'metadatas': [[item['metadata'] for item in boosted_items]],
            'distances': [[item['distance'] for item in boosted_items]]
        }

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


def display_book_card(book_data, rank, query, rag_system, summary_cache=None):
    """Display a professional library book card with relevance score"""
    metadata = book_data['metadata']
    document = book_data['document']
    distance = book_data['distance']
    chunk_type = metadata.get('chunk_type', 'publication_detail')
    
    # Calculate relevance score (0-100%)
    relevance_score = int((1 - distance) * 100)

    # Get description directly from metadata (CSV data)
    description = metadata.get('description', '')
    
    # If no description in metadata, extract from document
    if not description and 'Description:' in document:
        try:
            desc_line = [line for line in document.split('\n') if line.startswith('Description:')][0]
            description = desc_line.replace('Description:', '').strip()
        except:
            description = "No description available."
    
    # Fallback if still no description
    if not description:
        if chunk_type == 'author_summary':
            description = f"Publications by {metadata.get('author', 'this author')}."
        else:
            description = "Book information available at the library."

    # Determine card content based on chunk type
    if chunk_type == 'author_summary':
        card_title = metadata.get('author', 'NU Faculty')
        card_type = "Author Profile"
        author = metadata.get('author', '')
        button_url = f"https://elibrary.nu.edu.om/cgi-bin/koha/opac-search.pl?q=au:%22{author.replace(' ', '+')}%22"
    else:  # book_detail
        card_title = metadata.get('title', 'Library Book')
        card_type = "Book"
        author = metadata.get('author', 'Unknown Author')
        if author == 'Unknown Author' and 'Author:' in document:
            try:
                author_line = [line for line in document.split('\n') if line.startswith('Author:')][0]
                author = author_line.replace('Author:', '').strip()
            except:
                pass
        button_url = metadata.get('link', 'https://elibrary.nu.edu.om/')

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

    # Create professional book card (without relevance badge)
    st.markdown(f"""
    <div class="book-card">
        <div class="book-title">
            <a href="{button_url}" target="_blank">{rank}. {card_title}</a>
        </div>
        <div class="book-author">{author}</div>
        {f'<div class="book-metadata">{metadata_html}</div>' if metadata_html else ''}
        <div class="book-description">{description}</div>
        <a href="{button_url}" target="_blank" class="view-button">View in Library Catalog →</a>
    </div>
    """, unsafe_allow_html=True)


def keyword_search_tab(rag_system):
    """Professional search interface"""
    # Search interface
    st.markdown("### Search Library Catalog")
    
    # Search input
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input(
            "Search Query",
            placeholder="Enter keywords (e.g., research methodology, qualitative research, dissertation writing...)",
            key="keyword_search_input"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("Search", use_container_width=True, key="keyword_search_button", type="primary")
    
    # Perform search
    if query and (search_button or query):
        st.markdown("---")
        
        with st.spinner("🔍 Searching through hierarchical knowledge base..."):
            # Search for books with hierarchical chunking - show all relevant results
            results = rag_system.search_books(query, n_results=50)  # Show all relevant results

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
                st.markdown(f"**{books}** books and **{author_profiles}** author profiles found, sorted by relevance")
                st.markdown("---")

                # Display results directly (no AI summary generation needed)
                for i in range(len(results['documents'][0])):
                    book_data = {
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i]
                    }

                    display_book_card(book_data, i + 1, query, rag_system)

                # End of results
                st.markdown("---")
                st.info("📚 All results are sorted by relevance to your search query, with the most relevant items appearing first.")

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


def chatbot_tab(rag_system):
    """Conversational chatbot interface for book recommendations"""
    st.markdown("### 💬 AI Library Assistant")
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.chat_context = []
        st.session_state.processing_message = False
        st.session_state.current_user_input = None
    
    # Clear chat button at the top
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.session_state.chat_context = []
            st.session_state.processing_message = False
            st.session_state.current_user_input = None
    
    # Display welcome message
    if not st.session_state.chat_history and not st.session_state.processing_message:
        st.markdown("""
        <div style="background-color: #e8f4f8; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
            <h4 style="color: #192f59; margin-top: 0;">👋 Hello! I'm your AI Library Assistant</h4>
            <p style="color: #666; margin-bottom: 0.5rem;">
                I can help you discover research methodology books through natural conversation. Ask me questions like:
            </p>
            <ul style="color: #666;">
                <li>"I need books about qualitative research methods"</li>
                <li>"What resources do you have on dissertation writing?"</li>
                <li>"Show me highly rated books about research design"</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Display chat history in a container to keep it static
    history_container = st.container()
    with history_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display book recommendations if available (only for completed messages)
                if message["role"] == "assistant" and "books" in message:
                    st.markdown("---")
                    rating_sort = message.get("rating_sort")
                    if rating_sort == 'high':
                        st.markdown("**📚 Recommended Books (Sorted by Highest Rating):**")
                    elif rating_sort == 'low':
                        st.markdown("**📚 Recommended Books (Sorted by Lowest Rating):**")
                    elif rating_sort == 'average':
                        st.markdown("**📚 Recommended Books (Sorted by Average Rating):**")
                    else:
                        st.markdown(f"**📚 Recommended Books ({len(message['books'])} results):**")
                    for book in message["books"]:
                        display_chat_book_card(book)
    
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
                results = rag_system.search_books(user_input, n_results=50)
                
                if results['documents'][0]:
                    # Generate conversational response
                    response_text = generate_chat_response(rag_system, user_input, results, st.session_state.chat_context)
                    
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
                        "user": user_input,
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
    
    # Chat input
    user_input = st.chat_input("Ask me anything about books and resources...", key="chatbot_input")
    
    # Process user input
    if user_input:
        # Set processing flag and store input
        st.session_state.processing_message = True
        st.session_state.current_user_input = user_input
        st.rerun()


def generate_chat_response(rag_system, user_question, search_results, chat_context):
    """Generate conversational response based on user question and search results"""
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
        chunk_type = meta.get('chunk_type', 'book_detail')
        
        if title and title != 'Library Book':
            book_info = f"'{title}'"
            if author:
                book_info += f" by {author}"
            book_details.append(book_info)
        
        # Add document excerpt for context
        books_context += f"{i+1}. {doc[:400]}...\n\n"
    
    prompt = f"""You are an expert AI librarian assistant for National University eLibrary, designed to help researchers find the perfect books for their needs.

{context_str}

User's current question: "{user_question}"

{books_context}

Your task: Provide a warm, intelligent, and conversational response (4-5 sentences) that:

1. **Acknowledge their need**: Start by showing you understand what they're looking for
2. **Recommend specific books**: Mention 2-3 specific book titles by name from the search results
3. **Explain the value**: Briefly explain why each recommended book would be valuable for their research or study
4. **Provide guidance**: Suggest how they might use these resources effectively
5. **Invite follow-up**: End with an encouraging note that invites further questions

Style guidelines:
- Be conversational and warm, like a knowledgeable colleague helping a friend
- Use natural language, not robotic or overly formal
- Show enthusiasm about the books you're recommending
- Be specific about book titles and authors when mentioning them
- Keep it concise but informative

Remember: You're not just listing books, you're helping someone on their research journey."""
    
    try:
        response = rag_system.model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Enhanced fallback response with book details
        num_books = len(search_results['documents'][0])
        
        if book_details:
            if len(book_details) == 1:
                fallback = f"Great question! I found {book_details[0]} which looks perfect for what you're researching. "
            elif len(book_details) == 2:
                fallback = f"I've found some excellent resources for you! Check out {book_details[0]} and {book_details[1]}. "
            else:
                fallback = f"Perfect timing! I have several great recommendations including {book_details[0]}, {book_details[1]}, and {book_details[2]}. "
            
            fallback += "These books offer comprehensive coverage of your topic with practical insights and research-backed content. "
        else:
            fallback = f"I found {num_books} relevant resources that match your interest in '{user_question}'. "
        
        fallback += "Would you like me to help you narrow down to specific aspects of this topic, or do you have any other questions?"
        return fallback


def display_chat_book_card(book_data):
    """Display a detailed book card for chat interface (similar to search results)"""
    metadata = book_data['metadata']
    document = book_data['document']
    chunk_type = metadata.get('chunk_type', 'publication_detail')
    
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


def main():
    """Main application"""
    # Professional Library Header
    st.markdown("""
        <div class="library-header">
            <div class="library-title">National University eLibrary</div>
            <div class="library-subtitle">Intelligent Search System</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        with st.spinner("Initializing search system..."):
            st.session_state.rag_system = LibraryRAG()
    
    rag_system = st.session_state.rag_system
    
    # Tab selection
    st.markdown("<br>", unsafe_allow_html=True)
    selected_tab = st.radio(
        "Select Search Mode:",
        options=["🔍 Search", "💬 AI Assistant"],
        horizontal=True,
        key="main_tab_selector"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display selected tab content
    if selected_tab == "🔍 Search":
        keyword_search_tab(rag_system)
    else:  # AI Chatbot
        chatbot_tab(rag_system)
    
    # Professional Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 1.5rem; color: #718096; font-size: 0.875rem; background-color: #f7fafc; border-radius: 4px; margin-top: 2rem;">
            <div style="font-weight: 600; color: #2d3748; margin-bottom: 0.5rem;">National University eLibrary</div>
            <div>Azaiba, Bousher, Muscat, Sultanate of Oman</div>
            <div style="margin-top: 0.5rem;">
                <a href="https://elibrary.nu.edu.om/" target="_blank" style="color: #2c5282; text-decoration: none;">Visit Official Website</a> | 
                <a href="https://www.nu.edu.om/" target="_blank" style="color: #2c5282; text-decoration: none;">National University</a>
            </div>
            <div style="margin-top: 0.75rem; font-size: 0.8rem; color: #a0aec0;">
                © 2025 National University Libraries. All rights reserved.
            </div>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()