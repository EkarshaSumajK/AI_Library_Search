# 📚 NU eLibrary RAG Chatbot

AI-powered search and recommendation system for National University eLibrary using RAG (Retrieval Augmented Generation) with production-level conversational AI.

## 🎯 Recent Updates (Production-Ready)

### ✅ Data Source Migration
- **Migrated to `new_data.csv`**: Now using comprehensive CSV dataset with 41 research methodology books
- **Removed old ChromaDB**: Cleaned up previous database to ensure fresh start
- **Enhanced metadata**: Added publisher, publication year, ISBN, rating, pages, language, format, and research context

### ✅ Improved Search & RAG
- **Increased result count**: Now showing 10 results (up from 4) for comprehensive coverage
- **Better relevance scoring**: All results shown based on semantic similarity
- **Enhanced metadata display**: Shows publisher, year, rating, pages, and language for each book
- **Hierarchical chunking**: Keyword summaries → Author profiles → Individual books

### ✅ Production-Level Chatbot
- **ChatGPT-like conversation**: Natural, warm, and intelligent responses
- **Book recommendations**: Mentions specific book titles by name in responses
- **Context-aware**: Remembers last 3 conversation exchanges
- **More recommendations**: Shows 5 book suggestions (up from 3)
- **Enhanced prompts**: Detailed instructions for high-quality AI responses
- **Fallback handling**: Graceful degradation when API quota exceeded

### ✅ Better User Experience
- **Rich book cards**: Display all relevant metadata (rating, year, publisher, etc.)
- **Improved summaries**: Concise, focused AI-generated summaries
- **Visual enhancements**: Better formatting and information hierarchy
- **Faster loading**: Optimized batch processing for summaries

## 🌟 Features

- **🔍 Semantic Search**: Find books using natural language queries with relevance scoring
- **🤖 AI-Powered Recommendations**: Get intelligent book suggestions based on your research needs
- **💬 Conversational Chatbot**: Chat with an AI librarian for personalized recommendations (ChatGPT-style)
- **📊 Hierarchical Results**: View results organized by keyword overviews, author profiles, and individual books
- **⚡ Fast Performance**: Optimized with caching and parallel processing
- **📚 Rich Metadata**: Publisher, year, rating, ISBN, pages, language, and more
- **🎯 Context-Aware**: Chatbot remembers conversation history for better recommendations
- **🎨 Modern UI**: Beautiful Streamlit interface with responsive design

## 🏗️ Technology Stack

- **Frontend**: Streamlit
- **Vector Database**: ChromaDB (local, persistent)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **LLM**: Google Gemini 2.5 Flash
- **Data Source**: CSV file with comprehensive book metadata

## 📋 Prerequisites

- Python 3.8+
- Google API Key (for Gemini 2.5 Flash)

## 🚀 Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd /path/to/poc
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up secrets**

   **Option 1: Local Development**
   Create a `.streamlit/secrets.toml` file in the project root:
   ```toml
   GOOGLE_API_KEY = "your_gemini_api_key_here"
   ```

   **Option 2: Streamlit Cloud Deployment**
   - Go to your app dashboard on [share.streamlit.io](https://share.streamlit.io)
   - Click on "Settings" > "Secrets"
   - Add your `GOOGLE_API_KEY` directly in the dashboard

   To get a Gemini API key:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy and paste it into your secrets configuration

## 📊 Data Ingestion

Before running the app, you need to ingest the data into ChromaDB:

```bash
python data_ingestion.py
```

This will:
- Read all books from `new_data.csv` (41 research methodology books)
- Create hierarchical chunks (keyword summaries, author profiles, individual books)
- Generate embeddings using Sentence Transformers
- Store everything in a local ChromaDB database (82 documents total)

Expected output:
```
🚀 Starting NU eLibrary Data Ingestion...
   (Hierarchical chunking: Keyword → Author → Book levels)
🗑️  Cleared existing collection

📁 Processing CSV file: ./new_data.csv
   Found 41 book records

🏷️  Processing keyword: research methodology, beginners... (X books)
...

💾 Adding 82 documents to ChromaDB...
✅ Successfully loaded 82 documents into ChromaDB

📊 Collection Statistics:
   Total Documents: 82
   Collection Name: nu_library_books

✨ Data ingestion complete!
```

## 🧪 Testing

Test the RAG system after ingestion:

```bash
python test_rag.py
```

Expected output:
```
🧪 Testing RAG System...
✅ Collection found with 82 documents

🔍 Testing search for 'research methodology'...
✅ Found 5 results

📖 First result:
   Title: Research Methodology: An Introduction
   Author: Stuart Melville, Wayne Goddard
   Chunk Type: book_detail
   Relevance Score: 0.296

✅ RAG system is working correctly!
```

## 🎯 Running the Application

### Local Development

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

### Streamlit Cloud Deployment

1. **Push to GitHub**: Make sure your code is in a GitHub repository
2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set your main file path to `app.py`
3. **Configure Secrets**:
   - In your app dashboard, go to "Settings" > "Secrets"
   - Add your `GOOGLE_API_KEY`
4. **Deploy**: Click "Deploy" and wait for the build to complete

## 💡 Usage

### Keyword Search Tab

1. **Enter a search query** in the search box
   - Example: "qualitative research methods"
   - Example: "dissertation writing"
   - Example: "mixed methods research"

2. **Click Search** or press Enter

3. **View Results**:
   - AI-generated analysis of your query
   - Top 10 most relevant results (mix of keyword summaries, author profiles, and books)
   - AI-generated summaries for each result
   - Relevance scores
   - Rich metadata (publisher, year, rating, pages, language)
   - Direct links to NU eLibrary

### AI Chatbot Tab

1. **Ask questions naturally**:
   - "I need books about machine learning for beginners"
   - "What resources do you have on healthcare management?"
   - "Can you recommend books about research methodology?"

2. **Get conversational responses**:
   - AI mentions specific book titles by name
   - Explains why each book would be helpful
   - Provides guidance on using the resources
   - Remembers conversation context

3. **View book recommendations**:
   - 5 recommended books displayed as cards
   - Quick access to book details
   - Direct links to NU eLibrary

## 📁 Project Structure

```
poc/
├── app.py                          # Main Streamlit application
├── data_ingestion.py               # Data loading and processing (CSV → ChromaDB)
├── test_rag.py                     # Test script for RAG system
├── new_data.csv                    # Primary data source (41 books)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .streamlit/
│   └── secrets.toml               # Secrets configuration (create this)
└── chroma_db/                     # ChromaDB storage (82 documents, auto-created)
```

## 📚 Data Structure

### CSV Fields (new_data.csv)

Each book includes:
- **Core Info**: Title, Author(s), Publisher, Publication Year
- **Metadata**: ISBN, Pages, Language, Format, Rating (out of 5.0)
- **Content**: Description (detailed summary), Keywords (comma-separated), Context (research context)
- **Access**: Link to NU eLibrary

### Hierarchical Chunking

The system creates 3 levels of chunks:

1. **Level 1 - Keyword Summaries**: Overview of all books for a keyword
   - Total books and authors
   - Language distribution
   - Top publishers
   - Publication years

2. **Level 2 - Author Profiles**: Summary of an author's books
   - Total books by author
   - Publishers and years
   - Book titles

3. **Level 3 - Individual Books**: Full book details
   - Complete metadata
   - Description and context
   - All searchable fields

## 🔧 Configuration

### Secrets Configuration

**Required Secrets:**
- `GOOGLE_API_KEY`: Your Google Gemini API key (required)

**Setup:**
- For local development: Create `.streamlit/secrets.toml`
- For Streamlit Cloud: Use the app settings dashboard

### Search Parameters

- **Keyword Search**: Top 10 results (configurable in `app.py`)
- **Chatbot**: Top 5 book recommendations (configurable in `app.py`)
- **Embedding Model**: `all-MiniLM-L6-v2` (runs locally, no API needed)
- **Embedding Dimension**: 384 (automatic)

## 🎨 Features in Detail

### Semantic Search
The system uses sentence transformers to create embeddings of both the query and documents, enabling semantic understanding beyond keyword matching.

### RAG Architecture
1. **Retrieval**: ChromaDB finds top results using semantic similarity
2. **Augmentation**: Context is enriched with hierarchical chunks and metadata
3. **Generation**: Gemini 2.5 Flash generates conversational responses and summaries

### Hierarchical Results
- **Keyword Summaries**: Provide subject-level context
- **Author Profiles**: Show research focus and expertise
- **Individual Books**: Offer specific details and full metadata

### Production-Level Chatbot
- **Natural conversation**: Warm, friendly, and intelligent responses
- **Specific recommendations**: Mentions book titles by name
- **Context awareness**: Remembers conversation history
- **Fallback handling**: Graceful degradation when API limits reached

## 🐛 Troubleshooting

### "Database not found" error
Run `python data_ingestion.py` first to create the database.

### "GOOGLE_API_KEY not found" error
Make sure you've set up your secrets correctly:
- For local development: Create `.streamlit/secrets.toml` with your API key
- For Streamlit Cloud: Set the secret in your app settings

### No results found
- Make sure data ingestion completed successfully
- Check that `chroma_db/` directory exists and contains data
- Run `python test_rag.py` to verify the system

### Slow first search
First search might be slow as models load. Subsequent searches will be faster due to caching.

### API quota exceeded
The system has fallback responses when Gemini API quota is exceeded. Responses will still be generated but without AI enhancement.

## 🚀 Future Enhancements

- [ ] Add more books to the dataset
- [ ] Implement user feedback system
- [ ] Add book availability tracking
- [ ] Support for multiple languages
- [ ] Advanced filtering (by year, publisher, rating)
- [ ] Export search results to PDF/CSV
- [ ] Citation export (BibTeX, APA, etc.)
- [ ] Bookmark favorite books
- [ ] Reading list management

## 📝 License

This project is created for educational purposes. All book data is sourced from publicly available information.

## 🙏 Acknowledgments

- National University Libraries for inspiration
- Google for Gemini 2.5 Flash API
- ChromaDB team for the excellent vector database
- Streamlit for the amazing web framework
- Sentence Transformers for local embeddings

## 📧 Contact

For questions or issues, please refer to the [NU eLibrary](https://elibrary.nu.edu.om/) official website.

---

**Built with ❤️ using Gemini 2.5 Flash, ChromaDB, and Streamlit**

**Latest Update**: Production-ready chatbot with ChatGPT-like conversation quality
