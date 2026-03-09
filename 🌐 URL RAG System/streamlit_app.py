"""
Enterprise Knowledge Assistant - Streamlit Web UI
=================================================

Modern web interface for the RAG-based knowledge assistant.
Uses clean RAG pipeline: URLs → Ingestion → ChromaDB → Groq LLM → UI

Architecture:
- Production: ChromaDB vector store with semantic embeddings
- LLM: Groq Llama (Free tier)

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import os
import sys
import time
import logging
from typing import List, Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration
from config import (
    DATA_DIR, VECTOR_DB_DIR, GROQ_MODEL_NAME, 
    MAX_CHUNKS_PER_QUERY, LOG_FORMAT, LOG_LEVEL
)

# Import modules
from ingestion.ingest_docs import process_documentation, get_all_chunks
from rag.chroma_retrieval import (
    build_vector_database, 
    get_database_stats, 
    retrieve_relevant_chunks, 
    initialize_vector_store
)
from rag.groq_answering import generate_groq_answer, initialize_groq_generator
from utils.helpers import log_step, format_time_elapsed, validate_url, ensure_directory_exists

# Set up logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        'processed_urls': [],
        'vector_db_ready': False,
        'chat_history': [],
        'groq_initialized': False,
        'ingestion_complete': False,
        'current_question': ""
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.user-message {
    background-color: #e3f2fd;
    border-left: 4px solid #2196f3;
}
.assistant-message {
    background-color: #f1f8e9;
    border-left: 4px solid #4caf50;
}
.stats-box {
    background-color: #f5f5f5;
    padding: 1rem;
    border-radius: 5px;
    border-left: 3px solid #ff9800;
}
</style>
""", unsafe_allow_html=True)


def display_header():
    """Display the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Enterprise Knowledge Assistant</h1>
        <p>Clean RAG Pipeline • URLs → ChromaDB → Groq AI • Production Ready</p>
    </div>
    """, unsafe_allow_html=True)


def sidebar_controls():
    """Display sidebar controls."""
    # Ensure session state is initialized before accessing
    initialize_session_state()
    
    st.sidebar.title("🎛️ Control Panel")
    
    # System Status
    st.sidebar.subheader("📊 System Status")
    
    status_color = "🟢" if st.session_state.get('vector_db_ready', False) else "🔴"
    st.sidebar.markdown(f"{status_color} **Vector Database:** {'Ready' if st.session_state.get('vector_db_ready', False) else 'Not Ready'}")
    
    groq_color = "🟢" if st.session_state.get('groq_initialized', False) else "🔴"
    st.sidebar.markdown(f"{groq_color} **Groq AI:** {'Ready' if st.session_state.get('groq_initialized', False) else 'Not Ready'}")
    
    st.sidebar.markdown(f"📄 **Documents:** {len(st.session_state.get('processed_urls', []))}")
    
    # Quick Actions
    st.sidebar.subheader("⚡ Quick Actions")
    
    if st.sidebar.button("🔄 Reset System", help="Clear all data and start fresh"):
        try:
            logger.info("Resetting system...")
            
            # Clean data directories using config paths
            import shutil
            
            # Remove and recreate directories
            for directory in [DATA_DIR, VECTOR_DB_DIR]:
                try:
                    if os.path.exists(directory):
                        shutil.rmtree(directory)
                    os.makedirs(directory, exist_ok=True)
                    logger.info(f"Cleaned directory: {directory}")
                except Exception as e:
                    logger.warning(f"Could not clean {directory}: {e}")
            
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            # Reinitialize session state
            initialize_session_state()
            
            st.success("🧹 System reset complete! All data cleared.")
            logger.info("System reset completed successfully")
            
        except Exception as e:
            error_msg = f"Error during reset: {str(e)}"
            st.error(f"❌ {error_msg}")
            logger.error(error_msg)
    
    # System Info
    st.sidebar.subheader("ℹ️ System Info")
    st.sidebar.markdown(f"""
    - **Model:** {GROQ_MODEL_NAME}
    - **Max Chunks:** {MAX_CHUNKS_PER_QUERY}
    - **Storage:** ChromaDB Vector Store
    - **Data Directory:** {DATA_DIR}
    """)

def document_ingestion_section():
    """Handle document ingestion."""
    st.header("📥 Document Ingestion")
    
    with st.form("url_form"):
        st.markdown("**Enter documentation URLs to process:**")
        
        urls_text = st.text_area(
            "URLs (one per line)",
            placeholder="https://docs.python.org/3/tutorial/\nhttps://www.geeksforgeeks.org/machine-learning/",
            height=100
        )
        
        submit_button = st.form_submit_button("🚀 Process Documents")
        
        if submit_button and urls_text:
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            valid_urls = [url for url in urls if validate_url(url)]
            
            if not valid_urls:
                st.error("❌ No valid URLs provided!")
                return
            
            # Process documents
            with st.spinner(f"🔄 Processing {len(valid_urls)} URLs..."):
                results = process_documentation(valid_urls)
                
                if results:
                    st.session_state.processed_urls = list(results.keys())
                    st.success(f"✅ Successfully processed {len(results)} URLs!")
                    
                    # Build vector database
                    with st.spinner("🧠 Building vector database..."):
                        # Initialize ChromaDB
                        vector_ready = initialize_vector_store()
                        
                        # Initialize Groq
                        groq_generator = initialize_groq_generator()
                        st.session_state.groq_initialized = bool(groq_generator)
                        
                        # Build vector database
                        success = build_vector_database(force_rebuild=True)
                        st.session_state.vector_db_ready = success
                        
                        if success:
                            st.success("✅ Vector database built successfully!")
                            stats = get_database_stats()
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📚 Documents", stats.get('total_documents', 0))
                            with col2:
                                st.metric("📖 Vocabulary", stats.get('vocabulary_size', 0))
                            with col3:
                                st.metric("🌐 Sources", stats.get('unique_sources', 0))
                        else:
                            st.error("❌ Failed to build vector database!")
                else:
                    st.error("❌ Failed to process documents!")

def chat_interface():
    """Main chat interface."""
    # Ensure session state is initialized
    initialize_session_state()
    
    st.header("💬 Interactive Q&A")
    
    if not st.session_state.get('vector_db_ready', False):
        st.warning("⚠️ Please process some documents first to enable Q&A!")
        return
    
    # Display chat history
    for message in st.session_state.get('chat_history', []):
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>
                {message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong><br>
                {message['content']}
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        user_question = st.text_input("❓ Ask your question:", placeholder="What would you like to know?")
        submit_chat = st.form_submit_button("💬 Send")
        
        if submit_chat and user_question:
            # Ensure chat history exists
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # Add user message to history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_question
            })
            
            # Generate response
            with st.spinner("🤖 Generating answer..."):
                start_time = time.time()
                
                # Always use ChromaDB (production)
                chunks = retrieve_relevant_chunks(user_question, num_chunks=5)
                retrieval_method = "ChromaDB (Production)"

                
                if chunks:
                    # Generate answer
                    response = generate_groq_answer(user_question, chunks)
                    answer = response.get('answer', 'No answer generated')
                    
                    # Clean up answer
                    cleaned_answer = clean_source_citations(answer)
                    
                    # Add performance info
                    elapsed_time = time.time() - start_time
                    performance_info = f"""
                    
                    📊 **Performance Details:**
                    - ⚡ Response Time: {elapsed_time:.1f}s
                    - 🔍 Retrieval: {retrieval_method}
                    - 📄 Chunks Found: {len(chunks)}
                    - 💰 Cost: {response.get('cost', '$0.00 (FREE!)')}
                    - 🤖 Model: {response.get('method', 'Groq Llama-3.1-8B')}
                    """
                    
                    full_response = cleaned_answer + performance_info
                    
                    # Ensure chat history exists and add assistant response
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': full_response
                    })
                else:
                    # Ensure chat history exists
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': "❌ No relevant information found. Try rephrasing your question."
                    })
            
            st.rerun()

def show_content_summary():
    """Show content summary."""
    st.subheader("📝 Content Summary")
    
    try:
        data_dir = "data"
        if not os.path.exists(data_dir):
            st.error("❌ No content loaded. Please run document ingestion first.")
            return
        
        chunk_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
        if not chunk_files:
            st.error("❌ No documents found in data directory.")
            return
        
        st.info(f"📊 Total Documents: {len(chunk_files)} chunks")
        
        # Analyze content
        content_sample = ""
        for i, filename in enumerate(chunk_files[:3]):
            try:
                with open(os.path.join(data_dir, filename), 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    content_sample += content[:500] + " "
            except:
                continue
        
        if content_sample:
            # Extract key topics
            words = content_sample.lower().split()
            common_tech_words = ['algorithm', 'data', 'structure', 'graph', 'tree', 'node', 'vertex', 'edge', 
                               'python', 'programming', 'function', 'method', 'class', 'object', 'web', 'html', 
                               'css', 'javascript', 'database', 'sql', 'machine', 'learning', 'ai', 'model']
            
            found_topics = [word for word in set(words) if word in common_tech_words and len(word) > 3]
            
            if found_topics:
                st.markdown("**🔍 Key Topics Detected:**")
                topics_text = " • ".join([topic.capitalize() for topic in sorted(found_topics)[:10]])
                st.info(topics_text)
            
            # Content preview
            preview = content_sample[:300].replace('\n', ' ').strip()
            if len(preview) > 297:
                preview = preview[:297] + "..."
            
            st.markdown("**📖 Content Preview:**")
            st.text_area("Preview", preview, height=100, disabled=True)
            
    except Exception as e:
        st.error(f"❌ Error generating summary: {str(e)}")

def show_faq_questions():
    """Show FAQ questions."""
    st.subheader("❓ Frequently Asked Questions")
    
    try:
        # Generate FAQ based on content type
        data_dir = "data"
        if not os.path.exists(data_dir):
            st.error("❌ No content loaded.")
            return
        
        chunk_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')][:3]
        content_sample = ""
        
        for filename in chunk_files:
            try:
                with open(os.path.join(data_dir, filename), 'r', encoding='utf-8', errors='ignore') as f:
                    content_sample += f.read().lower()[:800]
            except:
                continue
        
        # Generate questions based on content
        questions = []
        if 'machine learning' in content_sample or 'algorithm' in content_sample:
            questions = [
                "What is machine learning?",
                "What are the types of machine learning algorithms?",
                "How does supervised learning work?",
                "What is the difference between classification and regression?",
                "What are common machine learning algorithms?",
                "What are some applications of Self-Supervised Learning?",
                "How do you evaluate machine learning models?"
            ]
        elif 'graph' in content_sample:
            questions = [
                "What is a graph data structure?",
                "How are graphs different from trees?",
                "What are the applications of graphs?",
                "What is BFS and DFS in graphs?",
                "How do you represent a graph?",
                "What are the types of graph algorithms?"
            ]
        else:
            questions = [
                "What is the main topic explained in this documentation?",
                "Can you provide an overview of the key concepts?",
                "What are the main features described?",
                "What are the practical applications?",
                "How does this technology work?",
                "What are the benefits and advantages?"
            ]
        
        st.markdown("**Here are some questions you can ask:**")
        
        for i, question in enumerate(questions, 1):
            if st.button(f"{i}. {question}", key=f"faq_{i}"):
                # Ensure chat history exists and add question
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': question
                })
                st.rerun()
                
    except Exception as e:
        st.error(f"❌ Error generating FAQ: {str(e)}")

def clean_source_citations(answer: str) -> str:
    """Clean source citations to remove URLs."""
    import re
    
    # Pattern to match: (Source: https://any-url/ Chunk X/Y)
    pattern = r'\(Source:\s*https?://[^\s)]+\s+(Chunk\s+\d+/\d+)\)'
    cleaned = re.sub(pattern, r'(\1)', answer)
    
    pattern2 = r'^Source:\s*https?://[^\s]+\s+(Chunk\s+\d+/\d+)'
    cleaned = re.sub(pattern2, r'(\1)', cleaned, flags=re.MULTILINE)
    
    pattern3 = r'Source:\s*\[?Source\s*\d*:\s*https?://[^\]]+\s+(Chunk\s+\d+/\d+)\]?'
    cleaned = re.sub(pattern3, r'(\1)', cleaned)
    
    return cleaned
    
    return cleaned

def main():
    """Main application."""
    # Ensure session state is initialized first
    initialize_session_state()
    
    # Verify critical session state variables exist
    if not hasattr(st.session_state, 'processed_urls') or st.session_state.processed_urls is None:
        st.session_state.processed_urls = []
    
    display_header()
    
    # Create layout
    col1, col2 = st.columns([2, 3])
    
    with col1:
        sidebar_controls()
        
        if not st.session_state.get('processed_urls', []):
            document_ingestion_section()
        else:
            st.success(f"✅ {len(st.session_state.get('processed_urls', []))} documents loaded")
            
            # Quick stats
            if st.session_state.get('vector_db_ready', False):
                stats = get_database_stats()
                st.markdown(f"""
                <div class="stats-box">
                    <strong>📊 Database Stats:</strong><br>
                    📚 Documents: {stats.get('total_documents', 0)}<br>
                    📖 Vocabulary: {stats.get('vocabulary_size', 0)} words<br>
                    🌐 Sources: {stats.get('unique_sources', 0)}
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        chat_interface()


# Initialize session state at module level
initialize_session_state()


if __name__ == "__main__":
    main()