"""
PDF RAG Q&A System — Streamlit Web UI
======================================

Web interface for the PDF RAG pipeline:
  PDF Upload → Ingestion → ChromaDB / Keyword Search → Groq LLM → Answer

Run with:
    streamlit run streamlit_app.py
"""

import os
# Set environment variables to avoid PyTorch DLL issues on Windows
os.environ['TORCH_CUDA_ARCH_LIST'] = ''
os.environ['TORCH_USE_CUDA_DSA'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

import html
import shutil
import streamlit as st
import sys
import time
import logging
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration
from config import (
    DATA_DIR, VECTOR_DB_DIR, GROQ_MODEL_NAME, 
    MAX_CHUNKS_PER_QUERY, LOG_FORMAT, LOG_LEVEL
)

# Import modules
from ingestion.ingest_pdfs import process_pdf_documents, get_all_chunks
from rag.chroma_retrieval import (
    build_vector_database, 
    get_database_stats, 
    retrieve_relevant_chunks, 
    initialize_vector_store
)
from rag.groq_answering import generate_groq_answer, initialize_groq_generator
from utils.helpers import log_step, format_time_elapsed, ensure_directory_exists

# Set up logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        'processed_files': [],
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
        <h1>📄 PDF RAG Assistant</h1>
        <p>Clean PDF RAG Pipeline • Upload PDFs → ChromaDB → Groq AI • Production Ready</p>
    </div>
    """, unsafe_allow_html=True)


def sidebar_controls():
    """Display sidebar controls."""
    # Ensure session state is initialized before accessing
    initialize_session_state()
    
    st.sidebar.title("🎛️ Control Panel")
    
    # System Status
    st.sidebar.subheader("📊 System Status")
    
    # Check if ML dependencies are available to show appropriate status
    try:
        from rag.chroma_retrieval import ML_AVAILABLE, CHROMADB_AVAILABLE
        if not ML_AVAILABLE or not CHROMADB_AVAILABLE:
            status_color = "🟡"
            status_text = "Disabled (No ML)"
        else:
            status_color = "🟢" if st.session_state.get('vector_db_ready', False) else "🔴"
            status_text = 'Ready' if st.session_state.get('vector_db_ready', False) else 'Not Ready'
    except Exception:
        status_color = "🔴"
        status_text = "Not Ready"
    
    st.sidebar.markdown(f"{status_color} **Vector Database:** {status_text}")
    
    groq_color = "🟢" if st.session_state.get('groq_initialized', False) else "🔴"
    st.sidebar.markdown(f"{groq_color} **Groq AI:** {'Ready' if st.session_state.get('groq_initialized', False) else 'Not Ready'}")
    
    st.sidebar.markdown(f"📄 **Documents:** {len(st.session_state.get('processed_files', []))}")
    
    # Quick Actions
    st.sidebar.subheader("⚡ Quick Actions")
    
    if st.sidebar.button("🔄 Reset System", help="Clear all data and start fresh"):
        try:
            logger.info("Resetting system...")
            
            # Clean data directories using config paths
            # Remove and recreate directories
            for directory in [DATA_DIR, VECTOR_DB_DIR]:
                try:
                    if os.path.exists(directory):
                        shutil.rmtree(directory)
                    os.makedirs(directory, exist_ok=True)
                    logger.info("Cleaned directory: %s", directory)
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
    """Handle PDF document ingestion."""
    st.header("📄 PDF Document Ingestion")
    
    # File upload interface
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files to process"
    )
    
    if uploaded_files:
        st.success(f"📁 {len(uploaded_files)} PDF file(s) selected")
        
        # Show file details
        with st.expander("📋 File Details", expanded=True):
            for file in uploaded_files:
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.text(f"📄 {file.name}")
                with col2:
                    st.text(f"{file.size / 1024:.1f} KB")
                with col3:
                    st.text(file.type)
        
        # Processing options
        col1, col2 = st.columns([1, 1])
        with col1:
            chunk_size = st.slider("Chunk Size", 500, 2000, 1200, 100,
                                 help="Maximum characters per chunk")
        with col2:
            overlap = st.slider("Chunk Overlap", 0, 500, 200, 50,
                              help="Overlap between consecutive chunks")
        
        # Process button
        if st.button("🚀 Process PDF Documents", type="primary"):
            
            with st.spinner(f"📖 Processing {len(uploaded_files)} PDF files..."):
                
                # Prepare PDF data for processing
                pdf_files = []
                for uploaded_file in uploaded_files:
                    pdf_files.append((uploaded_file.name, uploaded_file.read()))
                
                # Process PDFs
                results = process_pdf_documents(pdf_files)
                
                if results:
                    st.session_state.processed_files = list(results.keys())
                    st.success(f"✅ Successfully processed {len(results)} PDF files!")
                    
                    # Show processing results
                    total_chunks = sum(len(chunks) for chunks in results.values())
                    st.info(f"📊 Created {total_chunks} text chunks from {len(results)} PDFs")
                    
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
                                st.metric("📁 PDF Sources", len(results))
                                
                            # Show detailed results
                            with st.expander("📋 Processing Details"):
                                for filename, chunk_files in results.items():
                                    st.markdown(f"**{filename}:** {len(chunk_files)} chunks")
                        else:
                            # Check if it's due to missing ML dependencies
                            stats = get_database_stats()
                            if stats.get('status') == 'dependencies_unavailable':
                                st.info("ℹ️ Vector search disabled - ML dependencies not available. PDF processing works perfectly!")
                            else:
                                st.error("❌ Failed to build vector database!")
                        
                        # Always show processing summary
                        st.success(f"📄 PDF Processing Summary: {len(results)} file(s) processed successfully")
                        with st.expander("📋 Processing Details"):
                            for filename, chunk_files in results.items():
                                st.markdown(f"**{filename}:** {len(chunk_files)} chunks")
                else:
                    st.error("❌ Failed to process PDF documents!")
    else:
        st.info("👆 Please upload PDF files to get started")
        
        # Show sample content types
        st.markdown("""
        **📚 Supported Content Types:**
        - Research papers and academic documents
        - Technical manuals and documentation  
        - Books and educational materials
        - Reports and white papers
        - Any text-heavy PDF content
        """)

def chat_interface():
    """Main chat interface."""
    # Ensure session state is initialized
    initialize_session_state()
    
    st.header("💬 Interactive Q&A")
    
    # Check if we have processed documents (either with vector DB or fallback mode)
    has_documents = len(st.session_state.get('processed_files', [])) > 0
    vector_db_ready = st.session_state.get('vector_db_ready', False)
    groq_ready = st.session_state.get('groq_initialized', False)
    
    if not has_documents:
        st.warning("⚠️ Please process some documents first to enable Q&A!")
        return
    
    if not groq_ready:
        st.warning("⚠️ Groq AI is not initialized. Please check your API key in config.py!")
        return
    
    # Show mode indicator
    if vector_db_ready:
        st.info("🧠 **Semantic Search Mode** - Using AI-powered vector search for best results")
    else:
        st.info("📝 **Fallback Mode** - Using keyword search (AI vector search unavailable)")
    
    # Display chat history
    for message in st.session_state.get('chat_history', []):
        if message['role'] == 'user':
            safe_content = html.escape(message['content'])
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>
                {safe_content}
            </div>
            """, unsafe_allow_html=True)
        else:
            safe_content = html.escape(message['content'])
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong><br>
                {safe_content}
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
                
                # Retrieve with k=10 for broader coverage
                chunks = retrieve_relevant_chunks(user_question, num_chunks=10)
                retrieval_method = "ChromaDB (Vector)" if st.session_state.get('vector_db_ready') else "Keyword Fallback"

                
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
                        'content': "❌ No relevant chunks found. Please re-process your documents or try a different question."
                    })
            
            st.rerun()

def show_content_summary():
    """Show content summary."""
    st.subheader("📝 Content Summary")
    
    try:
        if not os.path.exists(DATA_DIR):
            st.error("❌ No content loaded. Please run document ingestion first.")
            return
        
        chunk_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
        if not chunk_files:
            st.error("❌ No documents found in data directory.")
            return
        
        st.info(f"📊 Total Documents: {len(chunk_files)} chunks")
        
        # Content preview from first few chunks
        content_sample = ""
        for i, filename in enumerate(chunk_files[:3]):
            try:
                with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    content_sample += content[:500] + " "
            except Exception:
                continue
        
        if content_sample:
            preview = content_sample[:300].replace('\n', ' ').strip()
            if len(preview) > 297:
                preview = preview[:297] + "..."
            
            st.markdown("**📖 Content Preview:**")
            st.text_area("Preview", preview, height=100, disabled=True)
            
    except Exception as e:
        st.error(f"❌ Error generating summary: {str(e)}")

def show_faq_questions():
    """Show document-aware starter questions."""
    st.subheader("❓ Starter Questions")
    
    try:
        if not os.path.exists(DATA_DIR):
            st.error("❌ No content loaded.")
            return
        
        chunk_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')][:3]
        content_sample = ""
        
        for filename in chunk_files:
            try:
                with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf-8', errors='ignore') as f:
                    content_sample += f.read().lower()[:800]
            except Exception:
                continue
        
        # Generic document-aware questions that work for any PDF
        questions = [
            "What is the main topic of this document?",
            "Give me a full summary of this document.",
            "What are the key findings or conclusions?",
            "Who are the authors or contributors mentioned?",
            "What methods or techniques are described?",
            "What are the practical applications discussed?",
        ]
        
        st.markdown("**Here are some questions you can ask:**")
        
        for i, question in enumerate(questions, 1):
            if st.button(f"{i}. {question}", key=f"faq_{i}"):
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
    """Clean source citations to remove raw URLs from LLM responses."""
    import re
    
    # Pattern to match: (Source: https://any-url/ Chunk X/Y)
    pattern = r'\(Source:\s*https?://[^\s)]+\s+(Chunk\s+\d+/\d+)\)'
    cleaned = re.sub(pattern, r'(\1)', answer)
    
    pattern2 = r'^Source:\s*https?://[^\s]+\s+(Chunk\s+\d+/\d+)'
    cleaned = re.sub(pattern2, r'(\1)', cleaned, flags=re.MULTILINE)
    
    pattern3 = r'Source:\s*\[?Source\s*\d*:\s*https?://[^\]]+\s+(Chunk\s+\d+/\d+)\]?'
    cleaned = re.sub(pattern3, r'(\1)', cleaned)
    
    return cleaned

def main():
    """Main application."""
    initialize_session_state()

    display_header()
    sidebar_controls()

    # Create layout
    col1, col2 = st.columns([2, 3])

    with col1:
        if not st.session_state.get('processed_files', []):
            document_ingestion_section()
        else:
            st.success(f"✅ {len(st.session_state.get('processed_files', []))} PDF documents loaded")

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