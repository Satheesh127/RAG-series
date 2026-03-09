"""
Enterprise Knowledge Assistant (RAG System)
==========================================

Main pipeline for RAG-based question answering:
- Document ingestion from URLs
- Vector database creation
- Interactive Q&A with Groq AI

This is the main entry point for the knowledge assistant.

Usage:
    python main.py

The system will:
1. Prompt for documentation URLs to ingest
2. Process and chunk the documentation
3. Build vector database for RAG
4. Enter interactive Q&A mode
"""

import os
import sys
import time
import logging
from typing import List, Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration
from config import (
    DATA_DIR, VECTOR_DB_DIR, GROQ_MODEL_NAME, EMBEDDING_MODEL_NAME,
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
from rag.groq_answering import (
    generate_groq_answer, 
    initialize_groq_generator
)
from utils.helpers import (
    log_step, format_time_elapsed, validate_url, ensure_directory_exists,
    summarize_file_stats
)

# Set up logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class KnowledgeAssistant:
    """
    Main class for the Enterprise Knowledge Assistant.
    
    This class orchestrates document ingestion, vector database creation,
    and provides the Q&A interface.
    """
    
    def __init__(self):
        """Initialize the Knowledge Assistant."""
        self.processed_urls = []
        self.vector_db_ready = False
        self.start_time = time.time()
        
        # Ensure required directories exist
        ensure_directory_exists(DATA_DIR)
        ensure_directory_exists(VECTOR_DB_DIR)
    
    def print_welcome(self):
        """Prints welcome message and system overview."""
        welcome_message = f"""
{'=' * 70}
🤖 ENTERPRISE KNOWLEDGE ASSISTANT
   RAG System with Groq AI
   Production Ready + Free Tier
{'=' * 70}

This system will:
📥 1. Ingest documentation from web sources
🧠 2. Build optimized vector database for Q&A
💬 3. Answer your questions using Groq AI

🔧 Technology Stack:
  📊 Vector Store: ChromaDB ({VECTOR_DB_DIR})
  🧠 Embeddings: {EMBEDDING_MODEL_NAME}
  🤖 LLM: {GROQ_MODEL_NAME}
  📁 Data Directory: {DATA_DIR}
  🔍 Max Chunks per Query: {MAX_CHUNKS_PER_QUERY}

🔍 Grounded answers only - if information isn't in the docs,
   the system will say: 'The knowledge base does not contain this information.'
        """
        print(welcome_message)
        logger.info("Knowledge Assistant initialized")
    
    def get_documentation_urls(self) -> List[str]:
        """
        Gets documentation URLs from user input.
        
        Returns:
            List[str]: List of valid URLs
        """
        print("📋 STEP 1: Documentation URLs")
        print("-" * 40)
        print("Enter documentation URLs to process (one per line).")
        print("Press Enter on empty line to finish.")
        print("Examples:")
        print("  - https://docs.python.org/3/tutorial/")
        print("  - https://docs.python.org/3/library/os.html")
        print()
        
        urls = []
        while True:
            url = input("Enter URL (or press Enter to finish): ").strip()
            
            if not url:  # Empty input, finish
                break
            
            if validate_url(url):
                urls.append(url)
                print(f"✅ Added: {url}")
            else:
                print(f"❌ Invalid URL: {url}")
        
        if not urls:
            print("\n📝 No URLs provided. Using sample Python documentation URLs...")
            urls = [
                "https://docs.python.org/3/tutorial/introduction.html",
                "https://docs.python.org/3/tutorial/controlflow.html",
                "https://docs.python.org/3/library/os.html"
            ]
            print("Sample URLs:")
            for url in urls:
                print(f"  - {url}")
        
        return urls
    
    def run_ingestion(self, urls: List[str]) -> bool:
        """
        Runs the document ingestion process.
        
        Args:
            urls (List[str]): URLs to process
            
        Returns:
            bool: True if successful
        """
        log_step("Starting document ingestion", 1)
        start = time.time()
        
        # Ensure data directory exists
        ensure_directory_exists(DATA_DIR)
        
        # Process documentation
        results = process_documentation(urls)
        
        if results:
            self.processed_urls = list(results.keys())
            total_chunks = sum(len(files) for files in results.values())
            
            print(f"\n✅ Ingestion complete!")
            print(f"   📄 Processed: {len(results)} URLs")
            print(f"   📦 Created: {total_chunks} text chunks")
            print(f"   ⏱️ Time: {format_time_elapsed(start)}")
            
            # Show data directory stats
            stats = summarize_file_stats(DATA_DIR)
            if "error" not in stats:
                print(f"   💾 Storage: {stats['total_size_mb']} MB in {stats['total_files']} files")
            
            return True
        else:
            print("❌ Ingestion failed - no documents were processed successfully")
            return False
    

    

    
    def build_vector_database(self) -> bool:
        """
        Builds the vector database for RAG.
        
        Returns:
            bool: True if successful
        """
        log_step("Building vector database for RAG", 2)
        start = time.time()
        
        # Ensure rag directory exists
        ensure_directory_exists("rag")
        
        # Initialize ChromaDB vector store (modern approach)
        print("📊 Initializing ChromaDB vector store...")
        vector_ready = initialize_vector_store()
        
        if not vector_ready:
            print("❌ Failed to initialize vector database")
            return False
        
        # Initialize Groq system for optimized token handling
        print("🤖 Initializing Groq AI system...")
        groq_generator = initialize_groq_generator()
        
        if not groq_generator:
            print("⚠️ Groq initialization failed - check API key in .env file")
            print("   The system can still work but answers may be limited")
        else:
            print("✅ Groq AI system ready (FREE tier with token optimization)")
        
        # Build vector database (legacy support)
        success = build_vector_database(force_rebuild=True)
        
        if success:
            self.vector_db_ready = True
            
            # Get database stats
            stats = get_database_stats()
            
            print(f"\n✅ Vector database built!")
            print(f"   🧠 Documents: {stats.get('total_documents', 0)}")
            print(f"   📚 Vocabulary: {stats.get('vocabulary_size', 0)} words")
            print(f"   🌐 Sources: {stats.get('unique_sources', 0)} unique URLs")
            print(f"   🔤 Token optimization: Enabled (3000 token limit)")
            print(f"   ⏱️ Time: {format_time_elapsed(start)}")
            
            return True
        else:
            print("❌ Vector database build failed")
            return False
    
    def show_examples(self):
        """Shows example outputs from all components - DISABLED for cleaner output."""
        # This method is disabled to provide cleaner console output

        pass
    
    def interactive_qa_mode(self):
        """Runs interactive Q&A mode."""
        if not self.vector_db_ready:
            print("❌ Vector database not ready. Cannot enter Q&A mode.")
            return
        
        print("\n" + "="*70)
        print("💬 INTERACTIVE Q&A MODE")
        print("="*70)
        print("Ask questions about the documentation!")
        print()
        print("Commands:")
        print("  • 'quit', 'exit', 'q' - Exit the system")
        print("  • 'summary' - Generate content summary")
        print("  • 'Give some questions based on the provided content' - faq")
        print("  • 'stats' - Show usage statistics")
        print("  • 'debug' - Show token optimization details")
        print("=" * 70)
        
        question_count = 0
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if not question:
                    continue
                
                # Check for exit commands
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                # Generate summary
                if question.lower() == 'summary':
                    self.generate_content_summary()
                    continue
                
                # Show FAQ
                if question.lower() == 'faq':
                    self.show_faq()
                    continue
                
                # Show statistics
                if question.lower() == 'stats':
                    print("\n📊 Usage Statistics:")
                    print(f"  📈 Questions Asked: {question_count}")
                    print(f"  💰 Total Cost: $0.00 (FREE!)")
                    print(f"  🤖 Model: Groq Llama-3.1-8B")
                    print(f"  🔤 Token Optimization: Enabled")
                    continue
                
                # Show debug info
                if question.lower() == 'debug':
                    print("\n🔧 Token Optimization Details:")
                    print("  1️⃣ Context limited to 3,000 tokens (top 3 chunks)")
                    print("  2️⃣ Code examples removed (saves 50-70% tokens)")
                    print("  3️⃣ Key sentences extracted with keyword prioritization")
                    print("  4️⃣ Smart truncation preserves sentence boundaries")
                    print("  ⚡ Emergency truncation at 5,500 tokens")
                    print("  🎯 Target: Under 6,000 token Groq FREE tier limit")
                    continue
                
                # Answer the question with token optimization
                question_count += 1
                print(f"\n🤖 Thinking... (Question #{question_count})")
                start = time.time()
                
                # Retrieve relevant chunks
                print("🔍 Searching knowledge base...")
                chunks = retrieve_relevant_chunks(question, num_chunks=5)
                
                if not chunks:
                    print("❌ No relevant information found. Try rephrasing your question.")
                    continue
                
                print(f"📚 Found {len(chunks)} relevant document(s)")
                
                # Generate answer with optimized Groq system
                print("🤖 Generating answer with Groq AI (FREE!)...")
                response = generate_groq_answer(question, chunks)
                
                print(f"\n💡 Answer:")
                # Clean up the answer to remove URLs from source citations
                answer = response.get('answer', 'No answer generated')
                cleaned_answer = self._clean_source_citations(answer)
                print(cleaned_answer)
                
                # Show token usage and performance details
                print(f"\n📊 Performance Details:")
                print(f"  ⚡ Response Time: {response.get('response_time', format_time_elapsed(start))}")
                print(f"  🔤 Tokens Used: {response.get('token_count', 'Unknown')}")
                print(f"  💰 Cost: {response.get('cost', '$0.00 (FREE!)')}")
                print(f"  🤖 Model: {response.get('method', 'Groq Llama-3.1-8B')}")
                
                # Show sources if available
                sources = response.get("sources", [])
                if sources:
                    print(f"📚 Sources:")
                    for i, source in enumerate(sources, 1):
                        if source and source != "Unknown":
                            print(f"  {i}. {source}")
                
                elapsed = format_time_elapsed(start)
                print(f"\n⏱️ Total processing time: {elapsed}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                continue
        
        print(f"\n📊 Session summary: Answered {question_count} questions")
    
    def generate_content_summary(self):
        """Generates a summary of the loaded content."""
        print("\n📝 CONTENT SUMMARY")
        print("-" * 40)
        
        try:
            data_dir = "data"
            if not os.path.exists(data_dir):
                print("❌ No content loaded. Please run document ingestion first.")
                return
            
            chunk_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
            if not chunk_files:
                print("❌ No documents found in data directory.")
                return
            
            print(f"📊 Total Documents: {len(chunk_files)} chunks")
            
            # Combine first few chunks for summary
            content_sample = ""
            for i, filename in enumerate(chunk_files[:3]):  # First 3 chunks
                try:
                    with open(os.path.join(data_dir, filename), 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        content_sample += content[:500] + " "  # First 500 chars of each
                except:
                    continue
            
            if content_sample:
                # Extract key topics from content
                words = content_sample.lower().split()
                # Simple keyword extraction
                common_tech_words = ['algorithm', 'data', 'structure', 'graph', 'tree', 'node', 'vertex', 'edge', 
                                   'python', 'programming', 'function', 'method', 'class', 'object', 'web', 'html', 
                                   'css', 'javascript', 'database', 'sql', 'machine', 'learning', 'ai', 'model']
                
                found_topics = [word for word in set(words) if word in common_tech_words and len(word) > 3]
                
                print(f"🔍 Key Topics Detected:")
                if found_topics:
                    for topic in sorted(found_topics)[:10]:
                        print(f"  • {topic.capitalize()}")
                else:
                    print("  • General technical documentation")
                
                # Quick content preview
                print(f"\n📖 Content Preview:")
                preview = content_sample[:300].replace('\n', ' ').strip()
                if len(preview) > 297:
                    preview = preview[:297] + "..."
                print(f"   {preview}")
                
            print(f"\n💡 Ask specific questions about these topics for detailed answers.")
            
        except Exception as e:
            print(f"❌ Error generating summary: {str(e)}")
    
    def show_faq(self):
        """Shows FAQ questions based on the loaded content."""
        print("\n❓ FREQUENTLY ASKED QUESTIONS")
        print("-" * 40)
        
        try:
            # Generate dynamic FAQ based on content
            faq_questions = self._generate_dynamic_faq()
            
            if faq_questions:
                print("Here are some questions you can ask based on your content:\n")
                for i, question in enumerate(faq_questions, 1):
                    print(f"{i}. {question}")
                
                print(f"\n💡 Tip: Copy and paste any question above to get a detailed answer!")
            else:
                print("❌ Could not generate FAQ. Please ensure documents are loaded.")
                
        except Exception as e:
            print(f"❌ Error generating FAQ: {str(e)}")
    
    def _generate_dynamic_faq(self) -> List[str]:
        """Generate FAQ questions based on the actual content."""
        try:
            data_dir = "data"
            if not os.path.exists(data_dir):
                return []
            
            # Analyze content from data directory
            content_sample = ""
            chunk_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')][:3]
            
            for filename in chunk_files:
                try:
                    with open(os.path.join(data_dir, filename), 'r', encoding='utf-8', errors='ignore') as f:
                        content_sample += f.read().lower()[:800]  # First 800 chars per file
                except:
                    continue
            
            # Generate questions based on content analysis
            if 'graph' in content_sample:
                return [
                    "What is a graph data structure?",
                    "How are graphs different from trees?",
                    "What are the applications of graphs?",
                    "What is BFS and DFS in graphs?",
                    "How do you represent a graph?",
                    "What are the types of graph algorithms?"
                ]
            elif 'machine learning' in content_sample or 'algorithm' in content_sample:
                return [
                    "What is machine learning?",
                    "What are the types of machine learning algorithms?",
                    "How does supervised learning work?",
                    "What is the difference between classification and regression?",
                    "What are common machine learning algorithms?",
                    "What are some applications of Self-Supervised Learning?",
                    "How do you evaluate machine learning models?"
                ]
            elif 'python' in content_sample and 'programming' in content_sample:
                return [
                    "How do you define a function in Python?",
                    "What are Python data types?",
                    "How to handle exceptions in Python?",
                    "What is the difference between lists and tuples?",
                    "How to work with files in Python?",
                    "What are Python modules and packages?"
                ]
            elif 'web' in content_sample or 'html' in content_sample:
                return [
                    "How does HTML structure web pages?",
                    "What are CSS selectors and properties?",
                    "How does JavaScript add interactivity?",
                    "What are responsive design principles?",
                    "How to optimize web page performance?",
                    "What are web accessibility best practices?"
                ]
            elif 'database' in content_sample or 'sql' in content_sample:
                return [
                    "What are the fundamental database concepts?",
                    "How do SQL queries work?",
                    "What are database relationships?",
                    "How to optimize database performance?",
                    "What are database normalization principles?",
                    "How to handle database transactions?"
                ]
            else:
                # Generic questions for any documentation
                return [
                    "What is the main topic explained in this documentation?",
                    "Can you provide an overview of the key concepts?",
                    "What are the main features described?",
                    "What are the practical applications?",
                    "How does this technology work?",
                    "What are the benefits and advantages?"
                ]
                
        except Exception as e:
            # Fallback questions
            return [
                "What is the main topic explained in this documentation?",
                "Can you provide an overview of the key concepts?",
                "What are the practical applications?",
                "How does this technology work?",
                "What are the benefits and advantages?"
            ]
    
    def show_help(self):
        """Shows help with dynamic sample questions based on content."""
        print("\n📋 SAMPLE QUESTIONS TO TRY:")
        print("-" * 40)
        
        # Generate dynamic questions based on actual content
        sample_questions = self._generate_dynamic_questions()
        
        for i, question in enumerate(sample_questions, 1):
            print(f"{i}. {question}")
        
        print("\n💡 Tips:")
        print("- Ask specific questions about the topics in your documentation")
        print("- Questions should relate to the documentation you provided")
        print("- If information isn't in the docs, I'll tell you honestly")
        print("- System uses FREE Groq API with token optimization")
        print("- Each answer costs $0.00 and responds in ~2-3 seconds")
    
    def _generate_dynamic_questions(self) -> List[str]:
        """Generate sample questions based on the actual content."""
        try:
            # Analyze content from data directory to generate relevant questions
            content_sample = ""
            data_dir = "data"
            
            if os.path.exists(data_dir):
                chunk_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')][:3]
                
                for filename in chunk_files:
                    try:
                        with open(os.path.join(data_dir, filename), 'r', encoding='utf-8', errors='ignore') as f:
                            content_sample += f.read().lower()[:1000]  # Sample first 1000 chars
                    except:
                        continue
            
            # Generate questions based on content type
            if 'machine learning' in content_sample or 'supervised' in content_sample:
                return [
                    "What is supervised machine learning?",
                    "What are the types of supervised learning?", 
                    "How does classification differ from regression?",
                    "What is the training process in supervised learning?",
                    "What are common supervised learning algorithms?",
                    "How do you evaluate a supervised learning model?",
                ]
            elif 'python' in content_sample and 'programming' in content_sample:
                return [
                    "How do you define a function in Python?",
                    "What are Python data types?",
                    "How to handle exceptions in Python?",
                    "What is the difference between lists and tuples?",
                    "How to work with files in Python?",
                    "What are Python modules and packages?",
                ]
            elif 'web' in content_sample or 'html' in content_sample or 'css' in content_sample:
                return [
                    "How does HTML structure web pages?",
                    "What are CSS selectors and properties?",
                    "How does JavaScript add interactivity?",
                    "What are responsive design principles?",
                    "How to optimize web page performance?",
                    "What are web accessibility best practices?",
                ]
            elif 'database' in content_sample or 'sql' in content_sample:
                return [
                    "What are the fundamental database concepts?",
                    "How do SQL queries work?",
                    "What are database relationships?",
                    "How to optimize database performance?",
                    "What are database normalization principles?",
                    "How to handle database transactions?",
                ]
            else:
                # Universal questions that work for any type of documentation
                return [
                    "What is the main topic or concept explained?",
                    "Can you provide a summary?",
                    "What are the key features or characteristics?",
                    "What are the main benefits and advantages?",
                    "What are the common use cases or applications?",
                    "How does this work or function?",
                ]
                
        except Exception as e:
            # Fallback to universal questions
            return [
                "What is the main topic or concept explained?",
                "Can you provide a summary?",
                "What are the key features or characteristics?",
                "What are the main benefits and advantages?",
                "What are the common use cases or applications?",
                "How does this work or function?",
            ]
    
    def _clean_source_citations(self, answer: str) -> str:
        """Clean source citations to remove URLs and keep only chunk information."""
        import re
        
        # Pattern to match: (Source: https://any-url/ Chunk X/Y)
        # Replace with: (Chunk X/Y)
        pattern = r'\(Source:\s*https?://[^\s)]+\s+(Chunk\s+\d+/\d+)\)'
        cleaned = re.sub(pattern, r'(\1)', answer)
        
        # Also handle cases where Source: is at the beginning of a line
        pattern2 = r'^Source:\s*https?://[^\s]+\s+(Chunk\s+\d+/\d+)'
        cleaned = re.sub(pattern2, r'(\1)', cleaned, flags=re.MULTILINE)
        
        # Handle cases with "Source:" followed by URL and chunk on separate lines
        pattern3 = r'Source:\s*\[?Source\s*\d*:\s*https?://[^\]]+\s+(Chunk\s+\d+/\d+)\]?'
        cleaned = re.sub(pattern3, r'(\1)', cleaned)
        
        return cleaned
    
    def show_final_summary(self):
        """Shows final summary of the session."""
        total_time = format_time_elapsed(self.start_time)
        
        print("\n" + "="*70)
        print("🎉 KNOWLEDGE ASSISTANT SESSION COMPLETE")
        print("="*70)
        print(f"⏱️ Total runtime: {total_time}")
        print(f"📄 URLs processed: {len(self.processed_urls)}")
        
        print(f"🧠 Vector database: {'Ready' if self.vector_db_ready else 'Failed'}")
        print(f"🤖 Groq AI: FREE tier with token optimization")
        print(f"💰 Total cost: $0.00")
        
        print("\n📁 Generated files:")
        files_to_check = [
            (VECTOR_DB_DIR, "ChromaDB vector database"),
        ]
        
        for filepath, description in files_to_check:
            if os.path.exists(filepath):
                print(f"   ✅ {filepath} - {description}")
            else:
                print(f"   ❌ {filepath} - {description} (not created)")
        
        print("\n🔄 To run again: python main.py")
        print("👋 Thank you for using the Enterprise Knowledge Assistant!")
    
    def run(self):
        """Main execution method."""
        try:
            # Welcome
            self.print_welcome()
            
            # Step 1: Get URLs
            urls = self.get_documentation_urls()
            
            # Step 2: Ingest documentation
            if not self.run_ingestion(urls):
                print("❌ Cannot continue without successful ingestion")
                return
            
            # Step 3: Build vector database
            if not self.build_vector_database():
                print("❌ Cannot continue without vector database")
                return
            
            # Step 4: Interactive Q&A
            self.interactive_qa_mode()
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Process interrupted by user")
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)}")
            print("📝 Please check the error and try again")
        finally:
            # Always show summary
            self.show_final_summary()


def main():
    """Main function - entry point of the application."""
    assistant = KnowledgeAssistant()
    assistant.run()


# Run the application
if __name__ == "__main__":
    main()