"""
Modern RAG Retrieval System with ChromaDB
=========================================

This module implements a modern RAG retrieval system using ChromaDB for vector storage
and sentence-transformers for embeddings.

Features:
- ChromaDB for persistent vector storage
- SentenceTransformers for semantic embeddings
- Efficient similarity search
- Metadata filtering and source attribution
"""

import os
import logging
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

# Import configuration
from config import (
    VECTOR_DB_DIR, EMBEDDING_MODEL_NAME, CHROMA_COLLECTION_NAME,
    SIMILARITY_THRESHOLD, MAX_CHUNKS_PER_QUERY, LOG_FORMAT, LOG_LEVEL
)

# Set up logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """ChromaDB-based vector store for document retrieval."""
    
    def __init__(self, collection_name: str = None, persist_directory: str = None):
        """
        Initialize ChromaDB vector store.
        
        Args:
            collection_name (str): Name of the ChromaDB collection (uses config default if None)
            persist_directory (str): Directory to persist ChromaDB data (uses config default if None)
        """
        self.collection_name = collection_name or CHROMA_COLLECTION_NAME
        self.persist_directory = persist_directory or VECTOR_DB_DIR
        
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize embedding model
        logger.info("Loading sentence transformer model...")
        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Initialize ChromaDB client
        logger.info("Initializing ChromaDB...")
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info(f"ChromaDB client initialized at {self.persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
        
        # Get or create collection
        self._initialize_collection()
    
    def _initialize_collection(self) -> None:
        """Initialize the ChromaDB collection."""
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection '{self.collection_name}'")
        except Exception:
            # Collection doesn't exist, create it
            try:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}  # Use cosine distance
                )
                logger.info(f"Created new collection '{self.collection_name}'")
            except Exception as e:
                logger.warning(f"Could not create collection with metadata: {e}")
                # Try to get an existing one or create with basic settings
                try:
                    collections = self.client.list_collections()
                    if collections:
                        self.collection = collections[0]
                        logger.info(f"Using existing collection '{self.collection.name}'")
                    else:
                        # Create with minimal settings
                        self.collection = self.client.create_collection(name=self.collection_name)
                        logger.info(f"Created basic collection '{self.collection_name}'")
                except Exception as e2:
                    logger.error(f"Failed to initialize collection: {e2}")
                    raise
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents (List[str]): List of document texts
            metadatas (List[Dict], optional): List of metadata dictionaries
            ids (List[str], optional): List of document IDs
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.embedding_model.encode(documents, show_progress_bar=False)
        
        # Prepare metadata
        if metadatas is None:
            metadatas = [{"timestamp": datetime.now().isoformat()} for _ in documents]
        
        # Add to ChromaDB
        logger.debug("Adding documents to ChromaDB...")
        self.collection.add(
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Successfully added {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 5, where: Dict = None) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            where (Dict, optional): Metadata filters
            
        Returns:
            List[Dict[str, Any]]: Search results with documents, metadata, and scores
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []
        
        logger.debug(f"Searching for: '{query[:100]}...'")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:  # Check if results exist
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'distance': results['distances'][0][i]
                })
        
        logger.debug(f"Found {len(formatted_results)} relevant documents")
        return formatted_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        count = self.collection.count()
        
        # Get unique sources from metadata
        unique_sources = set()
        vocabulary_size = 0
        
        try:
            # Query all documents to get metadata
            all_docs = self.collection.get(include=["metadatas"])
            
            if all_docs.get('metadatas'):
                for metadata in all_docs['metadatas']:
                    if metadata and 'source' in metadata:
                        unique_sources.add(metadata['source'])
                
                # Estimate vocabulary size (approximate)
                vocabulary_size = min(count * 100, 10000) if count > 0 else 0
        
        except Exception as e:
            print(f"[WARNING] Could not get detailed stats: {e}")
        
        return {
            "total_documents": count,
            "vocabulary_size": vocabulary_size,
            "unique_sources": len(unique_sources),
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model.get_sentence_embedding_dimension(),
            "persist_directory": self.persist_directory
        }
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        print("🗑️ Clearing collection...")
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("[SUCCESS] Collection cleared")


# Global vector store instance
vector_store: Optional[ChromaVectorStore] = None


def initialize_vector_store(force_rebuild: bool = False) -> bool:
    """
    Initialize the global vector store.
    
    Args:
        force_rebuild (bool): Whether to rebuild the entire database
        
    Returns:
        bool: True if successful, False otherwise
    """
    global vector_store
    
    try:
        print("[INFO] Initializing ChromaDB vector store...")
        vector_store = ChromaVectorStore()
        
        # Check if we need to build/rebuild
        stats = vector_store.get_collection_stats()
        if stats['total_documents'] == 0 or force_rebuild:
            logger.info("Building vector database from documents...")
            return build_vector_database(force_rebuild)
        else:
            logger.info(f"Vector store ready with {stats['total_documents']} documents")
            return True
            
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        return False


def build_vector_database(force_rebuild: bool = False) -> bool:
    """
    Build vector database from document chunks.
    
    Args:
        force_rebuild (bool): Whether to force rebuild
        
    Returns:
        bool: True if successful, False otherwise
    """
    global vector_store
    
    if vector_store is None:
        vector_store = ChromaVectorStore()
    
    if force_rebuild:
        vector_store.clear_collection()
    
    logger.info("Loading documents from data directory...")
    
    from config import DATA_DIR
    data_dir = DATA_DIR
    if not os.path.exists(data_dir):
        logger.error(f"Data directory '{data_dir}' not found")
        return False
    
    # Load document chunks
    documents = []
    metadatas = []
    ids = []
    
    chunk_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    if not chunk_files:
        logger.error(f"No .txt files found in '{data_dir}'")
        return False
    
    for filename in chunk_files:
        filepath = os.path.join(data_dir, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract source URL and clean content
            lines = content.split('\\n')
            source_url = "unknown"
            content_start = 0
            
            # Find the source line and locate actual content start
            for i, line in enumerate(lines):
                if line.startswith("Source:"):
                    source_url = line.replace("Source:", "").strip()
                elif line.strip() == "--------------------------------------------------":  # Separator line
                    content_start = i + 1
                    break
            
            # If no separator found, try to find first non-empty content line after headers
            if content_start == 0:
                for i, line in enumerate(lines):
                    if i > 2 and line.strip():  # Skip first 3 lines (source, chunk, separator)
                        content_start = i
                        break
            
            clean_content = '\n'.join(lines[content_start:]).strip()
            
            if len(clean_content) > 50:  # Only add substantial content
                documents.append(clean_content)
                metadatas.append({
                    'filename': filename,
                    'source': source_url,
                    'filepath': filepath,
                    'chunk_size': len(clean_content),
                    'timestamp': datetime.now().isoformat()
                })
                ids.append(f"chunk_{len(documents):04d}")
        
        except Exception as e:
            logger.error(f"Error reading {filepath}: {str(e)}")
            continue
    
    if not documents:
        logger.error("No valid documents found")
        return False
    
    logger.info(f"Loaded {len(documents)} document chunks")
    
    # Add documents to vector store
    vector_store.add_documents(documents, metadatas, ids)
    
    logger.info("Vector database built successfully")
    return True


def retrieve_relevant_chunks(question: str, num_chunks: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve relevant document chunks for a question.
    
    Args:
        question (str): User question
        num_chunks (int): Number of chunks to retrieve
        
    Returns:
        List[Dict[str, Any]]: List of relevant chunks with metadata
    """
    global vector_store
    
    if vector_store is None:
        logger.warning("Vector store not initialized, initializing now...")
        if not initialize_vector_store():
            return []
    
    logger.debug(f"Retrieving {num_chunks} relevant chunks for: '{question[:50]}...'")
    
    # Search for relevant documents
    results = vector_store.search(question, top_k=num_chunks)
    
    # Format results for compatibility with existing code
    formatted_chunks = []
    for result in results:
        formatted_chunks.append({
            'text': result['document'],
            'source': result['metadata'].get('source', 'Unknown'),
            'filename': result['metadata'].get('filename', 'Unknown'),
            'similarity_score': result['similarity_score'],
            'metadata': result['metadata']
        })
    
    if formatted_chunks:
        logger.info(f"Retrieved {len(formatted_chunks)} relevant chunks")
        for i, chunk in enumerate(formatted_chunks, 1):
            logger.debug(f"  Chunk {i}: {chunk['similarity_score']:.3f} similarity - {chunk['source']}")
    else:
        logger.warning("No relevant chunks found")
    
    return formatted_chunks


def search_documents(query: str, top_k: int = 5, filters: Dict = None) -> Dict[str, Any]:
    """
    Search documents and return formatted results.
    
    Args:
        query (str): Search query
        top_k (int): Number of results to return
        filters (Dict, optional): Metadata filters
        
    Returns:
        Dict[str, Any]: Formatted search results
    """
    global vector_store
    
    if vector_store is None:
        if not initialize_vector_store():
            return {"chunks": [], "total_results": 0, "query": query}
    
    results = vector_store.search(query, top_k=top_k, where=filters)
    
    # Format for compatibility
    formatted_results = {
        "query": query,
        "total_results": len(results),
        "chunks": []
    }
    
    for result in results:
        formatted_results["chunks"].append({
            "text": result['document'],
            "source": result['metadata'].get('source', 'Unknown'),
            "filename": result['metadata'].get('filename', 'Unknown'),
            "similarity_score": result['similarity_score'],
            "metadata": result['metadata']
        })
    
    return formatted_results


def get_database_stats() -> Dict[str, Any]:
    """Get vector database statistics."""
    global vector_store
    
    if vector_store is None:
        return {"status": "not_initialized"}
    
    return {
        "status": "ready",
        **vector_store.get_collection_stats()
    }


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing ChromaDB RAG Retrieval System")
    print("=" * 50)
    
    # Initialize and build database
    if initialize_vector_store(force_rebuild=True):
        # Test queries
        test_queries = [
            "What is a function?",
            "How to define a variable?", 
            "Python loops",
            "Error handling"
        ]
        
        for query in test_queries:
            print(f"\\n❓ Testing query: '{query}'")
            print("-" * 40)
            
            chunks = retrieve_relevant_chunks(query, num_chunks=3)
            
            if chunks:
                for i, chunk in enumerate(chunks, 1):
                    print(f"\\n📄 Result {i}:")
                    print(f"  [STAT] Similarity: {chunk['similarity_score']:.3f}")
                    print(f"  📂 Source: {chunk['source']}")
                    print(f"  📝 Preview: {chunk['text'][:150]}...")
            else:
                print("❌ No results found")
        
        # Print statistics
        print(f"\n[STATS] Database Stats:")
        stats = get_database_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    else:
        print("❌ Failed to initialize vector store")