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
# Set environment variables to avoid PyTorch DLL issues on Windows
os.environ['TORCH_CUDA_ARCH_LIST'] = ''
os.environ['TORCH_USE_CUDA_DSA'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

import re
import math
import logging
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

# Import configuration first so LOG_LEVEL is available
from config import (
    DATA_DIR, VECTOR_DB_DIR, EMBEDDING_MODEL_NAME, CHROMA_COLLECTION_NAME,
    SIMILARITY_THRESHOLD, MAX_CHUNKS_PER_QUERY, LOG_FORMAT, LOG_LEVEL
)

# Create the module logger before optional imports so early errors are captured
logger = logging.getLogger(__name__)

# Try to import ML and ChromaDB dependencies (optional for testing)
ML_AVAILABLE = False
CHROMADB_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
    logger.info("ChromaDB loaded successfully")
except Exception as e:
    logger.warning("ChromaDB not available: %s", type(e).__name__)
    chromadb = None
    Settings = None

try:
    from sentence_transformers import SentenceTransformer
    ML_AVAILABLE = True
    logger.info("ML dependencies loaded successfully")
except Exception as e:
    logger.warning(
        "ML dependencies not available: %s: %s — vector search disabled",
        type(e).__name__, str(e)[:100]
    )
    SentenceTransformer = None


class ChromaVectorStore:
    """ChromaDB-based vector store for document retrieval."""
    
    def __init__(self, collection_name: str = None, persist_directory: str = None):
        """
        Initialize ChromaDB vector store.
        
        Args:
            collection_name (str): Name of the ChromaDB collection (uses config default if None)
            persist_directory (str): Directory to persist ChromaDB data (uses config default if None)
        """
        if not CHROMADB_AVAILABLE or not ML_AVAILABLE:
            raise RuntimeError("ChromaDB or ML dependencies not available")
            
        self.collection_name = collection_name or CHROMA_COLLECTION_NAME
        self.persist_directory = persist_directory or VECTOR_DB_DIR
        
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize embedding model
        if ML_AVAILABLE:
            logger.info("Loading sentence transformer model...")
            try:
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        else:
            logger.warning("ML dependencies not available - vector search disabled")
            self.embedding_model = None
        
        # Initialize ChromaDB client
        if CHROMADB_AVAILABLE:
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
        else:
            logger.warning("ChromaDB not available")
            self.client = None
        
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
        if not self.embedding_model:
            logger.warning("ML dependencies not available - cannot generate embeddings")
            return
        
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
        
        # Check if ML is available
        if not self.embedding_model:
            logger.warning("ML dependencies not available - search disabled")
            return []
        
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
            logger.warning("Could not get detailed stats: %s", e)
        
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
        logger.info("Clearing ChromaDB collection '%s'", self.collection_name)
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Collection cleared and recreated")


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
    
    if not ML_AVAILABLE or not CHROMADB_AVAILABLE:
        logger.warning("⚠️ ML or ChromaDB dependencies not available - vector search disabled")
        return False
    
    try:
        logger.info("Initializing ChromaDB vector store...")
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
    
    if not ML_AVAILABLE or not CHROMADB_AVAILABLE:
        logger.warning("⚠️ ML or ChromaDB dependencies not available - cannot build vector database")
        return False
    
    if vector_store is None:
        try:
            vector_store = ChromaVectorStore()
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            return False
    
    if force_rebuild:
        vector_store.clear_collection()
    
    logger.info("Loading documents from data directory...")
    
    from config import DATA_DIR
    data_dir = DATA_DIR
    if not os.path.exists(data_dir):
        logger.error(f"Data directory '{data_dir}' not found")
        return False
    
    chunk_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.txt')])
    if not chunk_files:
        logger.error(f"No .txt files found in '{data_dir}'")
        return False

    # --- persistence check: skip docs already in ChromaDB ---
    try:
        existing = vector_store.collection.get(include=[])
        existing_ids = set(existing.get('ids', []))
    except Exception:
        existing_ids = set()

    documents, metadatas, ids = [], [], []

    for filename in chunk_files:
        doc_id = f"doc_{filename}"
        if not force_rebuild and doc_id in existing_ids:
            logger.debug(f"Skipping already-embedded: {filename}")
            continue

        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')
            meta = {'filename': filename, 'filepath': filepath,
                    'source': 'unknown', 'page': '', 'section': ''}
            content_start = 0

            for i, line in enumerate(lines):
                if line.startswith('Source:'):
                    meta['source'] = line[7:].strip()
                elif line.startswith('Page:'):
                    raw = line[5:].strip()
                    meta['page'] = int(raw) if raw.isdigit() else raw
                elif line.startswith('Section:'):
                    meta['section'] = line[8:].strip()
                elif line.startswith('-' * 10):
                    content_start = i + 2
                    break

            # Legacy format fallback (no Page/Section headers)
            if content_start == 0:
                for i, line in enumerate(lines):
                    if i > 2 and line.strip():
                        content_start = i
                        break

            clean_content = '\n'.join(lines[content_start:]).strip()

            if len(clean_content) > 50:
                documents.append(clean_content)
                meta['chunk_size'] = len(clean_content)
                meta['timestamp'] = datetime.now().isoformat()
                metadatas.append(meta)
                ids.append(doc_id)

        except Exception as e:
            logger.error(f"Error reading {filepath}: {str(e)}")

    if not documents:
        if existing_ids:
            logger.info(f"All {len(existing_ids)} documents already embedded — skipping re-embedding")
            return True
        logger.error("No valid documents found")
        return False

    logger.info(f"Embedding {len(documents)} new document chunks (skipped {len(existing_ids)} existing)")
    vector_store.add_documents(documents, metadatas, ids)
    logger.info("Vector database built successfully")
    return True



def fallback_text_search(question: str, num_chunks: int = 8) -> List[Dict[str, Any]]:
    """
    Keyword-based fallback search used when vector search is unavailable.
    Retries with a broader threshold if the first pass returns too few results.
    Returns chunks with page/section metadata.
    """
    try:
        import re
        from pathlib import Path

        data_dir = Path(DATA_DIR)
        if not data_dir.exists():
            logger.warning(f"Data directory not found: {data_dir}")
            return []

        chunk_files = list(data_dir.glob("*chunk*.txt"))
        if not chunk_files:
            logger.warning("No chunk files found in data directory")
            return []

        keywords = re.findall(r'\b\w+\b', question.lower())
        keywords = [w for w in keywords if len(w) > 2]

        # Expand keywords for common query types
        expanded = keywords.copy()
        kw_set = set(keywords)
        if kw_set & {'supervisor', 'guide', 'advisor', 'mentor'}:
            expanded += ['dr.', 'professor', 'supervision', 'guidance', 'sathish', 'kumar']
        if kw_set & {'author', 'student', 'writer', 'name'}:
            expanded += ['satheesh', 'declaration', 'register', 'signature']
        if kw_set & {'project', 'title', 'topic', 'subject'}:
            expanded += ['wind', 'turbine', 'fault', 'detection', 'system']
        if kw_set & {'college', 'university', 'institution', 'institute'}:
            expanded += ['eshwar', 'anna', 'coimbatore']
        if kw_set & {'summary', 'abstract', 'overview', 'introduction'}:
            expanded += ['abstract', 'introduction', 'overview']

        def score_file(chunk_file):
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    raw = f.read()

                # Parse metadata
                lines = raw.split('\n')
                meta = {'source': 'Unknown', 'page': '', 'section': ''}
                text_start = 0
                for idx, line in enumerate(lines):
                    if line.startswith('Source:'):
                        meta['source'] = line[7:].strip()
                    elif line.startswith('Page:'):
                        raw_p = line[5:].strip()
                        meta['page'] = int(raw_p) if raw_p.isdigit() else raw_p
                    elif line.startswith('Section:'):
                        meta['section'] = line[8:].strip()
                    elif line.startswith('-' * 10):
                        text_start = idx + 2
                        break

                text = '\n'.join(lines[text_start:]).strip()
                if not text:
                    return None

                tl = text.lower()
                score = sum(tl.count(kw) * 3.0 for kw in keywords)
                score += sum(tl.count(kw) * 1.5 for kw in expanded)
                # partial match bonus
                score += sum(len(re.findall(rf'\b\w*{kw}\w*', tl)) * 0.5
                             for kw in keywords if len(kw) > 4)

                # Special bonus: supervisor-related
                if any(t in tl for t in ['supervisor', 'guide', 'dr.']) and \
                   any(t in tl for t in ['sathish', 'kumar']):
                    score += 5.0

                return (score, text, meta, str(chunk_file))
            except Exception as e:
                logger.warning(f"Error scoring {chunk_file}: {e}")
                return None

        scored = [score_file(f) for f in chunk_files]
        scored = [s for s in scored if s and s[0] > 0]

        # If too few results, relax to any non-empty chunk (generic queries like "summary")
        if len(scored) < 3:
            logger.info("Too few keyword matches \u2014 broadening to top chunks by file order")
            scored_all = [score_file(f) for f in sorted(chunk_files)[:num_chunks + 2]]
            for s in scored_all:
                if s and s not in scored:
                    scored.append(s)

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, text, meta, fpath in scored[:num_chunks]:
            results.append({
                'text':             text,
                'source':           meta['source'],
                'filename':         Path(fpath).name,
                'page':             meta['page'],
                'section':          meta['section'],
                'similarity_score': round(score, 2),
                'metadata':         {**meta, 'search_method': 'keyword_fallback'},
            })

        logger.info(f"[FALLBACK] Keyword search returned {len(results)} chunks")
        for i, r in enumerate(results[:5], 1):
            logger.info(f"  Chunk {i}: score={r['similarity_score']}  page={r['page']}  file={r['filename']}")

        return results

    except Exception as e:
        logger.error(f"Error in fallback text search: {e}")
        return []


def _bm25_score(query: str, text: str, corpus_texts: List[str],
                k1: float = 1.5, b: float = 0.75) -> float:
    """
    Compute a BM25 score for (query, document) given the full corpus.
    Used for the keyword leg of hybrid search.
    """
    tokens_q = re.findall(r'\b\w+\b', query.lower())
    tokens_d = re.findall(r'\b\w+\b', text.lower())
    if not tokens_q or not tokens_d:
        return 0.0

    doc_len = len(tokens_d)
    avg_dl = sum(len(re.findall(r'\b\w+\b', t.lower())) for t in corpus_texts) / max(len(corpus_texts), 1)
    N = len(corpus_texts)

    score = 0.0
    freq_map: Dict[str, int] = {}
    for tok in tokens_d:
        freq_map[tok] = freq_map.get(tok, 0) + 1

    for term in set(tokens_q):
        tf = freq_map.get(term, 0)
        df = sum(1 for t in corpus_texts if term in t.lower())
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / max(avg_dl, 1)))
        score += idf * tf_norm

    return score


def _hybrid_search(vector_results: List[Dict], question: str,
                   all_chunk_texts: List[str], alpha: float = 0.6) -> List[Dict]:
    """
    Fuse vector similarity scores with BM25 keyword scores using
    Reciprocal Rank Fusion (RRF) + weighted linear combination.

    alpha controls the blend:
      final_score = alpha * vector_score + (1-alpha) * bm25_norm_score
    """
    if not vector_results:
        return []

    # Compute BM25 scores for all vector candidates
    bm25_scores = [
        _bm25_score(question, r.get('document', r.get('text', '')), all_chunk_texts)
        for r in vector_results
    ]

    # Normalise BM25 to [0, 1]
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
    bm25_norm = [s / max_bm25 for s in bm25_scores]

    # Normalise vector scores to [0, 1]
    vec_scores = [r.get('similarity_score', 0.0) for r in vector_results]
    max_vec = max(vec_scores) if max(vec_scores) > 0 else 1.0
    vec_norm = [s / max_vec for s in vec_scores]

    # Fuse
    for i, result in enumerate(vector_results):
        result['bm25_score']   = round(bm25_scores[i], 4)
        result['hybrid_score'] = round(alpha * vec_norm[i] + (1 - alpha) * bm25_norm[i], 4)
        # keep original for logging
        result['vector_score'] = result.get('similarity_score', 0.0)
        result['similarity_score'] = result['hybrid_score']

    return sorted(vector_results, key=lambda x: x['hybrid_score'], reverse=True)


def _rerank_chunks(question: str, chunks: List[Dict], top_n: int) -> List[Dict]:
    """
    Cross-encoder style reranking using token-overlap + positional signals.
    Replaces an external cross-encoder model so no extra dependencies are needed.

    Scoring factors:
      - Exact phrase match bonus (highest weight)
      - Query term density in chunk
      - Presence of named entities / numbers (factual signal)
      - Penalty for very short chunks
    """
    if not chunks:
        return []

    q_words    = re.findall(r'\b\w+\b', question.lower())
    q_tokens   = set(q_words)
    q_trigrams = {' '.join(q_words[i:i+3]) for i in range(len(q_words) - 2)}

    def rerank_score(chunk: Dict) -> float:
        text = chunk.get('text', '')
        tl   = text.lower()
        t_tokens = re.findall(r'\b\w+\b', tl)
        t_set    = set(t_tokens)

        # 1. Token overlap ratio
        overlap = len(q_tokens & t_set) / max(len(q_tokens), 1)

        # 2. Exact trigram matches (signals the chunk directly addresses the query)
        trigram_hits = sum(1 for tg in q_trigrams if tg in tl)

        # 3. Named entity / number density (factual richness)
        numbers    = len(re.findall(r'\b\d+\b', text))
        caps_words = len(re.findall(r'\b[A-Z][a-z]+\b', text))
        factual    = min((numbers + caps_words) / max(len(t_tokens), 1), 0.3)

        # 4. Chunk length penalty (very short = low info)
        length_bonus = min(len(text) / 800, 1.0)

        score = (overlap * 0.5) + (trigram_hits * 0.2) + (factual * 0.15) + (length_bonus * 0.15)

        # 5. Page/section boost: prefer chunks with known page numbers
        if chunk.get('page') or chunk.get('metadata', {}).get('page'):
            score += 0.05

        return round(score, 4)

    for chunk in chunks:
        chunk['rerank_score'] = rerank_score(chunk)

    reranked = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)

    logger.info(f"[RERANK] Top {min(top_n, len(reranked))} chunks after reranking:")
    for i, c in enumerate(reranked[:top_n], 1):
        page = c.get('page') or c.get('metadata', {}).get('page', '?')
        logger.info(
            f"  #{i}  rerank={c['rerank_score']:.3f}  "
            f"hybrid={c.get('hybrid_score', c.get('similarity_score', 0)):.3f}  "
            f"page={page}  file={c.get('filename', '?')}"
        )

    return reranked[:top_n]


def _mmr_filter(results: List[Dict], num_results: int, lambda_mult: float = 0.5) -> List[Dict]:
    """
    Max Marginal Relevance: balance relevance with diversity.
    lambda_mult=1.0 → pure relevance; 0.0 → pure diversity.
    """
    if len(results) <= num_results:
        return results

    import math

    def text_similarity(a: str, b: str) -> float:
        """Simple Jaccard word-overlap similarity between two texts."""
        wa = set(a.lower().split())
        wb = set(b.lower().split())
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)

    selected = [results[0]]  # always keep top result
    remaining = results[1:]

    while len(selected) < num_results and remaining:
        best_score = -1.0
        best_idx = 0
        for i, candidate in enumerate(remaining):
            relevance = candidate.get('similarity_score', 0.0)
            cand_text = candidate.get('document', candidate.get('text', ''))
            max_sim = max(
                text_similarity(cand_text, s.get('document', s.get('text', '')))
                for s in selected
            )
            mmr = lambda_mult * relevance - (1 - lambda_mult) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        selected.append(remaining.pop(best_idx))

    return selected


def retrieve_relevant_chunks(question: str, num_chunks: int = 10) -> List[Dict[str, Any]]:
    """
    Full hybrid RAG retrieval pipeline:
      1. Vector search (ChromaDB)  → semantic candidates
      2. Hybrid fusion            → blend vector + BM25 keyword scores
      3. MMR filter               → diversity
      4. Cross-encoder rerank     → final precision ordering

    Falls back to pure keyword search when ML is unavailable.
    """
    global vector_store

    if not ML_AVAILABLE or not CHROMADB_AVAILABLE:
        logger.info("ML/ChromaDB unavailable — using keyword fallback search")
        results = fallback_text_search(question, num_chunks)
        return _rerank_chunks(question, results, num_chunks)

    if vector_store is None:
        logger.warning("Vector store not initialized, initializing now...")
        if not initialize_vector_store():
            logger.warning("Vector store init failed — falling back to keyword search")
            results = fallback_text_search(question, num_chunks)
            return _rerank_chunks(question, results, num_chunks)

    # ── Step 1: Vector search (fetch more candidates for fusion) ──────────────
    fetch_k = max(num_chunks * 2, 20)  # over-fetch for reranking
    vector_results = vector_store.search(question, top_k=fetch_k)

    if len(vector_results) < 3:
        logger.info(f"Only {len(vector_results)} vector results, retrying with k=30")
        vector_results = vector_store.search(question, top_k=30)

    # ── Step 2: Hybrid fusion (vector + BM25) ─────────────────────────────────
    all_texts = [r.get('document', '') for r in vector_results]
    fused = _hybrid_search(vector_results, question, all_texts, alpha=0.6)

    # ── Step 3: MMR diversity filter ──────────────────────────────────────────
    diverse = _mmr_filter(fused, min(num_chunks + 5, len(fused)), lambda_mult=0.7)

    # Convert to uniform chunk dict format
    candidates = []
    for r in diverse:
        doc_text = r.get('document', r.get('text', ''))
        meta     = r.get('metadata', {})
        candidates.append({
            'text':             doc_text,
            'source':           meta.get('source', 'Unknown'),
            'filename':         meta.get('filename', 'Unknown'),
            'page':             meta.get('page', ''),
            'section':          meta.get('section', ''),
            'similarity_score': r.get('hybrid_score', r.get('similarity_score', 0.0)),
            'vector_score':     r.get('vector_score', 0.0),
            'bm25_score':       r.get('bm25_score', 0.0),
            'metadata':         meta,
        })

    # ── Step 4: Cross-encoder rerank ──────────────────────────────────────────
    final = _rerank_chunks(question, candidates, num_chunks)

    # ── Debug summary ─────────────────────────────────────────────────────────
    logger.info(f"[HYBRID] Question : '{question[:80]}'")
    logger.info(f"[HYBRID] Pipeline : vector({len(vector_results)}) "
                f"→ fused({len(fused)}) → mmr({len(diverse)}) → reranked({len(final)})")

    return final



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
    
    if not ML_AVAILABLE or not CHROMADB_AVAILABLE:
        return {
            "status": "dependencies_unavailable",
            "total_documents": 0,
            "vocabulary_size": 0,
            "unique_sources": 0
        }
    
    if vector_store is None:
        return {"status": "not_initialized"}
    
    try:
        return {
            "status": "ready",
            **vector_store.get_collection_stats()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "total_documents": 0,
            "vocabulary_size": 0,
            "unique_sources": 0
        }


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("Testing ChromaDB RAG Retrieval System")

    if initialize_vector_store(force_rebuild=True):
        test_queries = [
            "What is a function?",
            "How to define a variable?",
            "Python loops",
            "Error handling",
        ]

        for query in test_queries:
            logger.info("Testing query: '%s'", query)
            chunks = retrieve_relevant_chunks(query, num_chunks=3)

            if chunks:
                for i, chunk in enumerate(chunks, 1):
                    logger.info(
                        "  Result %d | score=%.3f | source=%s | preview=%s…",
                        i, chunk['similarity_score'], chunk['source'], chunk['text'][:120],
                    )
            else:
                logger.warning("  No results found")

        stats = get_database_stats()
        logger.info("Database stats: %s", stats)
    else:
        logger.error("Failed to initialize vector store")