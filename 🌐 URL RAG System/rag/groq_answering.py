"""
Groq LLM Answering Module
========================

This module provides AI answering using the Groq API with proper error handling,
logging, and configuration management.

Features:
- Free Groq API integration
- Efficient token management
- Robust error handling
- Proper logging

Usage:
    from rag.groq_answering import generate_groq_answer
    
    chunks = [...] # Retrieved document chunks
    answer = generate_groq_answer("Your question", chunks)
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Import configuration
from config import (
    GROQ_MODEL_NAME, MAX_TOKENS_PER_REQUEST, MAX_RESPONSE_TOKENS,
    MAX_CHUNKS_PER_QUERY, LOG_FORMAT, LOG_LEVEL
)

# Set up logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Try to import required libraries
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, using fallback token estimation")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
    logger.info("Groq library loaded successfully")
except ImportError:
    GROQ_AVAILABLE = False
    logger.error("Groq library not found. Install with: pip install groq")


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.
    
    Args:
        text (str): Text to count tokens for
        
    Returns:
        int: Estimated token count
    """
    if not text:
        return 0
        
    try:
        if TIKTOKEN_AVAILABLE:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"tiktoken estimation failed: {e}")
    
    # Fallback to word-based estimation
    words = len(text.split())
    return int(words * 0.75)


def remove_code_examples(text: str) -> str:
    """
    Remove code blocks and examples to save tokens.
    
    Args:
        text (str): Text to process
        
    Returns:
        str: Text with code examples removed
    """
    if not text:
        return text
        
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '[Code example removed]', text)
    text = re.sub(r'`[^`]+`', '[Code removed]', text)
    
    # Remove common code patterns
    text = re.sub(r'#include.*?\n', '', text)
    text = re.sub(r'import.*?;?\n', '', text)
    text = re.sub(r'using namespace.*?;', '', text)
    text = re.sub(r'class\s+\w+.*?{.*?}', '[Class definition removed]', text, flags=re.DOTALL)
    text = re.sub(r'function.*?{.*?}', '[Function removed]', text, flags=re.DOTALL)
    
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    
    return text.strip()


def extract_key_sentences(text: str, max_sentences: int = 3) -> str:
    """
    Extract most relevant sentences from text.
    
    Args:
        text (str): Text to process
        max_sentences (int): Maximum sentences to extract
        
    Returns:
        str: Extracted key sentences
    """
    if not text:
        return text
        
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    # Prioritize sentences with key terms
    key_terms = [
        'graph', 'vertex', 'edge', 'adjacency', 'matrix', 'list', 'algorithm', 
        'data structure', 'represent', 'connection', 'node', 'implementation', 
        'time complexity', 'space complexity'
    ]
    
    # Score sentences based on key terms
    scored_sentences = []
    for sentence in sentences:
        score = sum(1 for term in key_terms if term.lower() in sentence.lower())
        scored_sentences.append((score, sentence))
    
    # Sort by score and take top sentences
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    top_sentences = [sentence for _, sentence in scored_sentences[:max_sentences]]
    
    return '. '.join(top_sentences) + '.' if top_sentences else text[:200] + '...'


def smart_truncate(text: str, max_chars: int = 600) -> str:
    """
    Intelligently truncate text while preserving meaning.
    
    Args:
        text (str): Text to truncate
        max_chars (int): Maximum characters
        
    Returns:
        str: Truncated text
    """
    if not text or len(text) <= max_chars:
        return text
    
    # Try to cut at sentence boundaries
    sentences = text.split('.')
    result = ""
    
    for sentence in sentences:
        if len(result + sentence + '.') <= max_chars:
            result += sentence + '.'
        else:
            break
    
    if not result:
        # If no complete sentences fit, cut at word boundary
        words = text[:max_chars].split()
        result = ' '.join(words[:-1]) + '...'
    
    return result.strip()


class GroqAnswerGenerator:
    """
    Groq-powered answer generator with proper error handling and logging.
    """
    
    def __init__(self, model: str = None):
        """
        Initialize Groq client.
        
        Args:
            model (str): Model name to use
        """
        self.model = model or GROQ_MODEL_NAME
        self.client = None
        self.usage_stats = {
            'total_questions': 0,
            'total_tokens': 0,
            'average_response_time': 0.0
        }
        
        if GROQ_AVAILABLE:
            api_key = os.getenv('GROQ_API_KEY')
            if api_key:
                try:
                    self.client = Groq(api_key=api_key)
                    logger.info(f"Groq client initialized with {self.model}")
                except Exception as e:
                    logger.error(f"Failed to initialize Groq client: {e}")
            else:
                logger.error("GROQ_API_KEY not found in environment variables")
        else:
            logger.error("Groq library not available")
    
    def generate_answer(self, question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate answer using Groq API.
        
        Args:
            question (str): User's question
            context_chunks (List[Dict]): List of relevant document chunks
            
        Returns:
            Dict[str, Any]: Answer response with metadata
        """
        start_time = datetime.now()
        
        try:
            # Format context from chunks with token limits
            context = self._format_context(context_chunks)
            
            # Create prompt
            prompt = self._create_prompt(question, context)
            
            # Validate token count
            prompt_tokens = estimate_tokens(prompt)
            logger.info(f"Prompt token count: {prompt_tokens}")
            
            if prompt_tokens > MAX_TOKENS_PER_REQUEST:
                logger.warning(f"Prompt exceeds token limit ({prompt_tokens} > {MAX_TOKENS_PER_REQUEST}), truncating")
                prompt = self._truncate_prompt(prompt)
            
            if not self.client:
                logger.warning("Groq client not available, using fallback")
                return self._fallback_answer(question, context_chunks)
            
            logger.info(f"Generating answer using {self.model}")
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that answers questions based ONLY on the provided context. Be accurate and cite sources."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=MAX_RESPONSE_TOKENS,
                temperature=0.3,
                top_p=0.9
            )
            
            # Extract answer
            answer = response.choices[0].message.content.strip()
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Update statistics
            token_usage = getattr(response.usage, 'total_tokens', 0) if hasattr(response, 'usage') else 0
            self._update_stats(response_time, token_usage)
            
            # Calculate confidence
            confidence = self._calculate_confidence(question, context_chunks)
            
            logger.info(f"Answer generated successfully in {response_time:.2f}s")
            
            return {
                'question': question,
                'answer': answer,
                'confidence': confidence,
                'sources': [chunk.get('source', 'Unknown') for chunk in context_chunks[:3]],
                'method': f'Groq {self.model}',
                'response_time': response_time,
                'token_usage': token_usage,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return self._fallback_answer(question, context_chunks, error=str(e))
    
    def _format_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Format context from chunks with token limits.
        
        Args:
            context_chunks (List[Dict]): Document chunks
            
        Returns:
            str: Formatted context
        """
        if not context_chunks:
            return "No relevant context found."
        
        # Limit to top chunks
        top_chunks = sorted(
            context_chunks, 
            key=lambda x: x.get('similarity_score', 0), 
            reverse=True
        )[:MAX_CHUNKS_PER_QUERY]
        
        context_parts = []
        total_tokens = 0
        max_context_tokens = MAX_TOKENS_PER_REQUEST // 2  # Reserve half for question and response
        
        for i, chunk in enumerate(top_chunks, 1):
            text = chunk.get('text', '').strip()
            source = chunk.get('source', 'Unknown')
            
            if not text:
                continue
            
            # Process the chunk text to reduce tokens
            processed_text = self._process_chunk_text(text)
            
            # Count tokens for this chunk
            chunk_text = f"[Source {i}: {source}]\n{processed_text}"
            chunk_tokens = estimate_tokens(chunk_text)
            
            # Check if adding this chunk exceeds limit
            if total_tokens + chunk_tokens > max_context_tokens:
                if total_tokens < max_context_tokens:
                    # Try to fit a truncated version
                    remaining_tokens = max_context_tokens - total_tokens
                    if remaining_tokens > 100:
                        remaining_chars = remaining_tokens * 4
                        processed_text = smart_truncate(processed_text, remaining_chars)
                        context_parts.append(f"[Source {i}: {source}]\n{processed_text}")
                break
            
            context_parts.append(chunk_text)
            total_tokens += chunk_tokens
        
        result = "\n\n".join(context_parts)
        logger.debug(f"Context processed: {estimate_tokens(result)} tokens from {len(top_chunks)} chunks")
        
        return result
    
    def _process_chunk_text(self, text: str) -> str:
        """
        Process chunk text to reduce token count while preserving key information.
        
        Args:
            text (str): Raw text to process
            
        Returns:
            str: Processed text
        """
        # Apply text processing pipeline
        text = remove_code_examples(text)
        text = extract_key_sentences(text, max_sentences=4)
        text = smart_truncate(text, max_chars=600)
        
        # Clean up formatting
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'\[.*?\]', '', text)  # Remove reference brackets
        
        return text.strip()
    
    def _create_prompt(self, question: str, context: str) -> str:
        """
        Create prompt for Groq API.
        
        Args:
            question (str): User's question
            context (str): Formatted context
            
        Returns:
            str: Complete prompt
        """
        return f"""Based on the following context, please answer the question accurately and concisely.

**CONTEXT:**
{context}

**QUESTION:** {question}

**INSTRUCTIONS:**
1. Answer based ONLY on the provided context
2. If the context doesn't contain enough information, say so
3. Be specific and cite which source(s) you're using
4. Keep the answer clear and helpful

**ANSWER:**"""
    
    def _truncate_prompt(self, prompt: str) -> str:
        """
        Emergency truncation of prompt to fit token limits.
        
        Args:
            prompt (str): Original prompt
            
        Returns:
            str: Truncated prompt
        """
        target_words = int(MAX_TOKENS_PER_REQUEST / 0.75)
        words = prompt.split()
        
        if len(words) > target_words:
            truncated = ' '.join(words[:target_words])
            truncated += "\n\n[Context truncated to fit token limits]"
            logger.warning(f"Prompt truncated from {len(words)} to {target_words} words")
            return truncated
        
        return prompt
    
    def _calculate_confidence(self, question: str, context_chunks: List[Dict]) -> float:
        """
        Calculate confidence score based on context relevance.
        
        Args:
            question (str): User's question
            context_chunks (List[Dict]): Available context chunks
            
        Returns:
            float: Confidence score between 0 and 1
        """
        if not context_chunks:
            return 0.0
        
        # Base confidence on similarity scores
        avg_similarity = sum(chunk.get('similarity_score', 0) for chunk in context_chunks) / len(context_chunks)
        
        # Boost confidence based on number of relevant chunks
        chunk_bonus = min(len(context_chunks) * 0.1, 0.3)
        
        # Question-context relevance
        question_words = set(question.lower().split())
        context_text = ' '.join(chunk.get('text', '') for chunk in context_chunks).lower()
        context_words = set(context_text.split())
        
        overlap = len(question_words.intersection(context_words))
        relevance_bonus = min(overlap * 0.05, 0.2)
        
        final_confidence = min(avg_similarity + chunk_bonus + relevance_bonus, 0.95)
        return round(final_confidence, 2)
    
    def _update_stats(self, response_time: float, tokens: int) -> None:
        """
        Update usage statistics.
        
        Args:
            response_time (float): Response time in seconds
            tokens (int): Token count used
        """
        self.usage_stats['total_questions'] += 1
        self.usage_stats['total_tokens'] += tokens
        
        # Update average response time
        current_avg = self.usage_stats['average_response_time']
        total_questions = self.usage_stats['total_questions']
        self.usage_stats['average_response_time'] = (
            (current_avg * (total_questions - 1) + response_time) / total_questions
        )
    
    def _fallback_answer(self, question: str, context_chunks: List[Dict], error: str = None) -> Dict[str, Any]:
        """
        Fallback to rule-based answering if Groq fails.
        
        Args:
            question (str): User's question
            context_chunks (List[Dict]): Available context
            error (str): Error message if any
            
        Returns:
            Dict[str, Any]: Fallback answer response
        """
        try:
            # Try importing alternative answer generator
            from rag.free_llm_answering import generate_free_answer
            logger.info("Using fallback rule-based answering")
            return generate_free_answer(question, context_chunks)
        except ImportError:
            logger.warning("No fallback answering system available")
            
            # Simple fallback response
            context = self._format_context(context_chunks)
            fallback_answer = (
                f"Based on the available context: {context[:300]}..." 
                if context else "No relevant information found in the knowledge base."
            )
            
            return {
                'question': question,
                'answer': fallback_answer,
                'confidence': 0.5 if context else 0.0,
                'sources': [chunk.get('source', 'Unknown') for chunk in context_chunks[:3]],
                'method': 'Fallback (Rule-based)',
                'error': error,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Dict[str, Any]: Usage statistics
        """
        return {
            **self.usage_stats,
            'model': self.model,
            'provider': 'Groq'
        }


# Global instance
_groq_generator = None


def initialize_groq_generator(model: str = None) -> GroqAnswerGenerator:
    """
    Initialize the global Groq generator.
    
    Args:
        model (str): Model name to use
        
    Returns:
        GroqAnswerGenerator: Initialized generator
    """
    global _groq_generator
    
    model = model or os.getenv('GROQ_MODEL', GROQ_MODEL_NAME)
    
    logger.info(f"Initializing Groq answer generator with {model}")
    _groq_generator = GroqAnswerGenerator(model=model)
    logger.info("Groq answer generator ready")
    
    return _groq_generator


def generate_groq_answer(question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Main function to generate answers using Groq API.
    
    Args:
        question (str): User's question
        context_chunks (List[Dict]): List of relevant document chunks
        
    Returns:
        Dict[str, Any]: Answer dictionary with metadata
    """
    global _groq_generator
    
    if _groq_generator is None:
        _groq_generator = initialize_groq_generator()
    
    return _groq_generator.generate_answer(question, context_chunks)


def format_groq_response(response: Dict[str, Any]) -> str:
    """
    Format Groq response for display.
    
    Args:
        response (Dict[str, Any]): Response from generate_groq_answer
        
    Returns:
        str: Formatted response text
    """
    answer = response.get('answer', 'No answer generated.')
    confidence = response.get('confidence', 0.0)
    sources = response.get('sources', [])
    response_time = response.get('response_time', 0.0)
    
    formatted = f"📝 **Answer:**\n{answer}\n\n"
    
    if sources:
        formatted += "📚 **Sources:**\n"
        for i, source in enumerate(sources, 1):
            formatted += f"  {i}. {source}\n"
        formatted += "\n"
    
    formatted += f"📊 **Details:**\n"
    formatted += f"  🎯 Confidence: {confidence:.1%}\n"
    formatted += f"  ⚡ Response Time: {response_time:.2f}s\n"
    formatted += f"  🤖 Method: {response.get('method', 'Groq AI')}\n"
    
    if 'error' in response:
        formatted += f"  ⚠️ Error: {response['error']}\n"
    
    return formatted


def get_usage_stats() -> Dict[str, Any]:
    """
    Get Groq usage statistics.
    
    Returns:
        Dict[str, Any]: Usage statistics or error message
    """
    global _groq_generator
    
    if _groq_generator is None:
        return {"error": "Groq generator not initialized"}
    
    return _groq_generator.get_stats()


def test_groq_answering():
    """Test the Groq answering system."""
    test_chunks = [
        {
            'text': 'A graph is a data structure consisting of vertices (nodes) connected by edges. It represents relationships between entities.',
            'source': 'graph_tutorial.txt',
            'similarity_score': 0.85
        },
        {
            'text': 'The two main ways to represent graphs are adjacency matrix and adjacency list.',
            'source': 'graph_representations.txt',
            'similarity_score': 0.75
        }
    ]
    
    test_question = "What is a graph data structure?"
    
    logger.info("Testing Groq Answer Generation")
    
    response = generate_groq_answer(test_question, test_chunks)
    formatted = format_groq_response(response)
    
    print(formatted)
    logger.info("Groq test completed successfully")


if __name__ == "__main__":
    test_groq_answering()