"""
PDF Document Ingestion Module
============================

This module handles processing PDF files and extracting readable text.
It extracts text content and splits it into manageable chunks for RAG.

Functions:
- extract_text_from_pdf(pdf_path): Extract text from PDF file
- process_pdf_content(text, filename): Process and chunk PDF text
- save_pdf_chunks_to_files(chunks, filename): Save chunks as text files
- process_pdf_documents(pdf_files): Main function to process multiple PDFs
"""

import os
import re
import logging
from datetime import datetime
from typing import List, Dict, Optional, Union
import tempfile

# PDF processing libraries — import all available, prefer PyMuPDF > pdfplumber > PyPDF2
_pdf_libs: dict = {}
try:
    import PyPDF2
    _pdf_libs['pypdf2'] = True
except ImportError:
    _pdf_libs['pypdf2'] = False

try:
    import pdfplumber
    _pdf_libs['pdfplumber'] = True
except ImportError:
    _pdf_libs['pdfplumber'] = False

try:
    import fitz  # PyMuPDF
    _pdf_libs['pymupdf'] = True
except ImportError:
    _pdf_libs['pymupdf'] = False

if _pdf_libs.get('pymupdf'):
    PDF_LIBRARY = 'PyMuPDF'
elif _pdf_libs.get('pdfplumber'):
    PDF_LIBRARY = 'pdfplumber'
elif _pdf_libs.get('pypdf2'):
    PDF_LIBRARY = 'PyPDF2'
else:
    PDF_LIBRARY = None

# Import configuration
from config import (
    DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, 
    LOG_FORMAT, LOG_LEVEL
)

# Module logger — basicConfig is handled by the entry point (streamlit_app / main.py)
logger = logging.getLogger(__name__)

# OCR libraries for scanned PDFs
OCR_AVAILABLE = False
try:
    import platform
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_path, convert_from_bytes
    OCR_AVAILABLE = True

    # Tesseract path: use TESSERACT_PATH env var, then well-known defaults
    _tesseract_env = os.getenv('TESSERACT_PATH')
    if _tesseract_env:
        pytesseract.pytesseract.tesseract_cmd = _tesseract_env
        logger.info(f"Using Tesseract from env var: {_tesseract_env}")
    elif platform.system() == 'Windows':
        _win_default = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if os.path.exists(_win_default):
            pytesseract.pytesseract.tesseract_cmd = _win_default
            logger.info(f"Using default Windows Tesseract: {_win_default}")
        else:
            logger.warning("Tesseract not found at default Windows location")
        # On macOS/Linux Tesseract is usually on PATH — no override needed

except ImportError:
    logger.warning(
        "OCR libraries not available. "
        "Install 'pytesseract Pillow pdf2image' for scanned PDF support."
    )


def extract_text_from_pdf(pdf_path: str, method: str = "auto") -> str:
    """
    Extract text from PDF file using available PDF library.
    Falls back to OCR for scanned/image-based PDFs.
    
    Args:
        pdf_path (str): Path to PDF file
        method (str): PDF extraction method ('auto', 'pypdf2', 'pdfplumber', 'pymupdf', 'ocr')
        
    Returns:
        str: Extracted text content
    """
    if PDF_LIBRARY is None and not OCR_AVAILABLE:
        raise ImportError("No PDF processing library available. Install PyPDF2, pdfplumber, PyMuPDF, or OCR libraries")
    
    logger.info(f"Extracting text from PDF: {os.path.basename(pdf_path)}")
    
    text_content = ""
    
    try:
        # First try standard PDF text extraction
        if method != "ocr" and PDF_LIBRARY is not None:
            if method == "auto":
                method = PDF_LIBRARY.lower()
            
            if method == "pypdf2" or (method == "auto" and PDF_LIBRARY == "PyPDF2"):
                text_content = _extract_with_pypdf2(pdf_path)
            elif method == "pdfplumber" or (method == "auto" and PDF_LIBRARY == "pdfplumber"):
                text_content = _extract_with_pdfplumber(pdf_path)
            elif method == "pymupdf" or (method == "auto" and PDF_LIBRARY == "PyMuPDF"):
                text_content = _extract_with_pymupdf(pdf_path)
        
        # Automatic OCR fallback for image-based PDFs
        if (not text_content or len(text_content.strip()) < 50) and OCR_AVAILABLE:
            logger.info(f"Standard extraction yielded {len(text_content.strip()) if text_content else 0} chars - trying OCR...")
            try:
                ocr_text = _extract_with_ocr(pdf_path)
                if len(ocr_text.strip()) > len(text_content.strip() if text_content else ''):
                    text_content = ocr_text
                    logger.info(f"✅ OCR successful: {len(text_content)} characters extracted")
                else:
                    logger.warning(f"OCR yielded {len(ocr_text)} chars vs standard {len(text_content) if text_content else 0} chars")
                    # Use OCR text anyway if standard extraction was completely empty
                    if not text_content or len(text_content.strip()) == 0:
                        text_content = ocr_text
                        logger.info("Using OCR text as standard extraction was empty")
            except Exception as ocr_error:
                logger.error(f"❌ OCR processing failed: {str(ocr_error)}")
                logger.error("Check: 1) Tesseract installed? 2) Poppler in PATH? 3) Run debug_ocr.py")
        elif (not text_content or len(text_content.strip()) < 50) and not OCR_AVAILABLE:
            logger.error("📄 Image-based PDF detected but OCR dependencies missing!")
            logger.error("Install: pip install pytesseract Pillow pdf2image")
            logger.error("And Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
            logger.error("And Poppler: https://github.com/oschwartz10612/poppler-windows/releases")
            
        logger.info(f"Extracted {len(text_content)} characters from PDF")
        return text_content
        
    except Exception as e:
        logger.error(f"Failed to extract text from PDF {pdf_path}: {str(e)}")
        
        # Last resort: try OCR if available
        if OCR_AVAILABLE:
            try:
                logger.info("Attempting OCR as last resort...")
                return _extract_with_ocr(pdf_path)
            except Exception as ocr_e:
                logger.error(f"OCR also failed: {str(ocr_e)}")
        
        return ""


def _extract_with_pypdf2(pdf_path: str) -> str:
    """Extract text using PyPDF2."""
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            except Exception as e:
                logger.warning(f"Error extracting page {page_num + 1}: {e}")
                continue
    return text


def _extract_with_pdfplumber(pdf_path: str) -> str:
    """Extract text using pdfplumber."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            except Exception as e:
                logger.warning(f"Error extracting page {page_num + 1}: {e}")
                continue
    return text


def _extract_with_pymupdf(pdf_path: str) -> str:
    """Extract text using PyMuPDF (fitz)."""
    text = ""
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        try:
            page = doc.load_page(page_num)
            page_text = page.get_text()
            if page_text.strip():
                text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        except Exception as e:
            logger.warning(f"Error extracting page {page_num + 1}: {e}")
            continue
    doc.close()
    return text


def _extract_with_ocr(pdf_path: str) -> str:
    """Extract text using OCR for scanned/image-based PDFs."""
    if not OCR_AVAILABLE:
        raise ImportError("OCR libraries not available. Install 'pytesseract Pillow pdf2image'")
    
    logger.info(f"🔍 Starting OCR extraction for: {os.path.basename(pdf_path)}")
    text = ""
    
    try:
        # Convert PDF pages to images with error handling
        logger.info("Converting PDF pages to images...")
        try:
            if os.path.isfile(pdf_path):
                images = convert_from_path(pdf_path, dpi=300)  # Higher DPI for better OCR accuracy
            else:
                # If it's bytes data
                with open(pdf_path, 'rb') as file:
                    images = convert_from_bytes(file.read(), dpi=300)
        except Exception as conversion_error:
            logger.error(f"PDF→Image conversion failed: {conversion_error}")
            # Try lower DPI as fallback
            logger.info("Retrying with lower DPI (150)...")
            if os.path.isfile(pdf_path):
                images = convert_from_path(pdf_path, dpi=150)
            else:
                with open(pdf_path, 'rb') as file:
                    images = convert_from_bytes(file.read(), dpi=150)
        
        if not images:
            raise ValueError("No images generated from PDF - file may be corrupted")
            
        logger.info(f"Processing {len(images)} pages with OCR...")
        
        # OCR each page with multiple fallback strategies
        successful_pages = 0
        for page_num, image in enumerate(images):
            try:
                # Primary OCR attempt with optimal settings
                custom_config = r'--oem 3 --psm 6 -l eng'  # LSTM engine, document mode
                page_text = pytesseract.image_to_string(image, config=custom_config)
                
                # If primary OCR yields little text, try alternative settings
                if len(page_text.strip()) < 10:
                    logger.info(f"Page {page_num + 1}: trying alternative OCR settings...")
                    alt_config = r'--oem 3 --psm 3 -l eng'  # Different page segmentation
                    alt_text = pytesseract.image_to_string(image, config=alt_config)
                    if len(alt_text.strip()) > len(page_text.strip()):
                        page_text = alt_text
                
                if page_text.strip():
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    successful_pages += 1
                else:
                    logger.warning(f"Page {page_num + 1}: OCR returned no text")
                    
            except Exception as e:
                logger.warning(f"OCR failed for page {page_num + 1}: {e}")
                continue
        
        logger.info(f"✅ OCR completed: {len(text)} chars from {successful_pages}/{len(images)} pages")
        return text
        
    except Exception as e:
        logger.error(f"OCR extraction failed: {str(e)}")
        return ""


def clean_pdf_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving page markers for metadata tracking.
    """
    if not text:
        return ""

    # Fix broken words across lines (hyphenation)
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # Normalize whitespace within lines but keep page markers
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if line.startswith('--- Page'):
            cleaned_lines.append(line)  # preserve page markers
        else:
            cleaned_lines.append(re.sub(r'[ \t]+', ' ', line).strip())

    text = '\n'.join(cleaned_lines)

    # Collapse 3+ blank lines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()



def _recursive_character_text_splitter(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Recursively split text using a hierarchy of separators to preserve
    paragraph boundaries (mirrors LangChain's RecursiveCharacterTextSplitter).

    Separator hierarchy: paragraph > line > sentence > word
    """
    separators = ['\n\n', '\n', '. ', ' ', '']

    def _split(text: str, separators: List[str]) -> List[str]:
        if not text:
            return []
        if len(text) <= chunk_size:
            return [text]

        sep = separators[0]
        next_seps = separators[1:]

        if sep == '' or sep not in text:
            if next_seps:
                return _split(text, next_seps)
            # hard cut at chunk_size
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]

        parts = text.split(sep)
        chunks = []
        current = ''

        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # part itself too large → recurse
                if len(part) > chunk_size and next_seps:
                    chunks.extend(_split(part, next_seps))
                    current = ''
                else:
                    current = part

        if current:
            chunks.append(current)

        return chunks

    raw_chunks = _split(text, separators)

    # Apply overlap: prepend tail of previous chunk
    if chunk_overlap <= 0 or len(raw_chunks) <= 1:
        return [c for c in raw_chunks if c.strip()]

    overlapped = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        overlap_text = overlapped[-1][-chunk_overlap:]
        overlapped.append((overlap_text + ' ' + raw_chunks[i]).strip())

    return [c for c in overlapped if c.strip()]


def _detect_section_title(text: str) -> str:
    """
    Try to detect a section title from the beginning of a text chunk.
    Returns empty string if none found.
    """
    lines = text.strip().split('\n')
    for line in lines[:3]:  # check first 3 lines
        line = line.strip()
        if not line:
            continue
        # Heuristic: all-caps, or short line ending without punctuation
        if (line.isupper() and len(line) > 3) or \
           (len(line) < 80 and not line.endswith(('.', ',', ';', ':')) and
                re.match(r'^[\d\.]*\s*[A-Z]', line)):
            return line[:100]
    return ''


def split_pdf_into_chunks(text: str, filename: str = '',
                          chunk_size: int = None, chunk_overlap: int = None) -> List[Dict]:
    """
    Split PDF text into chunks with rich metadata (page number, section title).

    Returns a list of dicts::
        {
            'text': str,
            'page': int | None,
            'section': str,
            'source': str,
        }
    """
    from config import CHUNK_SIZE, CHUNK_OVERLAP
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP

    if not text:
        return []

    logger.info(f"Chunking PDF text: {len(text)} characters (chunk_size={chunk_size}, overlap={chunk_overlap})")

    # --- parse page-segmented text ---
    # Expected format from _extract_* functions: "\n--- Page N ---\n..."
    page_pattern = re.compile(r'---\s*Page\s*(\d+)\s*---')
    page_segments = []  # list of (page_num, page_text)

    parts = page_pattern.split(text)  # [pre, pg1, text1, pg2, text2, ...]
    if len(parts) > 1:
        # parts[0] is text before first page marker (usually empty)
        i = 1
        while i < len(parts) - 1:
            page_num = int(parts[i])
            page_text = parts[i + 1]
            page_segments.append((page_num, page_text.strip()))
            i += 2
    else:
        # No page markers found — treat entire text as page 1
        page_segments = [(1, text)]

    # --- chunk each page segment and tag with metadata ---
    all_chunks = []
    for page_num, page_text in page_segments:
        if not page_text.strip():
            continue
        sub_chunks = _recursive_character_text_splitter(page_text, chunk_size, chunk_overlap)
        for chunk_text in sub_chunks:
            chunk_text = chunk_text.strip()
            # More lenient threshold for short documents (MCQs, receipts, etc.)
            if len(chunk_text) < 15:  # Skip only truly empty fragments
                continue
            # Also skip chunks that are just page markers or pure whitespace/numbers
            if not any(c.isalpha() for c in chunk_text):  # No alphabetic characters
                continue
            all_chunks.append({
                'text': chunk_text,
                'page': page_num,
                'section': _detect_section_title(chunk_text),
                'source': filename,
            })

    # If we got very few chunks from a document with multiple pages,
    # it might be a low-quality OCR result - combine pages for better context
    if len(all_chunks) == 0 and len(page_segments) > 1:
        logger.warning(f"No chunks created from {len(page_segments)} pages - combining for low-quality OCR")
        combined_text = " ".join([f"Page {num}: {text}" for num, text in page_segments if text.strip()])
        if len(combined_text.strip()) > 15:
            all_chunks.append({
                'text': combined_text,
                'page': 1,  # Combined pages
                'section': f"Combined pages 1-{len(page_segments)}",
                'source': filename,
            })
            logger.info(f"Created 1 combined chunk with {len(combined_text)} characters")

    logger.info(f"Created {len(all_chunks)} chunks from PDF")
    return all_chunks



def save_pdf_chunks_to_files(chunks: List[Dict], filename: str, data_dir: str = None) -> List[str]:
    """
    Save PDF chunks (dicts with text + metadata) as individual text files.

    Args:
        chunks (List[Dict]): Chunk dicts produced by split_pdf_into_chunks
        filename (str): Original PDF filename
        data_dir (str): Directory to save files

    Returns:
        List[str]: List of saved file paths
    """
    if data_dir is None:
        data_dir = DATA_DIR

    os.makedirs(data_dir, exist_ok=True)

    base_name = os.path.splitext(filename)[0]
    safe_name = re.sub(r'[^\w\-_]', '_', base_name)

    saved_files = []
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        chunk_text = chunk.get('text', '') if isinstance(chunk, dict) else chunk
        page_num   = chunk.get('page', '') if isinstance(chunk, dict) else ''
        section    = chunk.get('section', '') if isinstance(chunk, dict) else ''

        if not chunk_text.strip():
            continue

        chunk_filename = f"{safe_name}_chunk{i+1:03d}.txt"
        filepath = os.path.join(data_dir, chunk_filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Source: {filename}\n")
                f.write(f"Chunk {i+1}/{total}\n")
                f.write(f"Page: {page_num}\n")
                f.write(f"Section: {section}\n")
                f.write(f"Processed: {datetime.now().isoformat()}\n")
                f.write("-" * 50 + "\n\n")
                f.write(chunk_text)

            saved_files.append(filepath)

        except Exception as e:
            logger.error(f"Error saving chunk {i+1}: {e}")

    logger.info(f"Saved {len(saved_files)} chunks from {filename}")
    return saved_files



def process_pdf_documents(pdf_files: List[Union[str, tuple]]) -> Dict[str, List[str]]:
    """
    Main function to process multiple PDF files.
    
    Args:
        pdf_files: List of PDF file paths or (filename, content) tuples
        
    Returns:
        Dict[str, List[str]]: Mapping of filenames to chunk file paths
    """
    logger.info(f"Processing {len(pdf_files)} PDF documents")
    
    results = {}
    
    for pdf_input in pdf_files:
        try:
            # Handle different input types
            if isinstance(pdf_input, tuple):
                filename, content = pdf_input
                # Save uploaded content to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(content)
                    pdf_path = tmp_file.name
            else:
                filename = os.path.basename(pdf_input)
                pdf_path = pdf_input
            
            logger.info(f"Processing: {filename}")
            
            # Extract text (includes automatic OCR fallback)
            text = extract_text_from_pdf(pdf_path)
            
            # More lenient threshold - even OCR from poor quality images can be useful
            if not text or len(text.strip()) < 20:
                logger.error(f"Minimal text extracted from {filename}: got {len(text) if text else 0} characters")
                logger.error(f"OCR_AVAILABLE: {OCR_AVAILABLE}, PDF_LIBRARY: {PDF_LIBRARY}")
                
                # Try forcing OCR if not already attempted
                if OCR_AVAILABLE and len(text.strip() if text else '') < 10:
                    logger.info(f"Attempting forced OCR extraction for {filename}...")
                    try:
                        ocr_text = extract_text_from_pdf(pdf_path, method="ocr")
                        if len(ocr_text.strip()) > len(text.strip() if text else ''):
                            text = ocr_text
                            logger.info(f"Forced OCR recovered {len(text)} characters")
                    except Exception as ocr_error:
                        logger.error(f"Forced OCR also failed for {filename}: {ocr_error}")
                
                # Final check - if still minimal, skip this PDF
                if not text or len(text.strip()) < 20:
                    logger.warning(f"Skipping {filename} - insufficient extractable content")
                    continue
                else:
                    logger.info(f"Proceeding with {filename} - extracted {len(text)} characters")
            
            # Clean text (preserves page markers)
            cleaned_text = clean_pdf_text(text)
            
            # Create chunks (returns list of dicts with metadata)
            chunks = split_pdf_into_chunks(cleaned_text, filename)
            if not chunks:
                logger.warning(f"No chunks created from {filename}")
                continue
            
            # Save chunks (accepts list of dicts)
            saved_files = save_pdf_chunks_to_files(chunks, filename)
            results[filename] = saved_files
            
            # Cleanup temporary file if created
            if isinstance(pdf_input, tuple):
                try:
                    os.unlink(pdf_path)
                except OSError as unlink_err:
                    logger.warning(f"Could not delete temp file {pdf_path}: {unlink_err}")
            
            logger.info(f"Successfully processed {filename}: {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_input}: {str(e)}")
            continue
    
    logger.info(f"PDF processing complete: {len(results)} files processed successfully")
    return results


def get_pdf_info(pdf_path: str) -> Dict[str, any]:
    """Get basic information about a PDF file."""
    try:
        if PDF_LIBRARY == "PyPDF2":
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                return {
                    "pages": len(reader.pages),
                    "title": reader.metadata.get('/Title', 'Unknown') if reader.metadata else 'Unknown',
                    "author": reader.metadata.get('/Author', 'Unknown') if reader.metadata else 'Unknown'
                }
        elif PDF_LIBRARY == "PyMuPDF":
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            info = {
                "pages": len(doc),
                "title": metadata.get('title', 'Unknown'),
                "author": metadata.get('author', 'Unknown')
            }
            doc.close()
            return info
    except Exception as e:
        logger.warning(f"Could not get PDF info: {e}")
    
    return {"pages": "Unknown", "title": "Unknown", "author": "Unknown"}


def get_all_chunks() -> List[Dict[str, str]]:
    """
    Retrieves all text chunks from the data directory, including metadata.

    Returns:
        List[Dict]: Each dict has 'text', 'source', 'page', 'section', 'file' keys.
    """
    chunks = []

    if not os.path.exists(DATA_DIR):
        logger.error(f"Data directory '{DATA_DIR}' not found")
        return chunks

    for filename in sorted(os.listdir(DATA_DIR)):
        if not filename.endswith('.txt'):
            continue
        filepath = os.path.join(DATA_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')
            meta = {}
            text_start = 0
            for idx, line in enumerate(lines):
                if line.startswith('Source: '):
                    meta['source'] = line[8:].strip()
                elif line.startswith('Page: '):
                    raw = line[6:].strip()
                    meta['page'] = int(raw) if raw.isdigit() else raw
                elif line.startswith('Section: '):
                    meta['section'] = line[9:].strip()
                elif line.startswith('-' * 10):
                    text_start = idx + 2  # skip separator + blank line
                    break

            text_content = '\n'.join(lines[text_start:]).strip()
            if text_content:
                chunks.append({
                    'text': text_content,
                    'source': meta.get('source', 'Unknown'),
                    'page': meta.get('page', ''),
                    'section': meta.get('section', ''),
                    'file': filename,
                })
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")

    logger.info(f"Retrieved {len(chunks)} chunks from data directory")
    return chunks



# Example usage (this runs only when file is executed directly)
if __name__ == "__main__":
    # Example PDF processing
    logger.info("PDF ingestion module loaded successfully")
    if PDF_LIBRARY:
        logger.info(f"Using PDF library: {PDF_LIBRARY}")
    else:
        logger.warning("No PDF processing library available. Install PyPDF2, pdfplumber, or PyMuPDF")