# 🌐 URL RAG System

An **Enterprise Knowledge Assistant** that transforms **web documentation into an intelligent Q&A system** using **Retrieval-Augmented Generation (RAG)**.

This project automatically ingests content from **web URLs**, converts it into embeddings, stores it in a **vector database**, and allows users to **ask natural language questions**.

---

# ✨ Features

* 🔗 **URL Document Ingestion** – Scrape and process documentation directly from web URLs
* 🧠 **Semantic Search** – ChromaDB vector database with transformer embeddings
* 💬 **AI-Powered Answers** – Contextual responses using **Groq Llama models**
* 🎨 **Modern Web Interface** – Clean and simple **Streamlit UI**
* 📊 **Smart Text Chunking** – Optimized document chunking for better retrieval
* ⚡ **Production Ready** – Logging, configuration, and modular architecture

---

# 🏗️ System Architecture

```
URLs
  ↓
Web Scraping
  ↓
Text Processing
  ↓
Embeddings
  ↓
ChromaDB Vector Store
  ↓
Semantic Retrieval
  ↓
Groq LLM
  ↓
Answer Generation
```

---

# ⚙️ Tech Stack

| Component       | Technology           |
| --------------- | -------------------- |
| Web Scraping    | BeautifulSoup        |
| Embeddings      | SentenceTransformers |
| Vector Database | ChromaDB             |
| LLM             | Groq (Llama-3.1-8B)  |
| Backend         | Python               |
| UI              | Streamlit            |

---

# 📁 Project Structure

```
.
├── config.py                 # Configuration settings
├── main.py                   # CLI interface
├── streamlit_app.py          # Streamlit web interface
├── requirements.txt          # Python dependencies
│
├── ingestion/
│   └── ingest_docs.py        # URL scraping and text extraction
│
├── rag/
│   ├── chroma_retrieval.py   # Vector database operations
│   └── groq_answering.py     # LLM response generation
│
├── utils/
│   └── helpers.py            # Utility functions
│
├── data/                     # Processed documents (auto-generated)
└── chroma_db/                # ChromaDB storage (auto-generated)
```

---

# 🚀 Quick Start

## 1️⃣ Prerequisites

* Python **3.8+**
* **Groq API Key**

Create a free account:
👉 https://console.groq.com

---

# 📥 Installation

### Clone the repository

```bash
git clone https://github.com/Satheesh127/rag-url-qa-system.git
cd rag-url-qa-system
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Create environment file

Create `.env` in project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

# ▶️ Usage

## Option 1 — Run Web Interface (Recommended)

```bash
streamlit run streamlit_app.py
```

Open in browser:

```
http://localhost:8501
```

---

## Option 2 — Run CLI

```bash
python main.py
```

---

# ⚙️ Configuration

Key parameters in **`config.py`**

| Setting              | Default              | Description                    |
| -------------------- | -------------------- | ------------------------------ |
| CHUNK_SIZE           | 600                  | Size of text chunks            |
| MAX_CHUNKS_PER_QUERY | 5                    | Retrieved chunks per query     |
| GROQ_MODEL_NAME      | llama-3.1-8b-instant | LLM model                      |
| EMBEDDING_MODEL_NAME | all-MiniLM-L6-v2     | Sentence transformer model     |
| SIMILARITY_THRESHOLD | 0.6                  | Retrieval similarity threshold |

---

# 💡 How It Works

## 1️⃣ Document Ingestion

* Fetches web pages from URLs
* Cleans HTML (removes ads, scripts, navigation)
* Extracts readable content
* Splits text into overlapping chunks

---

## 2️⃣ Vector Database

* Creates embeddings using **SentenceTransformers**
* Stores vectors in **ChromaDB**
* Enables **semantic similarity search**

---

## 3️⃣ Question Answering

* Retrieves relevant document chunks
* Builds context prompt
* Sends prompt to **Groq Llama model**
* Returns contextual answer with references

---

# 🛠️ API Configuration

### Groq API

| Setting             | Value                |
| ------------------- | -------------------- |
| Model               | llama-3.1-8b-instant |
| Context Tokens      | 4000                 |
| Max Response Tokens | 1000                 |

Get API key here:

👉 https://console.groq.com

---

# 📦 Dependencies

Core libraries used:

* chromadb
* sentence-transformers
* groq
* streamlit
* beautifulsoup4
* requests

Install with:

```bash
pip install -r requirements.txt
```

---

# 🏃 Usage Example

### Add URLs

Example documentation sources:

```
https://docs.python.org/3/tutorial/
https://fastapi.tiangolo.com/tutorial/
https://docs.streamlit.io/library/get-started
```

### Ask Questions

Examples:

```
How do I create a FastAPI endpoint?
What are Python decorators?
How do I deploy a Streamlit app?
```

---

# 🔍 Troubleshooting

### Missing API Key

Error:

```
GROQ_API_KEY not found
```

Solution:

Add the key to `.env`.

---

### ChromaDB Permission Error

Ensure write permissions for the project directory.

---

### URL Processing Failed

Check:

* Internet connection
* URL accessibility

---

# 📈 Performance Tips

| Optimization | Recommendation                    |
| ------------ | --------------------------------- |
| Chunk Size   | 400-600 for precise queries       |
| Max Chunks   | Increase for complex questions    |
| URL Sources  | Use well-structured documentation |

---

# 🤝 Contributing

Contributions are welcome!

Steps:

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Submit a pull request

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 🙏 Acknowledgments

* **Groq** – Free high-speed LLM API
* **ChromaDB** – Vector database
* **Sentence Transformers** – Embeddings
* **Streamlit** – Web application framework

---

⭐ If you find this project useful, consider **starring the repository**!
