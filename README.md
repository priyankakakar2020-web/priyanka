# JM Mutual Funds FAQ Chatbot

A Retrieval-Augmented Generation (RAG) based chatbot that provides instant answers about JM Mutual Funds using real scraped data from Groww.in.

## Features

- 🤖 **AI-Powered RAG System** - Uses FAISS vector store and sentence transformers for semantic search
- 📊 **Real Data** - Scraped from official Groww.in pages (no fake/demo data)
- 🎨 **Modern UI** - Dark-themed chat interface with robot mascot
- 🔗 **Source Citations** - Every answer includes the original source URL
- ⚡ **Fast Responses** - Local hosting with Flask backend

## Supported Funds

1. **JM Value Fund Direct Plan Growth**
2. **JM Aggressive Hybrid Fund Direct Growth**
3. **JM Flexicap Fund Direct Plan Growth**
##**source:**
1. Groww
2. Sebi
## Tech Stack

- **Backend**: Flask, Python 3
- **RAG**: FAISS, Sentence Transformers
- **Scraping**: BeautifulSoup4, Requests
- **Frontend**: HTML, CSS, JavaScript
- **AI**: Google Gemini (optional)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/jm-mutual-funds-chatbot.git
cd jm-mutual-funds-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the Chatbot

```bash
py -3 app.py
```

Then open your browser at: **http://localhost:5000**

### Scrape New Data

```bash
py -3 scripts/scrape_groww_jm_value_fund.py
py -3 scripts/scrape_groww_jm_aggressive_hybrid.py
py -3 scripts/scrape_groww_jm_flexicap.py
```

### Rebuild Vector Store

```bash
py -3 scripts/build_vector_store.py
```

### Run Tests

```bash
py -3 scripts/test_rag.py
```

### Command-Line Query (Optional)

```bash
py -3 scripts/rag_query.py --question "What is the expense ratio of JM Value Fund?"
```

## Project Structure

```
priyanka/
├── app.py                          # Flask web server
├── requirements.txt                # Python dependencies
├── static/
│   └── index.html                  # Frontend UI
├── scripts/
│   ├── scrape_groww_jm_value_fund.py
│   ├── scrape_groww_jm_aggressive_hybrid.py
│   ├── scrape_groww_jm_flexicap.py
│   ├── build_vector_store.py      # Build FAISS index
│   ├── rag_query.py                # Basic RAG query
│   ├── rag_query_gemini.py         # Gemini-powered RAG
│   └── test_rag.py                 # Unit tests
├── data/
│   ├── schemes/                    # Scraped fund data (JSON)
│   └── guides/                     # Guide documents (JSON)
└── vector_store/
    ├── faiss.index                 # FAISS vector index
    └── documents.json              # Document metadata
```

## Example Questions

- What is the expense ratio of JM Value Fund?
- What is the exit load for JM Aggressive Hybrid Fund?
- What is the minimum SIP investment for JM Flexicap Fund?
- How to download mutual fund statement from Groww?

## API Endpoints

### POST /api/query
Query the chatbot

**Request:**
```json
{
  "question": "What is the expense ratio of JM Value Fund?"
}
```

**Response:**
```json
{
  "success": true,
  "question": "What is the expense ratio of JM Value Fund?",
  "answer": "JM Value Fund Direct Plan Growth - Expense Ratio: 0.98%...",
  "source": "https://groww.in/mutual-funds/jm-basic-fund-direct-growth"
}
```

### GET /api/health
Health check endpoint

## License

MIT
