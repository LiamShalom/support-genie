# 🧞 SupportGenie

An AI-powered support assistant that uses RAG (Retrieval-Augmented Generation) to answer questions from a knowledge base and create support tickets.

## Features

- 💬 **Chat Interface** - Natural language Q&A with chat history
- 📚 **Knowledge Base Search** - Semantic search using sentence transformers and FAISS
- 🎫 **Smart Ticket Creation** - AI-generated ticket titles with severity classification
- 🔍 **Source Citations** - Shows which KB articles were used for answers

## Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd support-genie
```

### 2. Create and activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o  # Optional, defaults to gpt-4o
```

### 5. Run the application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

```
support-genie/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create this)
├── data/
│   └── kb_seed/
│       └── support_kb.json    # Knowledge base articles
└── utils/
    └── ticket_tool.py         # Ticket creation utilities
```

## Usage

### Ask Questions
Simply type your question in the chat:
- "How do I reset my password?"
- "What are the password requirements?"

### Create Tickets
Use keywords to trigger ticket creation:
- "Open a ticket for SSO issue with high severity"
- "Report issue: Login not working"

Severity levels: `high`, `medium`, `low`, or `unspecified`

## Tech Stack

- **Streamlit** - Web interface
- **OpenAI GPT-4o** - Language model for answers and summarization
- **Sentence Transformers** - Text embeddings (all-MiniLM-L6-v2)
- **FAISS** - Vector similarity search
- **Python-dotenv** - Environment configuration

## Requirements

- Python 3.8+
- OpenAI API key with GPT-4o access

