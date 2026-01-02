# RAG-as-a-Service (FastAPI)

A minimal Retrieval-Augmented Generation (RAG) service with two endpoints:
- `POST /ingest` to add knowledge (text)
- `POST /query` to ask questions grounded on ingested knowledge

## Requirements
- Python 3.12+
- An OpenAI API key

## Setup
```bash
cd rag-as-a-service
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

