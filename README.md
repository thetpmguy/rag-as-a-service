

# RAG-as-a-Service (FastAPI)

A minimal, working Retrieval-Augmented Generation (RAG) service exposed as an API.

This service allows you to:
- Ingest text documents
- Ask questions grounded only in the ingested data
- Receive retrieval-backed answers with supporting context

This project is intentionally simple, demo-friendly, and easy to run locally.

---

| Component           | Responsibility                              | What This Service Uses                                                                                                             |
| ------------------- | ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Chunking**        | Preserve meaning and improve retrievability | Deterministic text chunking implemented in Python, splitting input documents into small, overlapping text segments before indexing |
| **Embeddings**      | Represent semantic meaning numerically      | OpenAI embedding model to convert each text chunk and user query into high-dimensional vectors                                     |
| **Vector Search**   | Retrieve the most relevant evidence         | In-memory vector similarity search using cosine similarity over stored embedding vectors                                           |
| **LLM (Reasoning)** | Generate grounded, human-readable answers   | OpenAI chat completion model, constrained to answer strictly from retrieved context                                                |


## What this project demonstrates

- Document ingestion and chunking
- Vector embeddings and similarity search
- Retrieval-grounded LLM responses
- RAG exposed as a reusable backend service using FastAPI

---

## Requirements

- Python 3.12+
- OpenAI API key

---

## Setup

Clone the repository:
bash
git clone https://github.com/YOUR_GITHUB_USERNAME/rag-as-a-service.git
cd rag-as-a-service

## Create and activate a virtual environment:
python3.12 -m venv .venv
source .venv/bin/activate

## Install dependencies:
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

## Set environment variables:
cp .env.example .env

## Edit .env and add:
OPENAI_API_KEY=your_api_key_here

## Run the service
source .venv/bin/activate
python -m uvicorn app:app --reload --port 8000

http://127.0.0.1:8000/docs

## Demo
Ingest data:
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Refunds are allowed within 30 days with a receipt. Exchanges are allowed within 60 days.",
    "source_name": "Returns Policy v1"
  }'

Query the data:
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How long do I have to return an item?",
    "top_k": 5,
    "return_context": true
  }'




