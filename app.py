import os, re, uuid
from typing import List, Dict, Any, Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Missing OPENAI_API_KEY. Add it to .env and restart.")

client = OpenAI()
app = FastAPI(title="RAG-as-a-Service Demo", version="0.2")

INDEX: List[Dict[str, Any]] = []
DOCS: Dict[str, Dict[str, Any]] = {}

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    text = clean_text(text)
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

def embed(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]

def retrieve(question: str, top_k: int) -> List[Dict[str, Any]]:
    if not INDEX:
        return []
    q = np.array(embed([question])[0], dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-12)

    mat = np.array([c["embedding"] for c in INDEX], dtype=np.float32)
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

    sims = mat @ q
    top = np.argsort(-sims)[:top_k]
    return [INDEX[i] for i in top]

def answer_with_context(question: str, ctx: List[Dict[str, Any]]) -> str:
    context_block = "\n\n".join(
        [f"[{i+1}] {c['source']}\n{c['text']}" for i, c in enumerate(ctx)]
    )
    prompt = (
        "You are a retrieval-grounded assistant.\n"
        "Use ONLY the provided context. If the answer isn't in it, say: "
        "\"I don't know based on the provided documents.\"\n"
        "Cite sources like [1], [2].\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:"
    )
    resp = client.responses.create(model="gpt-4.1-mini", input=prompt)
    return resp.output_text.strip()

class IngestRequest(BaseModel):
    text: str
    source_name: Optional[str] = "manual"

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    return_context: bool = True

@app.get("/health")
def health():
    return {"ok": True, "docs": len(DOCS), "chunks": len(INDEX)}

@app.post("/ingest")
def ingest(req: IngestRequest):
    raw = clean_text(req.text)
    if not raw:
        raise HTTPException(status_code=400, detail="Empty text")

    doc_id = str(uuid.uuid4())
    DOCS[doc_id] = {"source": req.source_name, "length": len(raw)}

    chunks = chunk_text(raw)
    embs = embed(chunks)

    for t, e in zip(chunks, embs):
        INDEX.append({
            "doc_id": doc_id,
            "chunk_id": str(uuid.uuid4()),
            "source": req.source_name,
            "text": t,
            "embedding": e,
        })

    return {"doc_id": doc_id, "chunks_indexed": len(chunks)}

@app.post("/query")
def query(req: QueryRequest):
    ctx = retrieve(req.question, req.top_k)
    if not ctx:
        return {"answer": "I don't know based on the provided documents.", "sources": []}

    ans = answer_with_context(req.question, ctx)

    sources = []
    if req.return_context:
        for i, c in enumerate(ctx, start=1):
            sources.append({
                "rank": i,
                "source": c["source"],
                "preview": c["text"][:200] + ("..." if len(c["text"]) > 200 else "")
            })

    return {"answer": ans, "sources": sources}
