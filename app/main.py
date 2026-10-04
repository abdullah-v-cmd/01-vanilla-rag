from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

@dataclass
class Chunk:
    id: int
    text: str
    source: str
    start: int
    end: int

class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...

class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)

class LLMClient(Protocol):
    def generate(self, question: str, contexts: list[Chunk]) -> str: ...

class OpenAICompatibleLLM:
    def __init__(self, base_url: str, api_key: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
    def generate(self, question: str, contexts: list[Chunk]) -> str:
        context = "\n\n".join(f"[{c.id}] {c.text}" for c in contexts)
        prompt = ("Answer only from the supplied context. If the context does not contain "
                  "the answer, say that the evidence is insufficient. Cite supporting chunk "
                  "numbers like [1].\n\nContext:\n" + context)
        response = self.client.chat.completions.create(
            model=self.model, temperature=0,
            messages=[
                {"role": "system", "content": "You are a grounded retrieval assistant."},
                {"role": "user", "content": f"{prompt}\n\nQuestion: {question}"},
            ],
        )
        return response.choices[0].message.content or "No answer generated."

class RAGStore:
    def __init__(self, embedder: Embedder, chunk_size: int = 900, overlap: int = 120):
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: list[Chunk] = []
        self.vectors = np.empty((0, 0), dtype=np.float32)
    @staticmethod
    def chunk_text(text: str, source: str, size: int, overlap: int) -> list[Chunk]:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return []
        result: list[Chunk] = []
        start = 0
        index = 1
        while start < len(cleaned):
            end = min(start + size, len(cleaned))
            if end < len(cleaned):
                boundary = cleaned.rfind(" ", start, end)
                if boundary > start + size // 2:
                    end = boundary
            result.append(Chunk(index, cleaned[start:end], source, start, end))
            index += 1
            if end >= len(cleaned):
                break
            start = max(end - overlap, start + 1)
        return result
    def add_document(self, text: str, source: str) -> int:
        new_chunks = self.chunk_text(text, source, self.chunk_size, self.overlap)
        if not new_chunks:
            return 0
        offset = len(self.chunks)
        for c in new_chunks:
            c.id += offset
        vectors = self.embedder.encode([c.text for c in new_chunks])
        if vectors.ndim != 2 or vectors.shape[0] != len(new_chunks):
            raise ValueError("Embedder returned an invalid vector matrix")
        self.chunks.extend(new_chunks)
        self.vectors = vectors if not len(self.vectors) else np.vstack([self.vectors, vectors])
        return len(new_chunks)
    def search(self, query: str, top_k: int = 4) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        q = self.embedder.encode([query])[0]
        scores = self.vectors @ q
        indices = np.argsort(scores)[::-1][: min(top_k, len(self.chunks))]
        return [(self.chunks[i], float(scores[i])) for i in indices]

class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)

class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
app = FastAPI(title="Vanilla RAG API", version="1.0.0")
_embedder: Embedder | None = None
_store: RAGStore | None = None
_llm: LLMClient | None = None

def get_store() -> RAGStore:
    global _embedder, _store
    if _store is None:
        _embedder = SentenceTransformerEmbedder(MODEL_NAME)
        _store = RAGStore(_embedder)
    return _store

def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        api_key = os.getenv("LLM_API_KEY", "")
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        if not api_key:
            raise HTTPException(503, "LLM is not configured. Set LLM_API_KEY in the environment.")
        _llm = OpenAICompatibleLLM(base_url, api_key, model)
    return _llm

@app.get("/health")
def health():
    return {"status": "ok", "documents_indexed": len(_store.chunks) if _store else 0}

@app.post("/api/documents")
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "Filename is required")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".txt", ".md"}:
        raise HTTPException(400, "Only .txt and .md files are supported in Project 01")
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(400, "File must be UTF-8 encoded") from exc
    store = get_store()
    count = store.add_document(text, file.filename)
    if count == 0:
        raise HTTPException(400, "Document is empty")
    return {"source": file.filename, "chunks_added": count, "total_chunks": len(store.chunks)}

@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest):
    store = get_store()
    results = store.search(request.question, request.top_k)
    if not results:
        raise HTTPException(404, "No documents have been indexed")
    contexts = [item[0] for item in results]
    answer = get_llm().generate(request.question, contexts)
    citations = [{"chunk_id": c.id, "source": c.source, "score": round(score, 4), "text": c.text} for c, score in results]
    return QueryResponse(answer=answer, citations=citations)

@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(Path(__file__).parent.parent / "frontend" / "index.html")
