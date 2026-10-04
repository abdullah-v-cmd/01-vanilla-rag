import numpy as np
from fastapi.testclient import TestClient

from app.main import RAGStore, app


class FakeEmbedder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            vectors.append([text.lower().count("python"), text.lower().count("rag"), len(text) / 1000])
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-8)


def test_chunking_preserves_source_and_bounds():
    chunks = RAGStore.chunk_text("one two three four five six", "demo.txt", 12, 2)
    assert chunks
    assert all(c.source == "demo.txt" for c in chunks)
    assert all(len(c.text) <= 12 for c in chunks)


def test_vector_search_returns_relevant_chunk():
    store = RAGStore(FakeEmbedder(), chunk_size=100, overlap=10)
    store.add_document("Python is used for machine learning.", "ml.txt")
    store.add_document("RAG combines retrieval and generation.", "rag.txt")
    results = store.search("RAG retrieval", top_k=1)
    assert results[0][0].source == "rag.txt"
    assert results[0][1] > 0


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
