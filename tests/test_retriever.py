
import numpy as np
import faiss
from src.retriever import hybrid_retrieve, rerank

from src.retriever import dense_retrieve
from src.retriever import tokenize, bm25_retrieve, build_bm25


def sample_chunks():
    return [
        {
            "source": "attention.pdf",
            "page": 1,
            "chunk_id": 0,
            "text": "Self attention computes relationships between tokens."
        },
        {
            "source": "lora.pdf",
            "page": 2,
            "chunk_id": 1,
            "text": "LoRA reduces trainable parameters using low rank matrices."
        },
        {
            "source": "gan.pdf",
            "page": 3,
            "chunk_id": 2,
            "text": "A generator creates synthetic samples for a discriminator."
        },
    ]


def test_tokenize_lowercases_text():
    tokens = tokenize("Self Attention")
    assert tokens == ["self", "attention"]


def test_tokenize_preserves_hyphenated_terms():
    tokens = tokenize("low-rank matrices")
    assert tokens == ["low-rank", "matrices"]


def test_build_bm25():
    chunks = sample_chunks()
    bm25 = build_bm25(chunks)

    assert bm25 is not None


def test_bm25_retrieves_relevant_chunk():
    chunks = sample_chunks()
    bm25 = build_bm25(chunks)

    results = bm25_retrieve(
        query="low rank matrices",
        bm25=bm25,
        chunks=chunks,
        k=1,
    )

    assert len(results) == 1
    assert results[0]["source"] == "lora.pdf"
    assert "bm25_score" in results[0]


def test_bm25_results_preserve_metadata():
    chunks = sample_chunks()
    bm25 = build_bm25(chunks)

    results = bm25_retrieve(
        query="synthetic samples",
        bm25=bm25,
        chunks=chunks,
        k=1,
    )

    assert results[0]["page"] == 3
    assert results[0]["chunk_id"] == 2




class MockEmbeddingModel:
    """A small deterministic embedding model for unit tests."""

    def encode(self, texts, convert_to_numpy=True):
        embeddings = []

        for text in texts:
            text = text.lower()

            if "attention" in text:
                embeddings.append([1.0, 0.0, 0.0])
            elif "lora" in text or "rank" in text:
                embeddings.append([0.0, 1.0, 0.0])
            else:
                embeddings.append([0.0, 0.0, 1.0])

        return np.array(embeddings, dtype="float32")


def test_dense_retrieves_relevant_chunk():
    chunks = sample_chunks()
    mock_model = MockEmbeddingModel()

    embeddings = mock_model.encode(
        [chunk["text"] for chunk in chunks]
    )
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    results = dense_retrieve(
        query="attention",
        model=mock_model,
        index=index,
        chunks=chunks,
        k=1,
    )

    assert len(results) == 1
    assert results[0]["source"] == "attention.pdf"
    assert "dense_score" in results[0]


def test_dense_retrieval_preserves_metadata():
    chunks = sample_chunks()
    mock_model = MockEmbeddingModel()

    embeddings = mock_model.encode(
        [chunk["text"] for chunk in chunks]
    )
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    results = dense_retrieve(
        query="attention",
        model=mock_model,
        index=index,
        chunks=chunks,
        k=1,
    )

    assert results[0]["page"] == 1
    assert results[0]["chunk_id"] == 0


def test_hybrid_retrieval_returns_ranked_chunks():
    chunks = sample_chunks()
    mock_model = MockEmbeddingModel()

    embeddings = mock_model.encode(
        [chunk["text"] for chunk in chunks]
    )
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    bm25 = build_bm25(chunks)

    results = hybrid_retrieve(
        query="attention",
        model=mock_model,
        index=index,
        bm25=bm25,
        chunks=chunks,
        dense_k=3,
        bm25_k=3,
        final_k=3,
    )

    assert len(results) == 3
    assert results[0]["source"] == "attention.pdf"
    assert all("rrf_score" in result for result in results)


def test_rerank_sorts_candidates_by_score():
    candidates = [
        {
            "source": "attention.pdf",
            "page": 1,
            "chunk_id": 0,
            "text": "Attention is a mechanism for sequence modeling.",
        },
        {
            "source": "lora.pdf",
            "page": 2,
            "chunk_id": 1,
            "text": "LoRA uses low-rank matrices.",
        },
        {
            "source": "gan.pdf",
            "page": 3,
            "chunk_id": 2,
            "text": "A generator creates synthetic samples.",
        },
    ]

    class MockReranker:
        def predict(self, pairs):
            # Higher score means a better match in this test.
            return np.array([0.2, 0.9, -0.1])

    results = rerank(
        query="How does LoRA work?",
        candidates=candidates,
        reranker=MockReranker(),
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["source"] == "lora.pdf"
    assert results[0]["rerank_score"] > results[1]["rerank_score"]