
import re
import numpy as np
import faiss

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# --------------------------------------------------
# 1. Tokenization for BM25
# --------------------------------------------------

def tokenize(text):
    """Convert text into lowercase word tokens."""
    text = text.lower()
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)


# --------------------------------------------------
# 2. Dense Retrieval using FAISS
# --------------------------------------------------

def dense_retrieve(query, model, index, chunks, k=5):
    """
    Retrieve chunks using semantic similarity.

    model: SentenceTransformer embedding model
    index: FAISS index containing normalized chunk embeddings
    chunks: List of chunk dictionaries
    """
    if not chunks:
        return []

    query_embedding = model.encode(
        [query], convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(query_embedding)

    k = min(k, index.ntotal)
    if k == 0:
        return []

    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        result = chunks[int(idx)].copy()
        result["dense_score"] = float(score)
        results.append(result)

    return results


# --------------------------------------------------
# 3. BM25 Keyword Retrieval
# --------------------------------------------------

def build_bm25(chunks):
    """Build a BM25 index from document chunks."""
    tokenized_chunks = [
        tokenize(chunk["text"]) for chunk in chunks
    ]
    return BM25Okapi(tokenized_chunks)


def bm25_retrieve(query, bm25, chunks, k=5):
    """Retrieve chunks using keyword relevance."""
    if not chunks:
        return []

    scores = bm25.get_scores(tokenize(query))
    k = min(k, len(chunks))

    top_indices = np.argsort(scores)[::-1][:k]

    results = []

    for idx in top_indices:
        result = chunks[int(idx)].copy()
        result["bm25_score"] = float(scores[idx])
        results.append(result)

    return results


# --------------------------------------------------
# 4. Hybrid Retrieval using Reciprocal Rank Fusion
# --------------------------------------------------

def hybrid_retrieve(
    query,
    model,
    index,
    bm25,
    chunks,
    dense_k=10,
    bm25_k=10,
    final_k=5,
    rrf_constant=60,
):
    """
    Combine dense and keyword retrieval using RRF.

    RRF combines rankings rather than comparing raw scores.
    """
    if not chunks:
        return []

    dense_results = dense_retrieve(
        query, model, index, chunks, k=dense_k
    )

    keyword_results = bm25_retrieve(
        query, bm25, chunks, k=bm25_k
    )

    rrf_scores = {}

    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result["chunk_id"]
        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0.0)
            + 1.0 / (rrf_constant + rank)
        )

    for rank, result in enumerate(keyword_results, start=1):
        chunk_id = result["chunk_id"]
        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0.0)
            + 1.0 / (rrf_constant + rank)
        )

    ranked_chunk_ids = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True,
    )[:final_k]

    chunk_lookup = {
        chunk["chunk_id"]: chunk for chunk in chunks
    }

    results = []

    for chunk_id in ranked_chunk_ids:
        result = chunk_lookup[chunk_id].copy()
        result["rrf_score"] = float(rrf_scores[chunk_id])
        results.append(result)

    return results


# --------------------------------------------------
# 5. Cross-Encoder Reranking
# --------------------------------------------------

def rerank(query, candidates, reranker, top_k=5):
    """
    Rerank retrieved candidates using a cross-encoder.

    Cross-encoder scores are ranking scores, not probabilities.
    """
    if not candidates:
        return []

    pairs = [
        [query, candidate["text"]]
        for candidate in candidates
    ]

    scores = reranker.predict(pairs)

    ranked = []

    for candidate, score in zip(candidates, scores):
        result = candidate.copy()
        result["rerank_score"] = float(score)
        ranked.append(result)

    ranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    return ranked[:top_k]


# --------------------------------------------------
# 6. Complete Retrieval Pipeline
# --------------------------------------------------

def retrieve_relevant_chunks(
    query,
    model,
    index,
    bm25,
    chunks,
    reranker,
    dense_k=10,
    bm25_k=10,
    candidate_k=10,
    top_k=3,
):
    """
    Complete retrieval flow:
    Dense Retrieval + BM25 -> RRF -> Cross-Encoder Reranking.
    """
    candidates = hybrid_retrieve(
        query=query,
        model=model,
        index=index,
        bm25=bm25,
        chunks=chunks,
        dense_k=dense_k,
        bm25_k=bm25_k,
        final_k=candidate_k,
    )

    return rerank(
        query=query,
        candidates=candidates,
        reranker=reranker,
        top_k=top_k,
    )