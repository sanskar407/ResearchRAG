
from src.retriever import retrieve_relevant_chunks
from src.generator import generate_answer


def answer_question(
    query,
    model,
    index,
    bm25,
    chunks,
    reranker,
    top_k=3,
    dense_k=10,
    bm25_k=10,
    candidate_k=10,
    groq_client=None,
):
    """
    Run the complete RAG pipeline.

    Flow:
    1. Retrieve relevant chunks using dense and BM25 retrieval.
    2. Fuse rankings using RRF.
    3. Rerank candidates with a cross-encoder.
    4. Generate an answer grounded in retrieved context.
    5. Return the answer and source metadata.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    # Step 1: Retrieve and rerank relevant chunks
    retrieved_chunks = retrieve_relevant_chunks(
        query=query,
        model=model,
        index=index,
        bm25=bm25,
        chunks=chunks,
        reranker=reranker,
        dense_k=dense_k,
        bm25_k=bm25_k,
        candidate_k=candidate_k,
        top_k=top_k,
    )

    # Step 2: Generate a grounded answer
    answer = generate_answer(
        query=query,
        results=retrieved_chunks,
        client=groq_client,
    )

    # Step 3: Prepare source metadata
    sources = []

    for chunk in retrieved_chunks:
        sources.append(
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "rerank_score": chunk.get("rerank_score"),
            }
        )

    # Step 4: Return answer and supporting metadata
    return {
        "query": query,
        "answer": answer,
        "sources": sources,
    }