
import pytest

from src.pipeline import answer_question


def test_answer_question_rejects_empty_query():
    with pytest.raises(ValueError, match="Query cannot be empty"):
        answer_question(
            query="   ",
            model=None,
            index=None,
            bm25=None,
            chunks=[],
            reranker=None,
        )


def test_answer_question_returns_answer_and_sources(monkeypatch):
    import src.pipeline as pipeline_module

    fake_chunks = [
        {
            "source": "attention.pdf",
            "page": 5,
            "chunk_id": 4,
            "text": "Self-attention computes relationships between tokens.",
            "rerank_score": 0.95,
        }
    ]

    def fake_retrieve(**kwargs):
        return fake_chunks

    def fake_generate_answer(query, results, client=None):
        return "Self-attention relates tokens to one another."

    monkeypatch.setattr(
        pipeline_module,
        "retrieve_relevant_chunks",
        fake_retrieve,
    )
    monkeypatch.setattr(
        pipeline_module,
        "generate_answer",
        fake_generate_answer,
    )

    result = answer_question(
        query="What is self-attention?",
        model=object(),
        index=object(),
        bm25=object(),
        chunks=fake_chunks,
        reranker=object(),
    )

    assert result["query"] == "What is self-attention?"
    assert "self-attention" in result["answer"].lower()
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source"] == "attention.pdf"
    assert result["sources"][0]["page"] == 5
    assert result["sources"][0]["chunk_id"] == 4