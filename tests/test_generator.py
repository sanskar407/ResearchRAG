
import pytest

from src.generator import (
    build_context,
    build_prompt,
    generate_answer,
)


def sample_results():
    return [
        {
            "source": "attention-is-all-you-need-Paper.pdf",
            "page": 5,
            "chunk_id": 4,
            "text": "Self-attention relates positions in a sequence.",
        },
        {
            "source": "deep-learning.pdf",
            "page": 2,
            "chunk_id": 7,
            "text": "Attention weights determine the contribution of values.",
        },
    ]


def test_build_context_includes_source_metadata():
    context = build_context(sample_results())

    assert "attention-is-all-you-need-Paper.pdf" in context
    assert "Page: 5" in context
    assert "Chunk: 4" in context
    assert "Self-attention relates positions" in context


def test_build_context_includes_all_results():
    context = build_context(sample_results())

    assert "SOURCE 1" in context
    assert "SOURCE 2" in context
    assert "deep-learning.pdf" in context


def test_build_prompt_includes_query_and_context():
    query = "What is self-attention?"
    context = build_context(sample_results())

    prompt = build_prompt(query, context)

    assert query in prompt
    assert context in prompt
    assert "ONLY the provided context" in prompt
    assert "Do not invent citations" in prompt


def test_generate_answer_handles_empty_results():
    answer = generate_answer(
        query="What is self-attention?",
        results=[],
    )

    assert "could not find relevant information" in answer.lower()


def test_generate_answer_uses_llm_response():
    class FakeMessage:
        content = "Self-attention relates positions in a sequence."

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["temperature"] == 0
            assert kwargs["model"] == "openai/gpt-oss-120b"
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    answer = generate_answer(
        query="What is self-attention?",
        results=sample_results(),
        client=FakeClient(),
    )

    assert answer == "Self-attention relates positions in a sequence."