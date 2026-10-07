
import re


def extract_citations(answer):
    """
    Extract citations formatted as:
    [document.pdf, p. 5]
    """

    pattern = r"\[([^\[\]\n]+?\.pdf),\s*p\.\s*(\d+)\]"

    matches = re.findall(
        pattern,
        answer,
        flags=re.IGNORECASE
    )

    return [
        {
            "source": source.strip(),
            "page": int(page)
        }
        for source, page in matches
    ]


def evaluate_citations(answer, retrieved_chunks):
    """
    Check whether generated citations match
    the retrieved source documents and pages.
    """

    citations = extract_citations(answer)

    valid_sources = {
        (chunk["source"], chunk["page"])
        for chunk in retrieved_chunks
    }

    valid_citations = [
        citation
        for citation in citations
        if (citation["source"], citation["page"])
        in valid_sources
    ]

    total = len(citations)

    precision = (
        len(valid_citations) / total
        if total > 0 else 0.0
    )

    return {
        "total_citations": total,
        "valid_citations": len(valid_citations),
        "citation_precision": precision,
        "has_citations": total > 0,
        "all_citations_valid": (
            total > 0 and len(valid_citations) == total
        )
    }

def evaluate_grounding(answer, retrieved_chunks):
    """
    Estimate whether important answer terms are present
    in the retrieved context.

    This is a lightweight lexical baseline, not a semantic
    or LLM-based faithfulness evaluator.
    """

    if not answer.strip() or not retrieved_chunks:
        return {
            "grounding_score": 0.0,
            "supported_terms": 0,
            "total_terms": 0,
        }

    context = " ".join(
        chunk["text"]
        for chunk in retrieved_chunks
    ).lower()

    # Remove citation metadata from the answer.
    answer_text = re.sub(
        r"\[[^\]]+\]",
        "",
        answer.lower()
    )

    # Extract meaningful words.
    words = re.findall(
        r"\b[a-zA-Z]{4,}\b",
        answer_text
    )

    # Remove common English words.
    stopwords = {
        "this", "that", "these", "those",
        "with", "from", "they", "their",
        "there", "which", "where", "when",
        "what", "does", "have", "been",
        "using", "used", "into", "also",
        "than", "then", "such", "only",
        "provided", "documents", "information",
    }

    meaningful_words = [
        word for word in words
        if word not in stopwords
    ]

    if not meaningful_words:
        return {
            "grounding_score": 0.0,
            "supported_terms": 0,
            "total_terms": 0,
        }

    supported_terms = sum(
        1
        for word in meaningful_words
        if word in context
    )

    grounding_score = (
        supported_terms / len(meaningful_words)
    )

    return {
        "grounding_score": grounding_score,
        "supported_terms": supported_terms,
        "total_terms": len(meaningful_words),
    }