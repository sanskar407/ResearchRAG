
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