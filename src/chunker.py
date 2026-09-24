import re


def split_into_paragraphs(text):
    """
    Split page text into paragraphs using blank lines.
    """
    paragraphs = re.split(r"\n\s*\n", text)

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


def create_page_chunks(
    page_texts,
    chunk_size=1500,
    overlap_paragraphs=1
):
    """
    Create paragraph-aware chunks while preserving metadata.

    Parameters
    ----------
    page_texts : list
        Page dictionaries containing:
        - source
        - page
        - text

    chunk_size : int
        Approximate maximum characters per chunk.

    overlap_paragraphs : int
        Number of paragraphs shared between consecutive chunks.
    """

    chunks = []

    for page in page_texts:

        text = page.get("text", "")
        source = page.get("source", "unknown")
        page_number = page.get("page")

        if not text.strip():
            continue

        paragraphs = split_into_paragraphs(text)

        current_chunk = []
        current_length = 0

        for paragraph in paragraphs:

            paragraph_length = len(paragraph)

            if (
                current_chunk
                and current_length + paragraph_length > chunk_size
            ):
                chunk_text = "\n\n".join(current_chunk)

                chunks.append({
                    "source": source,
                    "page": page_number,
                    "text": chunk_text
                })

                # Keep the last paragraph as overlap
                if overlap_paragraphs > 0:
                    current_chunk = current_chunk[
                        -overlap_paragraphs:
                    ]

                    current_length = sum(
                        len(p) for p in current_chunk
                    )

                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(paragraph)
            current_length += paragraph_length

        # Add remaining paragraphs
        if current_chunk:

            chunk_text = "\n\n".join(current_chunk)

            chunks.append({
                "source": source,
                "page": page_number,
                "text": chunk_text
            })

    # Assign IDs
    for chunk_id, chunk in enumerate(chunks):
        chunk["chunk_id"] = chunk_id

    return chunks