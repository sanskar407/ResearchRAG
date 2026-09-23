def create_page_chunks(
    page_texts,
    chunk_size=1000,
    overlap=100
):
    """
    Split page-level text into overlapping chunks.

    Each chunk contains:
    - chunk_id
    - source
    - page
    - text
    """

    chunks = []

    for page in page_texts:

        text = page.get("text", "")
        source = page.get("source", "unknown")

        if not text.strip():
            continue

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append({
                    "source": source,
                    "page": page["page"],
                    "text": chunk_text
                })

            start += chunk_size - overlap

    for chunk_id, chunk in enumerate(chunks):
        chunk["chunk_id"] = chunk_id

    return chunks