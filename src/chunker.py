def create_page_chunks(
    page_texts,
    chunk_size=1000,
    overlap=100
):
    """
    Split page-level text into overlapping chunks.

    Returns chunks with metadata such as:
    - chunk_id
    - page
    - source
    - text
    """

    chunks = []

    for page in page_texts:

        text = page["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append({
                    "page": page["page"],
                    "text": chunk_text
                })

            start += chunk_size - overlap

    for chunk_id, chunk in enumerate(chunks):
        chunk["chunk_id"] = chunk_id

    return chunks