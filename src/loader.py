from pathlib import Path
from pypdf import PdfReader


def load_pdf(pdf_path):
    """
    Extract text from a PDF while preserving page information.
    """

    pdf_path = Path(pdf_path)

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        pages.append({
            "page": page_number + 1,
            "text": text
        })

    return pages