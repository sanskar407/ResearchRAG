
from pathlib import Path

import pymupdf


def extract_pdf_images(pdf_path, output_dir):
    """
    Extract embedded images from a PDF.

    Returns metadata for each extracted image.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_images = []

    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            for image_index, image_info in enumerate(
                page.get_images(full=True), start=1
            ):
                xref = image_info[0]
                image_data = document.extract_image(xref)

                extension = image_data["ext"]
                image_bytes = image_data["image"]

                image_path = output_dir / (
                    f"{pdf_path.stem}_page_{page_index + 1}"
                    f"_image_{image_index}.{extension}"
                )

                image_path.write_bytes(image_bytes)

                extracted_images.append({
                    "source": pdf_path.name,
                    "page": page_index + 1,
                    "image_id": image_index,
                    "image_path": str(image_path),
                })

    return extracted_images
