
import base64
import mimetypes
from pathlib import Path

from src.generator import get_groq_client


def caption_image(image_path, source, page, client=None):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if client is None:
        client = get_groq_client()

    mime_type = mimetypes.guess_type(str(image_path))[0]
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image type: {image_path}")

    encoded_image = base64.b64encode(
        image_path.read_bytes()
    ).decode("utf-8")

    prompt = """
    Analyze this image extracted from a research paper.

    Describe:
    1. What the figure or diagram represents.
    2. Its important components and their relationships.
    3. Any visible labels, equations, or key findings.

    Be precise and concise. Do not guess details that are
    not visible. Return only the description.
    """

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{mime_type};base64,"
                                f"{encoded_image}"
                            )
                        },
                    },
                ],
            }
        ],
        temperature=0.2,
        max_completion_tokens=600,
    )

    return {
        "source": source,
        "page": page,
        "image_path": str(image_path),
        "description": response.choices[0].message.content,
    }
