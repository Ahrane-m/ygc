"""
Cloudinary storage (Phase 4)
=========================================
Uploads the original file a patient's document was extracted from (not the
per-page rendered images used for vision OCR) to Cloudinary, so the
extracted structured data can link back to the source document. One upload
per uploaded file, under a per-user folder:

    mediscan/<user_id>/<sanitized_filename>_<random8>

Env:
    CLOUDINARY_CLOUD_NAME
    CLOUDINARY_API_KEY
    CLOUDINARY_API_SECRET
"""

import os
import re
import uuid
from typing import Dict

import cloudinary
import cloudinary.uploader

_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return
    cloudinary.config(
        cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
        api_key=os.environ["CLOUDINARY_API_KEY"],
        api_secret=os.environ["CLOUDINARY_API_SECRET"],
        secure=True,
    )
    _configured = True


def _sanitize_public_id(filename: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", filename).strip("_") or "file"
    return f"{stem}_{uuid.uuid4().hex[:8]}"


def upload_patient_document(user_id: str, file_path: str, original_filename: str) -> Dict[str, str]:
    """Uploads one file to Cloudinary under mediscan/<user_id>/. Returns
    {"document_url": secure_url, "cloudinary_public_id": public_id}."""
    _configure()
    result = cloudinary.uploader.upload(
        file_path,
        folder=f"mediscan/{user_id}",
        public_id=_sanitize_public_id(original_filename),
        resource_type="auto",  # PDFs and images both need to round-trip as the original file
        overwrite=False,
    )
    return {
        "document_url": result["secure_url"],
        "cloudinary_public_id": result["public_id"],
    }
