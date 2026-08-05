"""
Non-Medical Document Filter
=========================================
Guards against files that pass the extension check in medical_extractor.py
(.pdf/.png/.jpg/.jpeg/.webp) but aren't actually medical documents — a
boarding pass, a receipt, a screenshot, a random photo. Nothing upstream
rejects them: process_document() will happily run OCR/vision extraction on
any image or PDF and hand back a structurally-valid (if empty/junk) result,
because the extraction JSON schema's document_type enum includes "other" as
a catch-all rather than failing.

Efficiency: this filter does NOT make a second model call. It re-uses the
document_type / medications / lab_results / allergies_noted / clinical_notes
fields already produced by process_document()'s single extraction call, and
applies cheap, local, deterministic checks on that structure. So the cost of
filtering is O(1) dict lookups — no extra OCR, no extra OpenAI request, no
added latency — and it runs *before* the expensive downstream work (timeline
rebuild, cross-check LLM call, Chroma re-indexing), so a rejected file never
pays for any of that either.
"""

from typing import Any, Dict, List, Tuple

# document_type values the extraction schema recognizes as genuinely
# clinical. "other" is the extractor's catch-all for anything that isn't
# one of these — which is exactly what a boarding pass / receipt / random
# photo will come back as.
RECOGNIZED_MEDICAL_TYPES = frozenset({"prescription", "lab_report", "discharge_summary"})

# Below this, an "other"-typed extraction with no clinical content is
# treated as noise rather than a low-confidence-but-real medical document.
LOW_CONFIDENCE_THRESHOLD = 0.35


class NonMedicalDocumentError(ValueError):
    """Raised when an uploaded file's extraction result doesn't look like
    a medical document. Carries the filename so callers can build a clear,
    per-file error message without re-deriving context."""

    def __init__(self, filename: str, reason: str):
        self.filename = filename
        self.reason = reason
        super().__init__(f"'{filename}' does not appear to be a medical document: {reason}")


def _has_medical_content(doc: Dict[str, Any]) -> bool:
    """True if the extraction actually pulled out *structured* clinical
    substance — at least one medication, lab result, or noted allergy.

    Deliberately does NOT treat a non-empty clinical_notes string as
    evidence on its own: unlike medications/lab_results/allergies_noted,
    clinical_notes is populated generically with whatever text the vision
    model transcribes off the page — a conference-slide screenshot or a
    Zoom participant list produces a perfectly non-empty, well-formed
    clinical_notes description with zero clinical content in it. The
    structured fields only get populated when the model recognizes an
    actual medication/lab/allergy entity, so they're the reliable signal."""
    if doc.get("medications"):
        return True
    if doc.get("lab_results"):
        return True
    if doc.get("allergies_noted"):
        return True
    return False


def looks_like_medical_document(doc: Dict[str, Any]) -> bool:
    """
    Decides whether one extraction result (the dict returned by
    process_document() for a single file/page) represents a real medical
    document.

    A document passes if either:
      - it actually contains *structured* medical content (medications,
        lab results, or allergies) — regardless of what document_type got
        assigned, e.g. an intake note typed "other" that still mentions
        an allergy, OR
      - its document_type is one of the recognized clinical types
        (prescription / lab_report / discharge_summary) AND its
        overall_confidence meets LOW_CONFIDENCE_THRESHOLD.

    The document_type label alone is NOT sufficient: a vision model can
    mistag a non-medical image (screenshot, boarding pass, receipt) as a
    recognized clinical type while extracting no actual clinical content
    and reporting low confidence. Requiring confidence too closes that
    hole — content is the strong signal, a merely-labeled-but-unconfident
    "prescription" with nothing in it is not.

    A document typed "other" with no clinical content is rejected outright.
    """
    doc_type = doc.get("document_type")

    if _has_medical_content(doc):
        return True

    if doc_type in RECOGNIZED_MEDICAL_TYPES:
        confidence = doc.get("overall_confidence", 0.0)
        if not isinstance(confidence, (int, float)):
            confidence = 0.0
        return confidence >= LOW_CONFIDENCE_THRESHOLD

    return False


def rejection_reason(doc: Dict[str, Any]) -> str:
    """Human-readable explanation for why a document was rejected, for use
    in API error messages / logs."""
    doc_type = doc.get("document_type", "unknown")
    confidence = doc.get("overall_confidence", 0.0)
    if doc_type not in RECOGNIZED_MEDICAL_TYPES:
        return (
            f"classified as '{doc_type}' with no medications, lab results, or "
            f"allergies found (overall_confidence={confidence})."
        )
    return (
        f"classified as '{doc_type}' but overall_confidence={confidence} is below "
        f"{LOW_CONFIDENCE_THRESHOLD} and no medications, lab results, or allergies "
        f"were found to support that label."
    )


def filter_non_medical_documents(
    docs: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits a list of extraction results into (kept, rejected) without any
    additional model calls. `kept` preserves order; `rejected` entries keep
    the original dict so a caller can still log/inspect what was thrown out.
    """
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for doc in docs:
        if looks_like_medical_document(doc):
            kept.append(doc)
        else:
            rejected.append(doc)
    return kept, rejected


def assert_medical_document(doc: Dict[str, Any], filename: str) -> None:
    """Raises NonMedicalDocumentError if `doc` (one file's extraction
    result) doesn't look medical. Intended to be called once per uploaded
    file, right after process_document(), before the result is merged into
    the patient's timeline."""
    if not looks_like_medical_document(doc):
        raise NonMedicalDocumentError(filename, rejection_reason(doc))


if __name__ == "__main__":
    # Lightweight self-test, no pytest dependency needed.
    real_prescription = {
        "document_type": "prescription",
        "medications": [{"name": "Amoxicillin", "confidence": 0.9}],
        "lab_results": [],
        "allergies_noted": [],
        "clinical_notes": None,
        "overall_confidence": 0.9,
    }
    empty_other = {
        "document_type": "other",
        "medications": [],
        "lab_results": [],
        "allergies_noted": [],
        "clinical_notes": None,
        "overall_confidence": 0.4,
    }
    unusual_but_real = {
        "document_type": "other",
        "medications": [],
        "lab_results": [],
        "allergies_noted": ["Penicillin"],
        "clinical_notes": "Patient noted a penicillin allergy on intake form.",
        "overall_confidence": 0.6,
    }
    mistagged_screenshot = {
        # what a non-medical screenshot can look like when the vision
        # model mistags document_type instead of picking "other"
        "document_type": "lab_report",
        "medications": [],
        "lab_results": [],
        "allergies_noted": [],
        "clinical_notes": None,
        "overall_confidence": 0.2,
    }
    conference_slide_screenshot = {
        # real observed case: document_type correctly "other", but
        # clinical_notes is a non-empty verbatim OCR transcription of a
        # conference slide / Zoom window — no structured medical content
        "document_type": "other",
        "medications": [],
        "lab_results": [],
        "allergies_noted": [],
        "clinical_notes": "Presentation slide: 'Welcome Note VarDial 2026'. Zoom meeting participant panel visible.",
        "overall_confidence": 0.78,
    }

    kept, rejected = filter_non_medical_documents(
        [real_prescription, empty_other, unusual_but_real, mistagged_screenshot, conference_slide_screenshot]
    )
    assert len(kept) == 2, f"expected 2 kept, got {len(kept)}"
    assert len(rejected) == 3, f"expected 3 rejected, got {len(rejected)}"

    try:
        assert_medical_document(empty_other, "boarding_pass.jpg")
        raise SystemExit("expected NonMedicalDocumentError to be raised")
    except NonMedicalDocumentError as e:
        print("OK — correctly rejected:", e)

    print("All checks passed.")
