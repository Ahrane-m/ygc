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
rebuild, cross-check LLM call, lab trend tracking, Cloudinary upload), so a
rejected file never pays for any of that either.

Multilingual documents: this filter never inspects raw document text or
language — every check below is a presence/type/number check on already-
extracted structured fields, which medical_extractor.py's extraction prompt
normalizes consistently (e.g. medication "ingredients" -> English INN)
regardless of what language the source document was in. That means the cost
and behavior of filtering are identical for a Tamil prescription, an Arabic
lab report, or an English discharge summary — there is no per-language
branch to add or maintain here. See the multilingual cases in this file's
__main__ self-test below for worked examples (a real Tamil prescription
kept, a Tamil-language non-medical screenshot rejected).
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


def has_medical_content(doc: Dict[str, Any]) -> bool:
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


_CONFIDENCE_MISSING = object()


def _confidence_value(doc: Dict[str, Any]) -> float:
    """The confidence used for the DECISION. Anything missing or non-numeric
    counts as 0.0 — an unreported score is not evidence of a good read."""
    raw = doc.get("overall_confidence", _CONFIDENCE_MISSING)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 0.0
    return float(raw)


def _describe_confidence(doc: Dict[str, Any]) -> str:
    """How to TALK about that confidence in an error the user reads.

    The decision coerces a missing or non-numeric score to 0.0, but saying so
    verbatim produced messages that were plainly untrue: a document with no
    score at all was reported as "overall_confidence=0.0" (implying the model
    read it and had zero confidence), and a string score produced the
    nonsense "overall_confidence=high is below 0.35". The decision is
    unchanged; only the description is honest about which case it hit.
    """
    raw = doc.get("overall_confidence", _CONFIDENCE_MISSING)
    if raw is _CONFIDENCE_MISSING:
        return "no confidence score was reported for it"
    if raw is None:
        return "its confidence score came back empty"
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return f"its confidence score was {raw!r}, which is not a number"
    return f"its confidence score of {raw} is below {LOW_CONFIDENCE_THRESHOLD}"


def _carries_demo_marker(doc: Dict[str, Any]) -> bool:
    """Whether this page carries a demo/placeholder marker.

    api.py stamps `is_demo_document` before calling this module, so the flag
    is normally already there. The fallback keeps the function correct for
    any other caller, and imports locally because medical_extractor pulls in
    the OpenAI client and this module is deliberately dependency-free.
    """
    flag = doc.get("is_demo_document")
    if isinstance(flag, bool):
        return flag
    try:
        from medical_extractor import _is_demo_document

        return _is_demo_document(doc)
    except Exception:
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

    if has_medical_content(doc):
        return True

    if doc_type in RECOGNIZED_MEDICAL_TYPES:
        # A demo/sample document typed as a real clinical document is
        # accepted without the confidence floor.
        #
        # The floor exists to catch a boarding pass the vision model mislabels
        # as a prescription: no clinical content AND no confidence in the
        # label. A demo marker is evidence pointing the other way — someone
        # deliberately made a prescription-shaped document — and the very
        # things that trip the floor (blank placeholder rows, a low-confidence
        # read of a mock layout) are what a template IS, not evidence it is
        # something else. A marker never rescues a document typed 'other';
        # that remains the genuine junk bucket.
        if _carries_demo_marker(doc):
            return True
        return _confidence_value(doc) >= LOW_CONFIDENCE_THRESHOLD

    return False


def rejection_reason(doc: Dict[str, Any]) -> str:
    """Human-readable explanation for why a document was rejected, for use
    in API error messages / logs."""
    doc_type = doc.get("document_type", "unknown")
    if doc_type not in RECOGNIZED_MEDICAL_TYPES:
        return (
            f"it was classified as '{doc_type}' and no medications, lab results "
            f"or allergies were found in it."
        )
    return (
        f"it was classified as '{doc_type}', but {_describe_confidence(doc)} "
        f"and no medications, lab results or allergies were found to support "
        f"that label."
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

    # --- Multilingual cases -------------------------------------------------
    # The filter never inspects raw document text or language — it only
    # checks the already-extracted structured fields (medications,
    # lab_results, allergies_noted, document_type, overall_confidence).
    # medical_extractor.py's extraction prompt normalizes "ingredients" to
    # English INN regardless of source language, so these fields are
    # populated the same way whether the source document was English,
    # Tamil, Arabic, or anything else — no language-specific logic is
    # needed here, and none should be added.
    tamil_prescription = {
        # corresponds to multilingual/M1.1_tamil_prescription.txt in the
        # test dataset: a real Tamil-script prescription for Metformin +
        # Amlodipine. "name" stays as printed (Tamil); "ingredients" is
        # normalized to English by the extractor.
        "document_type": "prescription",
        "medications": [
            {"name": "மெட்ஃபோர்மின்", "ingredients": ["Metformin"], "confidence": 0.88},
            {"name": "அம்லோடிபின்", "ingredients": ["Amlodipine"], "confidence": 0.85},
        ],
        "lab_results": [],
        "allergies_noted": [],
        "clinical_notes": "மருந்தளிக்கும் மருத்துவர்: டாக்டர் சு. பெரேரா",
        "overall_confidence": 0.86,
    }
    arabic_low_confidence_prescription = {
        # corresponds to multilingual/M3.3_rtl_arabic_sample.txt: a
        # right-to-left script document, correctly typed but with lower
        # confidence (handwriting/RTL uncertainty) — should still pass on
        # document_type + confidence even before considering content.
        "document_type": "prescription",
        "medications": [],
        "lab_results": [],
        "allergies_noted": [],
        "clinical_notes": None,
        "overall_confidence": 0.42,
    }
    non_medical_tamil_screenshot = {
        # a Tamil-language screenshot with no clinical content — must be
        # rejected exactly like the English conference-slide case above,
        # proving clinical_notes text is ignored regardless of language.
        "document_type": "other",
        "medications": [],
        "lab_results": [],
        "allergies_noted": [],
        "clinical_notes": "விழா அழைப்பிதழ் - திருமண அழைப்பு",  # "Event invitation - wedding invite"
        "overall_confidence": 0.7,
    }

    kept, rejected = filter_non_medical_documents([
        real_prescription, empty_other, unusual_but_real, mistagged_screenshot,
        conference_slide_screenshot, tamil_prescription,
        arabic_low_confidence_prescription, non_medical_tamil_screenshot,
    ])
    assert len(kept) == 4, f"expected 4 kept, got {len(kept)}"
    assert len(rejected) == 4, f"expected 4 rejected, got {len(rejected)}"
    assert tamil_prescription in kept
    assert arabic_low_confidence_prescription in kept
    assert non_medical_tamil_screenshot in rejected

    try:
        assert_medical_document(empty_other, "boarding_pass.jpg")
        raise SystemExit("expected NonMedicalDocumentError to be raised")
    except NonMedicalDocumentError as e:
        print("OK — correctly rejected:", e)

    # --- demo/sample documents are accepted, junk still is not ----------
    base = {"lab_results": [], "allergies_noted": [], "clinical_notes": None,
            "medications": [], "recommended_investigations": []}
    # A demo prescription is prescription-SHAPED: blank placeholder rows and a
    # low-confidence read of a mock layout are what a template is, not
    # evidence it is something else.
    assert looks_like_medical_document(
        {**base, "document_type": "prescription", "is_demo_document": True,
         "overall_confidence": 0.2})
    # Without the marker, the same empty low-confidence page is still junk.
    assert not looks_like_medical_document(
        {**base, "document_type": "prescription", "is_demo_document": False,
         "overall_confidence": 0.2})
    # A marker never rescues 'other' — that is the boarding-pass bucket.
    assert not looks_like_medical_document(
        {**base, "document_type": "other", "is_demo_document": True,
         "clinical_notes": "Gate 42 seat 14C", "overall_confidence": 0.9})
    # The flag is derived when a caller has not stamped it.
    assert looks_like_medical_document(
        {**base, "document_type": "prescription", "patient_name": "DEMO PATIENT",
         "overall_confidence": 0.2})

    print("All checks passed.")
