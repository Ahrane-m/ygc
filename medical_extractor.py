"""
Medical Document Extraction Pipeline
=====================================
Handles PDF (text-based or scanned) and image uploads (prescriptions, lab
reports, discharge summaries), extracts structured data using an OpenAI
vision-capable model, and returns clean JSON ready for timeline building,
RAG indexing, and cross-checking.

Install:
    pip install openai pdfplumber pymupdf pillow --break-system-packages

Env:
    export OPENAI_API_KEY="sk-..."
"""

import os
import io
import re
import json
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pdfplumber
import fitz  # PyMuPDF, used to rasterize scanned PDFs
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MODEL = "gpt-5-mini"       # vision-capable, low cost, good enough for structured extraction
FALLBACK_MODEL = "gpt-5-nano"    # even cheaper, use for high-volume / less critical docs

 #---------------------------------------------------------------------------
# 1. Extraction schema — keeps every document's output shape consistent
# ---------------------------------------------------------------------------

EXTRACTION_SCHEMA_PROMPT = """
You are a medical document extraction engine. You will be shown an image of
a medical document (prescription, lab report, or discharge summary).

Extract every field defined in the JSON schema provided. For medications,
always attempt to identify the active ingredient(s) using your medical
knowledge, even if the document only prints a brand name (e.g. brand
"Panadol" -> ingredients ["Paracetamol"]). Use an empty array only if the
ingredient is genuinely unknown/undeterminable.

CONFIDENCE SCORING — anchor every confidence value to these bands. Do not
default to a high score; think about which band actually applies before
writing a number:
- 0.90-1.00: text is clearly printed/typed and the field maps to the schema
  with no judgment required.
- 0.60-0.89: text is legible but you had to exercise judgment — expanding an
  abbreviation, reading a partially cut-off table cell, or inferring an
  active ingredient from a brand name that was NOT itself printed on the
  document.
- Below 0.60: handwriting is genuinely hard to read, the text is blurry or
  cut off, or you are inferring a value rather than reading one directly off
  the page.
A medication's active ingredient being inferred (not printed) rather than
read directly is, by itself, enough to keep that medication's confidence
below 0.90 — brand-to-generic mapping is your knowledge substituting for
what the document actually says, not a transcription.

LANGUAGE AND UNIT NORMALIZATION — documents may be in any language, and a
patient's timeline may combine documents from several languages. Two
prescriptions for the same drug at the same dose must be recognizable as
the same regardless of what language or units each was printed in:
- ingredients must always be the English INN (International Nonproprietary
  Name) / generic drug name, regardless of the document's language (e.g.
  "Amoxicilina" (Spanish) or "アモキシシリン" (Japanese) -> ingredients:
  ["Amoxicillin"]).
- dosage and frequency stay exactly as printed, in the original language —
  these are for human/audit display, so a reviewer can see literally what
  the document said.
- dosage_value / dosage_unit are your best-effort NORMALIZED numeric dose,
  independent of source language: "500 mg" -> dosage_value=500,
  dosage_unit="mg"; "0.5 g" -> dosage_value=500, dosage_unit="mg" (convert
  mass units to mg so entries become directly comparable); "5 mL" ->
  dosage_value=5, dosage_unit="mL" (do not convert volume/count/unit-based
  dosing). Use null for both if the dose can't be reduced to one
  value+unit (e.g. a titration schedule).
- frequency_per_day is your best-effort NORMALIZED doses-per-day count,
  independent of source language or phrasing — "cada 8 horas" (Spanish),
  "3 fois par jour" (French), and "3x daily" (English) must all normalize
  to frequency_per_day=3. Set is_as_needed=true (and frequency_per_day=
  null) for PRN/as-needed dosing with no fixed daily count. Set
  is_as_needed=false and frequency_per_day=null only if genuinely
  indeterminate.
- Watch for locale-specific number formatting: some locales use a comma as
  the decimal separator (e.g. "1,5 g" means 1.5 grams, not 15 grams).
  Misreading this is a real dosing error, not a cosmetic one.
- Translating an ingredient name, converting a unit, or resolving a
  frequency phrase is itself inference, not transcription — factor that
  into the medication's confidence the same way an inferred brand-to-
  generic mapping is.

Rules:
- If handwriting is unclear, make your best guess but LOWER the confidence
  score for that field and add a note to illegible_or_low_confidence_fields.
- Never invent data. Use null for missing string fields (per the schema).
- Do not provide medical advice or diagnosis — extraction only.
"""

# Strict JSON Schema (OpenAI Structured Outputs) — guarantees every field,
# including "ingredients", is always present in the response.
EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {
            "type": "string",
            "enum": ["prescription", "lab_report", "discharge_summary", "other"],
        },
        "date": {"type": ["string", "null"]},
        "provider_or_doctor": {"type": ["string", "null"]},
        "patient_name": {"type": ["string", "null"]},
        "medications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "ingredients": {"type": "array", "items": {"type": "string"}},
                    "dosage": {"type": "string"},
                    "frequency": {"type": "string"},
                    "duration": {"type": ["string", "null"]},
                    "dosage_value": {"type": ["number", "null"]},
                    "dosage_unit": {"type": ["string", "null"]},
                    "frequency_per_day": {"type": ["number", "null"]},
                    "is_as_needed": {"type": "boolean"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "name", "ingredients", "dosage", "frequency", "duration",
                    "dosage_value", "dosage_unit", "frequency_per_day", "is_as_needed",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        },
        "lab_results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "test_name": {"type": "string"},
                    "value": {"type": "string"},
                    "unit": {"type": ["string", "null"]},
                    "reference_range": {"type": ["string", "null"]},
                    "flag": {"type": "string", "enum": ["normal", "high", "low", "unknown"]},
                    "confidence": {"type": "number"},
                },
                "required": ["test_name", "value", "unit", "reference_range", "flag", "confidence"],
                "additionalProperties": False,
            },
        },
        "allergies_noted": {"type": "array", "items": {"type": "string"}},
        "clinical_notes": {"type": ["string", "null"]},
        "illegible_or_low_confidence_fields": {"type": "array", "items": {"type": "string"}},
        "overall_confidence": {"type": "number"},
    },
    "required": [
        "document_type", "date", "provider_or_doctor", "patient_name",
        "medications", "lab_results", "allergies_noted", "clinical_notes",
        "illegible_or_low_confidence_fields", "overall_confidence",
    ],
    "additionalProperties": False,
}

EXTRACTION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "medical_document_extraction",
        "strict": True,
        "schema": EXTRACTION_JSON_SCHEMA,
    },
}


# ---------------------------------------------------------------------------
# 2. File-type detection and preprocessing
# ---------------------------------------------------------------------------

def pdf_has_text_layer(pdf_path: str, min_chars: int = 30) -> bool:
    """Quick check: does this PDF have a usable embedded text layer?"""
    with pdfplumber.open(pdf_path) as pdf:
        total_chars = 0
        for page in pdf.pages[:3]:  # sample first few pages only
            text = page.extract_text() or ""
            total_chars += len(text.strip())
        return total_chars >= min_chars


def extract_text_from_pdf(pdf_path: str) -> str:
    """Direct text extraction for digital PDFs (no OCR/vision needed)."""
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            chunks.append(f"--- Page {i + 1} ---\n{text}")
    return "\n\n".join(chunks)


def pdf_pages_to_images(pdf_path: str, dpi: int = 200) -> List[Image.Image]:
    """Render each page of a scanned/image-only PDF into a PIL image."""
    images = []
    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    doc.close()
    return images


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# 3. Vision extraction call
# ---------------------------------------------------------------------------

def extract_from_image(img: Image.Image, model: str = MODEL) -> Dict[str, Any]:
    """Send a single page image to the vision model and parse structured JSON."""
    b64 = image_to_base64(img)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACTION_SCHEMA_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract structured data from this medical document image.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            },
        ],
        response_format=EXTRACTION_RESPONSE_FORMAT,
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Defensive fallback: strip stray code fences if the model added them
        cleaned = raw.strip().strip("`").replace("json\n", "", 1)
        return json.loads(cleaned)


def extract_from_text(text: str, model: str = MODEL) -> Dict[str, Any]:
    """For digital PDFs — run the same schema extraction on plain text."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACTION_SCHEMA_PROMPT},
            {
                "role": "user",
                "content": f"Extract structured data from this document text:\n\n{text}",
            },
        ],
        response_format=EXTRACTION_RESPONSE_FORMAT,
    )
    return json.loads(response.choices[0].message.content)


VISION_OCR_CONFIDENCE_CEILING = 0.85  # a vision/handwriting read is never "fully certain"


def _apply_confidence_ceiling(result: Dict[str, Any], ceiling: float) -> Dict[str, Any]:
    """
    Caps every confidence value in an extraction result at `ceiling`. Used
    for vision_ocr-sourced documents (scanned PDFs, photographed
    prescriptions) so a model's self-reported 0.95 on a handwriting read
    can't outrank what the extraction method itself can actually support.
    Text-layer (text_layer) extractions are left uncapped since those come
    from a digital text source, not a visual read.
    """
    if "overall_confidence" in result and isinstance(result["overall_confidence"], (int, float)):
        result["overall_confidence"] = min(result["overall_confidence"], ceiling)
    for med in result.get("medications", []) or []:
        if isinstance(med.get("confidence"), (int, float)):
            med["confidence"] = min(med["confidence"], ceiling)
    for lab in result.get("lab_results", []) or []:
        if isinstance(lab.get("confidence"), (int, float)):
            lab["confidence"] = min(lab["confidence"], ceiling)
    return result


# ---------------------------------------------------------------------------
# 4. Top-level entry point — routes any uploaded file correctly
# ---------------------------------------------------------------------------

def process_document(file_path: str, model: str = MODEL) -> Dict[str, Any]:
    """
    Accepts a path to a PDF or image file. Detects type and routes to the
    right extraction path. Returns structured JSON (or a list of per-page
    JSON objects for multi-page scanned PDFs).
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    # --- Friendly diagnostics for the most common mistakes ---
    if ".zip" in file_path.lower():
        raise ValueError(
            "This path still points INSIDE a .zip file — that doesn't work. "
            "Right-click the .zip in File Explorer, choose 'Extract All', "
            "then re-run this script pointing at the EXTRACTED folder "
            "(the path should not contain '.zip' anywhere)."
        )
    if not path.exists():
        raise FileNotFoundError(
            f"Path does not exist: {file_path}\n"
            "  Common causes: the .zip wasn't extracted yet, a typo in the "
            "path, or a trailing backslash right before a closing quote "
            "(e.g. \"...\\Year 1\\\" breaks Windows' command-line parsing — "
            "remove the final backslash so it ends \"...\\Year 1\")."
        )
    if path.is_dir():
        raise IsADirectoryError(
            f"'{file_path}' is a folder, not a file. Pass it directly to "
            "process_patient_folder(), or from the command line just run: "
            f'python medical_extractor.py "{file_path}"  (without pointing '
            "at a specific file — the script auto-detects folders)."
        )
    if suffix not in (".pdf", ".png", ".jpg", ".jpeg", ".webp"):
        raise ValueError(
            f"Unsupported file type '{suffix or '(no extension)'}' for "
            f"'{file_path}'. Supported: .pdf, .png, .jpg, .jpeg, .webp"
        )
    # --- End diagnostics ---

    if suffix == ".pdf":
        if pdf_has_text_layer(file_path):
            text = extract_text_from_pdf(file_path)
            result = extract_from_text(text, model=model)
            result["_source"] = {"file": path.name, "method": "text_layer"}
            return result
        else:
            # Scanned PDF -> render pages -> vision extraction per page
            pages = pdf_pages_to_images(file_path)
            page_results = []
            for i, img in enumerate(pages):
                res = extract_from_image(img, model=model)
                res = _apply_confidence_ceiling(res, VISION_OCR_CONFIDENCE_CEILING)
                res["_source"] = {
                    "file": path.name,
                    "method": "vision_ocr",
                    "page": i + 1,
                }
                page_results.append(res)
            return {"multi_page": True, "pages": page_results}

    else:  # image types
        img = Image.open(file_path)
        result = extract_from_image(img, model=model)
        result = _apply_confidence_ceiling(result, VISION_OCR_CONFIDENCE_CEILING)
        result["_source"] = {"file": path.name, "method": "vision_ocr"}
        return result


def process_patient_folder(folder_path: str, model: str = MODEL) -> List[Dict[str, Any]]:
    """
    Walks a patient's folder (including subfolders like 'Year 1', 'Year 2')
    and processes every supported document it finds. Returns a flat list of
    extraction results, same shape as calling process_document() repeatedly.
    """
    supported = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in supported
    )

    if not files:
        print(f"No supported documents found in {folder_path}")
        return []

    results = []
    for f in files:
        print(f"Extracting {f} ...")
        try:
            result = process_document(str(f), model=model)
            results.append(result)
        except Exception as e:
            print(f"  Failed: {e}")

    return results


# ---------------------------------------------------------------------------
# 5. Timeline builder — merge multiple documents into one patient timeline
# ---------------------------------------------------------------------------

def _flatten_documents(raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Multi-page scanned PDFs return {'multi_page': True, 'pages': [...]}.
    Flatten everything into a single flat list of per-document dicts."""
    flat = []
    for r in raw_results:
        if r.get("multi_page"):
            flat.extend(r["pages"])
        else:
            flat.append(r)
    return flat


# Best-effort, non-exhaustive placeholder markers across the languages this
# pipeline is expected to see (see EXTRACTION_SCHEMA_PROMPT's language/unit
# normalization section). patient_name and medication "name" are NOT
# translated by the extractor (only "ingredients" is normalized to English),
# so an English-only check misses a demo document printed in another
# language. This list can't cover every language a patient's documents
# might be in — it's a pragmatic net for the common cases, not a guarantee.
DEMO_PLACEHOLDER_MARKERS = frozenset({
    "DEMO", "SAMPLE", "DUMMY", "TEST PATIENT", "PLACEHOLDER",   # English
    "மாதிரி", "டெமோ",                                             # Tamil (sample / demo)
    "නියැදිය", "ආදර්ශ",                                          # Sinhala (sample / model-example)
    "डेमो", "नमूना", "उदाहरण",                                    # Hindi (demo / sample / example)
    "MUESTRA", "EJEMPLO", "PRUEBA",                               # Spanish (sample / example / test)
    "EXEMPLE", "ÉCHANTILLON",                                     # French (example / sample)
    "デモ", "サンプル",                                             # Japanese (demo / sample)
    "تجريبي", "عينة", "نموذج",                                    # Arabic (trial/demo / sample / model)
})


def _is_demo_document(d: Dict[str, Any]) -> bool:
    """Detect placeholder/template documents (e.g. sample datasets that
    include a 'DEMO PATIENT' / 'DEMO MEDICINE' mock page) so they don't get
    silently treated as real patient data. Checks both English and a
    best-effort set of transliterated equivalents (see
    DEMO_PLACEHOLDER_MARKERS) since patient/medication names are printed in
    the document's original language, not translated during extraction."""
    name = (d.get("patient_name") or "").upper()
    if any(marker in name for marker in DEMO_PLACEHOLDER_MARKERS):
        return True
    for med in d.get("medications", []):
        med_name = (med.get("name") or "").upper()
        if any(marker in med_name for marker in DEMO_PLACEHOLDER_MARKERS):
            return True
    return False


def _normalize_patient_key(name: Any) -> str:
    """Group documents by patient name. Missing/null names go into their
    own 'unknown_patient' bucket rather than being silently merged with
    everything else."""
    if not name or not isinstance(name, str) or not name.strip():
        return "unknown_patient"
    return name.strip().lower()


def find_patient_name_mismatch(
    labeled_pages: List[Tuple[str, Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """
    Checks a list of (label, extracted_page) pairs — e.g. filenames paired
    with their process_document() result — for more than one distinct
    patient name (via _normalize_patient_key). Pages with no name
    ("unknown_patient") are ignored rather than treated as a conflict.

    Returns None if every named page agrees, otherwise a dict describing
    the distinct names found and which label each came from, for the
    caller to surface as a confirmation prompt.
    """
    by_name: Dict[str, List[str]] = {}
    for label, page in labeled_pages:
        key = _normalize_patient_key(page.get("patient_name"))
        if key == "unknown_patient":
            continue
        by_name.setdefault(key, []).append(label)

    if len(by_name) <= 1:
        return None

    return {
        "distinct_patient_names": sorted(by_name),
        "documents": [
            {"label": label, "patient_name": name}
            for name, labels in by_name.items()
            for label in labels
        ],
    }


def group_documents_by_patient(
    raw_results: List[Dict[str, Any]], drop_demo_documents: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Splits a flat list of extracted documents into groups keyed by patient
    name. This prevents unrelated prescriptions (e.g. a folder that
    accidentally contains sample docs for different people) from being
    merged into one timeline and cross-checked against each other.

    Returns: { "amit sharma": [doc, doc, ...], "mary smith": [...], ... }
    Also prints a warning if more than one distinct real patient is found,
    or if demo/placeholder documents were dropped.
    """
    docs = _flatten_documents(raw_results)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    dropped = []

    for d in docs:
        if drop_demo_documents and _is_demo_document(d):
            dropped.append(d.get("_source", {}).get("file", "unknown_file"))
            continue
        key = _normalize_patient_key(d.get("patient_name"))
        groups.setdefault(key, []).append(d)

    if dropped:
        print(f"  Skipped {len(dropped)} demo/placeholder document(s): {dropped}")

    real_patients = [k for k in groups if k != "unknown_patient"]
    if len(real_patients) > 1:
        print(
            f"  WARNING: found {len(real_patients)} distinct patient names in this "
            f"batch ({real_patients}) — building a SEPARATE timeline for each, "
            f"they will NOT be cross-checked against one another."
        )

    return groups


def build_patient_timeline(raw_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge extracted documents (output of process_document, one per file) into
    a single chronological patient timeline: one entry per visit/document,
    sorted by date, plus flattened rollups of all medications and lab
    results for easy downstream cross-checking.

    NOTE: assumes all documents passed in already belong to ONE patient.
    Use group_documents_by_patient() first if a batch might mix patients
    or contain demo/placeholder documents.
    """
    docs = _flatten_documents(raw_results)

    # Sort by date; undated docs go to the end
    def sort_key(d):
        return d.get("date") or "9999-99-99"

    docs_sorted = sorted(docs, key=sort_key)

    all_medications = []
    all_lab_results = []
    all_allergies = set()

    for d in docs_sorted:
        visit_date = d.get("date")
        source_file = d.get("_source", {}).get("file")

        for med in d.get("medications", []):
            all_medications.append({**med, "date": visit_date, "source_file": source_file})

        for lab in d.get("lab_results", []):
            all_lab_results.append({**lab, "date": visit_date, "source_file": source_file})

        for allergy in d.get("allergies_noted", []) or []:
            all_allergies.add(allergy)

    return {
        "visits": docs_sorted,               # one entry per document, chronological
        "medications_timeline": all_medications,
        "lab_results_timeline": all_lab_results,
        "known_allergies": sorted(all_allergies),
    }


# ---------------------------------------------------------------------------
# 6. Cross-checking — interactions, duplicates, conflicting dosages
# ---------------------------------------------------------------------------

CROSS_CHECK_PROMPT = """
You are a clinical safety cross-checking assistant. You are given a
patient's full medication timeline (medications prescribed across multiple
visits, each with a date and source document) and their known allergies.

Analyze the list and return STRICT JSON (no markdown, no commentary) in
this shape:

{
  "potential_drug_interactions": [
    {
      "medications_involved": ["Drug A", "Drug B"],
      "explanation": "plain language explanation of the interaction risk",
      "severity": "low | moderate | high",
      "confidence": 0.0-1.0
    }
  ],
  "duplicate_prescriptions": [
    {
      "medication": "string",
      "occurrences": [{"date": "YYYY-MM-DD", "source_file": "string", "dosage": "string"}],
      "explanation": "why this looks like a duplicate",
      "confidence": 0.0-1.0
    }
  ],
  "conflicting_dosage_instructions": [
    {
      "medication": "string",
      "conflicting_instructions": [{"date": "YYYY-MM-DD", "source_file": "string", "dosage": "string", "frequency": "string"}],
      "explanation": "what conflicts and why it matters",
      "confidence": 0.0-1.0
    }
  ],
  "allergy_conflicts": [
    {
      "medication": "string",
      "allergy": "string",
      "explanation": "string",
      "confidence": 0.0-1.0
    }
  ],
  "overall_recommendation": "1-2 sentence plain-language summary that ALWAYS recommends the patient consult a doctor or pharmacist before making any changes. Never present this as a diagnosis."
}

CONFIDENCE SCORING — anchor every confidence value to these bands. Do not
default to a high score:
- 0.90-1.00: the interaction/conflict/duplicate is well-established,
  unambiguous clinical knowledge (e.g. a textbook contraindicated pairing,
  an exact-ingredient duplicate).
- 0.60-0.89: plausible and worth surfacing, but depends on dose, timing, or
  patient-specific factors you cannot verify from this data alone.
- Below 0.60: a weak or speculative signal — include it only if omitting it
  would be the more dangerous error, and mark it clearly as low-confidence.

Rules:
- Compare medications by their active ingredients (not just brand names) —
  two different brand names with the same active ingredient is a likely
  duplicate.
- Medications are the SAME regardless of source language or printed
  wording — compare using ingredients (already normalized to English
  generic names), dosage_value + dosage_unit, and frequency_per_day
  (already normalized numeric fields), NOT the original dosage/frequency
  text. Do not flag something as a conflict or a difference if it is only
  a translation or unit-formatting difference — e.g. "500 mg" and "0.5 g"
  that both normalized to dosage_value=500/dosage_unit="mg" are the SAME
  dose, not a conflict. Only flag genuine differences in the normalized
  values.
- Only flag interactions you have reasonable clinical confidence about;
  lower the confidence score rather than omitting a plausible risk.
- Do not diagnose. Do not tell the patient to stop or start any medication.
  Always defer to a licensed professional.
- You are a reasoning layer over extracted text, NOT a validated clinical
  drug-interaction database. overall_recommendation must state plainly that
  this analysis is not a substitute for a pharmacist or a licensed
  drug-interaction checking tool, in addition to recommending consultation.
"""


CROSS_CHECK_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "potential_drug_interactions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "medications_involved": {"type": "array", "items": {"type": "string"}},
                    "explanation": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "moderate", "high"]},
                    "confidence": {"type": "number"},
                },
                "required": ["medications_involved", "explanation", "severity", "confidence"],
                "additionalProperties": False,
            },
        },
        "duplicate_prescriptions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "medication": {"type": "string"},
                    "occurrences": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "date": {"type": ["string", "null"]},
                                "source_file": {"type": ["string", "null"]},
                                "dosage": {"type": ["string", "null"]},
                            },
                            "required": ["date", "source_file", "dosage"],
                            "additionalProperties": False,
                        },
                    },
                    "explanation": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["medication", "occurrences", "explanation", "confidence"],
                "additionalProperties": False,
            },
        },
        "conflicting_dosage_instructions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "medication": {"type": "string"},
                    "conflicting_instructions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "date": {"type": ["string", "null"]},
                                "source_file": {"type": ["string", "null"]},
                                "dosage": {"type": ["string", "null"]},
                                "frequency": {"type": ["string", "null"]},
                            },
                            "required": ["date", "source_file", "dosage", "frequency"],
                            "additionalProperties": False,
                        },
                    },
                    "explanation": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["medication", "conflicting_instructions", "explanation", "confidence"],
                "additionalProperties": False,
            },
        },
        "allergy_conflicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "medication": {"type": "string"},
                    "allergy": {"type": "string"},
                    "explanation": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["medication", "allergy", "explanation", "confidence"],
                "additionalProperties": False,
            },
        },
        "overall_recommendation": {"type": "string"},
    },
    "required": [
        "potential_drug_interactions", "duplicate_prescriptions",
        "conflicting_dosage_instructions", "allergy_conflicts",
        "overall_recommendation",
    ],
    "additionalProperties": False,
}

CROSS_CHECK_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "medical_cross_check",
        "strict": True,
        "schema": CROSS_CHECK_JSON_SCHEMA,
    },
}


def detect_exact_duplicate_medications(timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Deterministic (non-LLM) duplicate detection using the normalized
    ingredients + dosage_value + dosage_unit fields set during extraction.

    Why this exists alongside the LLM cross-check: the LLM pass is
    instructed to compare medications via normalized fields rather than
    raw printed text, but it's still a probabilistic reasoning step run
    once per patient. An exact match on ingredient set + numeric dose,
    across two different source documents, is something code can determine
    for certain — independent of what language either document was
    written in — and shouldn't depend on the model reliably catching it
    every single time. This function only flags matches it can verify
    exactly; anything looser (different doses that might still interact,
    brand-name-only duplicates with no normalized dose available) is left
    to the LLM pass, which remains the primary check.
    """
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for med in timeline.get("medications_timeline", []):
        ingredients = tuple(sorted(med.get("ingredients") or []))
        dosage_value = med.get("dosage_value")
        dosage_unit = med.get("dosage_unit")
        if not ingredients or dosage_value is None or not dosage_unit:
            continue  # nothing normalized to compare — leave this one to the LLM pass
        key = (ingredients, dosage_value, dosage_unit)
        groups.setdefault(key, []).append(med)

    duplicates: List[Dict[str, Any]] = []
    for (ingredients, dosage_value, dosage_unit), meds in groups.items():
        distinct_sources = {(m.get("date"), m.get("source_file")) for m in meds}
        if len(distinct_sources) < 2:
            continue  # same medication appearing once is not a duplicate
        duplicates.append({
            "medication": " / ".join(ingredients),
            "occurrences": [
                {"date": m.get("date"), "source_file": m.get("source_file"), "dosage": m.get("dosage")}
                for m in meds
            ],
            "explanation": (
                f"Deterministic check: identical active ingredient(s) ({', '.join(ingredients)}) "
                f"at the same normalized dose ({dosage_value} {dosage_unit}) appear in "
                f"{len(distinct_sources)} separate documents, regardless of source language or "
                "printed wording."
            ),
            "confidence": 0.95,  # exact numeric/ingredient match, not model inference
        })
    return duplicates


def cross_check_prescriptions(timeline: Dict[str, Any], model: str = MODEL) -> Dict[str, Any]:
    """
    Runs interaction / duplicate / dosage-conflict / allergy cross-checking
    over a patient's merged medication timeline (output of
    build_patient_timeline). Merges in a deterministic, language-
    independent duplicate check (see detect_exact_duplicate_medications)
    alongside the LLM's own duplicate detection, rather than relying on
    the LLM pass alone to catch exact cross-language matches.
    """
    payload = {
        "medications_timeline": timeline["medications_timeline"],
        "known_allergies": timeline["known_allergies"],
    }

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CROSS_CHECK_PROMPT},
            {
                "role": "user",
                "content": f"Patient medication data:\n\n{json.dumps(payload, indent=2)}",
            },
        ],
        response_format=CROSS_CHECK_RESPONSE_FORMAT,
    )
    result = json.loads(response.choices[0].message.content)

    deterministic_duplicates = detect_exact_duplicate_medications(timeline)
    existing = result.setdefault("duplicate_prescriptions", [])
    existing_source_sets = [
        frozenset((occ.get("date"), occ.get("source_file")) for occ in d.get("occurrences", []))
        for d in existing
    ]
    for dup in deterministic_duplicates:
        dup_sources = frozenset((occ["date"], occ["source_file"]) for occ in dup["occurrences"])
        if dup_sources not in existing_source_sets:
            existing.append(dup)

    return result


# ---------------------------------------------------------------------------
# 7. Persistence helpers — patient report / raw-document cache on disk
# ---------------------------------------------------------------------------
# Shared by the CLI (__main__ below) and the HTTP API (api.py), so a patient
# processed via either entry point is visible to the other. Two files per
# patient:
#   patient_docs_<name>.json   - raw extracted per-document dicts (flattened,
#                                pre-timeline), so a later API upload can
#                                merge new documents in rather than
#                                replacing the patient's whole history.
#   patient_report_<name>.json - the merged {"patient_key", "patient_timeline",
#                                "cross_check_report"} snapshot, same shape
#                                the CLI has always written.

def _safe_patient_filename(patient_key: str) -> str:
    """Maps a patient_key into a filesystem-safe name for the two files
    above. Same sanitization the CLI used to do inline."""
    return re.sub(r"[^a-z0-9_]+", "_", patient_key.lower()).strip("_") or "patient"


def _patient_docs_path(patient_key: str) -> str:
    return f"patient_docs_{_safe_patient_filename(patient_key)}.json"


def _patient_report_path(patient_key: str) -> str:
    return f"patient_report_{_safe_patient_filename(patient_key)}.json"


def load_patient_documents(patient_key: str) -> List[Dict[str, Any]]:
    """Loads the raw extracted documents previously saved for this patient
    via save_patient_documents(). Returns [] if this patient has never been
    processed before (nothing to merge new uploads into)."""
    path = _patient_docs_path(patient_key)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def save_patient_documents(patient_key: str, docs: List[Dict[str, Any]]) -> None:
    """Persists the full raw extracted-document list for a patient (flat,
    already run through _flatten_documents) so a future upload can extend
    it instead of overwriting this patient's document history."""
    with open(_patient_docs_path(patient_key), "w") as f:
        json.dump(docs, f, indent=2)


def load_patient_report(patient_key: str) -> Optional[Dict[str, Any]]:
    """Loads the {"patient_key", "patient_timeline", "cross_check_report"}
    snapshot previously written for this patient, or None if this patient
    hasn't been processed yet."""
    path = _patient_report_path(patient_key)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_patient_report(
    patient_key: str,
    timeline: Dict[str, Any],
    cross_check: Dict[str, Any],
    lab_trends: Optional[Dict[str, Any]] = None,
) -> None:
    """Writes the merged timeline + cross-check report (+ optional lab
    trend analysis) to disk — same shape and naming convention the CLI
    __main__ flow has always used. `lab_trends` is optional so callers
    (and old saved reports on disk, loaded back via load_patient_report())
    that predate lab trend tracking keep working unchanged."""
    output = {
        "patient_key": patient_key,
        "patient_timeline": timeline,
        "cross_check_report": cross_check,
    }
    if lab_trends is not None:
        output["lab_trends"] = lab_trends
    with open(_patient_report_path(patient_key), "w") as f:
        json.dump(output, f, indent=2)


# ---------------------------------------------------------------------------
# 8. Example usage — full pipeline: extract -> merge -> cross-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single/multiple files:  python medical_extractor.py file1.pdf file2.jpg ...")
        print("  Whole patient folder:   python medical_extractor.py \"C:\\path\\to\\Patient x\"")
        print("  Then chat about it:     add --chat to either form above")
        sys.exit(1)

    args = sys.argv[1:]
    chat_mode = "--chat" in args
    args = [a for a in args if a != "--chat"]

    for a in args:
        if ".zip" in a.lower():
            print(f"ERROR: This path still points inside a .zip file:\n  {a}")
            print("Extract the zip first (right-click -> Extract All in File Explorer),")
            print("then re-run this script pointing at the extracted folder.")
            sys.exit(1)

    # Step 1: extract — folder mode if a single directory was passed, else file list
    if len(args) == 1 and Path(args[0]).is_dir():
        print(f"Scanning folder: {args[0]}")
        all_results = process_patient_folder(args[0])
    else:
        all_results = []
        for file_path in args:
            print(f"Extracting {file_path} ...")
            try:
                result = process_document(file_path)
                all_results.append(result)
            except Exception as e:
                print(f"  Failed: {e}")

    if not all_results:
        print("No documents were successfully extracted. Exiting.")
        sys.exit(1)

    # Step 2: split by patient name, dropping demo/placeholder documents.
    # This stops unrelated prescriptions (e.g. sample docs for different
    # people sitting in the same folder) from being merged into one
    # timeline and cross-checked against each other.
    print("\nGrouping documents by patient ...")
    patient_groups = group_documents_by_patient(all_results, drop_demo_documents=True)

    if not patient_groups:
        print("No real (non-demo) documents remained after filtering. Exiting.")
        sys.exit(1)

    # Step 3 + 4: for EACH distinct patient found, merge into a timeline and
    # cross-check independently.
    for patient_key, docs in patient_groups.items():
        print(f"\n=== Patient: {patient_key} ({len(docs)} document(s)) ===")
        print("Building patient timeline ...")
        timeline = build_patient_timeline(docs)

        print("Cross-checking prescriptions ...")
        cross_check = cross_check_prescriptions(timeline)

        print("Tracking lab result trends ...")
        from lab_trends import track_lab_trends
        lab_trends = track_lab_trends(timeline)

        # Persist raw docs too (not just the merged report) so a later API
        # upload for this same patient can merge new documents in. Saving the
        # report is also what makes this patient answerable via --chat:
        # retrieval.py reads the saved report directly, so there is no
        # separate index to build.
        save_patient_documents(patient_key, docs)
        save_patient_report(patient_key, timeline, cross_check, lab_trends=lab_trends)
        out_path = _patient_report_path(patient_key)

        print(f"Saved report to {out_path}")
        print(f"  Documents in timeline: {len(timeline['visits'])}")
        print(f"  Medications tracked: {len(timeline['medications_timeline'])}")
        print(f"  Interaction flags: {len(cross_check.get('potential_drug_interactions', []))}")
        print(f"  Duplicate flags: {len(cross_check.get('duplicate_prescriptions', []))}")
        print(f"  Dosage conflict flags: {len(cross_check.get('conflicting_dosage_instructions', []))}")

    # Step 5 (optional): interactive Q&A over whatever was just indexed.
    if chat_mode:
        patient_keys = list(patient_groups.keys())
        if len(patient_keys) == 1:
            active_patient = patient_keys[0]
        else:
            print("\nMultiple patients were processed:")
            for i, k in enumerate(patient_keys):
                print(f"  [{i}] {k}")
            choice = input("Select a patient index to chat about: ").strip()
            try:
                active_patient = patient_keys[int(choice)]
            except (ValueError, IndexError):
                print("Invalid selection. Exiting.")
                sys.exit(1)

        from conversation import get_or_create_session, ask as conversation_ask

        print(f"\nChatting about patient '{active_patient}'. Type 'exit' to quit.")
        session = get_or_create_session(active_patient, session_id="cli")
        while True:
            question = input("\nQuestion: ").strip()
            if question.lower() in ("exit", "quit"):
                break
            if not question:
                continue
            try:
                result = conversation_ask(session, question)
            except Exception as e:
                print(f"  Error: {e}")
                continue
            print(f"  [retrieval query]: {result.get('rewritten_query')}")
            print(json.dumps(result, indent=2))
