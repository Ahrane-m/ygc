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
import json
import base64
from pathlib import Path
from typing import List, Dict, Any

import pdfplumber
import fitz  # PyMuPDF, used to rasterize scanned PDFs
from PIL import Image
from openai import OpenAI

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
                    "confidence": {"type": "number"},
                },
                "required": ["name", "ingredients", "dosage", "frequency", "duration", "confidence"],
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


def _is_demo_document(d: Dict[str, Any]) -> bool:
    """Detect placeholder/template documents (e.g. sample datasets that
    include a 'DEMO PATIENT' / 'DEMO MEDICINE' mock page) so they don't get
    silently treated as real patient data."""
    name = (d.get("patient_name") or "").upper()
    if "DEMO" in name or "SAMPLE" in name or "DUMMY" in name:
        return True
    for med in d.get("medications", []):
        med_name = (med.get("name") or "").upper()
        if "DEMO" in med_name or "SAMPLE" in med_name:
            return True
    return False


def _normalize_patient_key(name: Any) -> str:
    """Group documents by patient name. Missing/null names go into their
    own 'unknown_patient' bucket rather than being silently merged with
    everything else."""
    if not name or not isinstance(name, str) or not name.strip():
        return "unknown_patient"
    return name.strip().lower()


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

Rules:
- Compare medications by their active ingredients (not just brand names) —
  two different brand names with the same active ingredient is a likely
  duplicate.
- Only flag interactions you have reasonable clinical confidence about;
  lower the confidence score rather than omitting a plausible risk.
- Do not diagnose. Do not tell the patient to stop or start any medication.
  Always defer to a licensed professional.
"""


def cross_check_prescriptions(timeline: Dict[str, Any], model: str = MODEL) -> Dict[str, Any]:
    """
    Runs interaction / duplicate / dosage-conflict / allergy cross-checking
    over a patient's merged medication timeline (output of
    build_patient_timeline).
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
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# 7. Example usage — full pipeline: extract -> merge -> cross-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import re

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single/multiple files:  python medical_extractor.py file1.pdf file2.jpg ...")
        print("  Whole patient folder:   python medical_extractor.py \"C:\\path\\to\\Patient x\"")
        sys.exit(1)

    args = sys.argv[1:]

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

        output = {
            "patient_key": patient_key,
            "patient_timeline": timeline,
            "cross_check_report": cross_check,
        }

        safe_name = re.sub(r"[^a-z0-9_]+", "_", patient_key.lower()).strip("_") or "patient"
        out_path = f"patient_report_{safe_name}.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Saved report to {out_path}")
        print(f"  Documents in timeline: {len(timeline['visits'])}")
        print(f"  Medications tracked: {len(timeline['medications_timeline'])}")
        print(f"  Interaction flags: {len(cross_check.get('potential_drug_interactions', []))}")
        print(f"  Duplicate flags: {len(cross_check.get('duplicate_prescriptions', []))}")
        print(f"  Dosage conflict flags: {len(cross_check.get('conflicting_dosage_instructions', []))}")
