# `medical_extractor.py` reference

Extraction → grouping → timeline → cross-check pipeline. This is the "source of truth" module: it turns raw documents into structured JSON and a per-patient timeline. [`retrieval.py`](../retrieval.py) builds on top of its output and imports `client` and `MODEL` from here — don't rename those without checking [retrieval.md](retrieval.md).

## Install / env

```
pip install openai pdfplumber pymupdf pillow --break-system-packages
export OPENAI_API_KEY="sk-..."
```

Module-level constants:
- `client` — the shared `OpenAI` client instance (reused by `retrieval.py`)
- `MODEL = "gpt-5-mini"` — vision-capable, used for extraction and cross-checking (and reused by `retrieval.py` as its chat model)
- `FALLBACK_MODEL = "gpt-5-nano"` — cheaper, not currently wired up anywhere automatic; pass `model=FALLBACK_MODEL` explicitly if needed

## 1. Extraction schema

`EXTRACTION_JSON_SCHEMA` / `EXTRACTION_RESPONSE_FORMAT` — OpenAI Structured Outputs (`strict: True`) schema that every extraction call is forced into. This is the shape of one extracted document:

```jsonc
{
  "document_type": "prescription" | "lab_report" | "discharge_summary" | "other",
  "date": "YYYY-MM-DD" | null,
  "provider_or_doctor": string | null,
  "patient_name": string | null,
  "medications": [
    {"name", "ingredients": [string], "dosage", "frequency", "duration", "confidence"}
  ],
  "lab_results": [
    {"test_name", "value", "unit", "reference_range", "flag": "normal"|"high"|"low"|"unknown", "confidence"}
  ],
  "allergies_noted": [string],
  "clinical_notes": string | null,
  "illegible_or_low_confidence_fields": [string],
  "overall_confidence": number
}
```

If you need to add a field, update `EXTRACTION_JSON_SCHEMA` **and** the `"required"` list — with `strict: True` a field missing from `required` will error, not just be optional.

`ingredients` is inferred by the model even from brand names (e.g. "Panadol" → `["Paracetamol"]`) — this is what lets `cross_check_prescriptions()` match brand-name duplicates later.

## 2. File-type detection & preprocessing

| Function | Purpose |
|---|---|
| `pdf_has_text_layer(pdf_path, min_chars=30)` | Samples first 3 pages; `True` if there's a real embedded text layer (digital PDF) |
| `extract_text_from_pdf(pdf_path)` | Pulls plain text out of a digital PDF, page by page |
| `pdf_pages_to_images(pdf_path, dpi=200)` | Rasterizes a scanned/image-only PDF into a list of `PIL.Image` (one per page) via PyMuPDF |
| `image_to_base64(img)` | PNG-encodes a `PIL.Image` to base64 for the vision API payload |

Routing rule: digital PDF → text extraction (cheap, no vision call). Scanned PDF or a bare image file → vision OCR, one API call per page.

## 3. Extraction calls

- `extract_from_image(img, model=MODEL) -> dict` — one page image → one extracted-document dict. Has a defensive fallback that strips stray markdown code fences if the model doesn't return clean JSON (shouldn't happen with `strict` mode, but kept as a safety net).
- `extract_from_text(text, model=MODEL) -> dict` — same schema, plain text input (digital PDFs).

Neither function attaches `_source` — that's the caller's job (see `process_document`).

## 4. Top-level entry points

### `process_document(file_path, model=MODEL) -> dict`

The router. Given one file path:
- Validates the path first with **friendly, specific errors** for the mistakes people actually make: path still pointing inside a `.zip`, nonexistent path, a folder passed instead of a file, unsupported extension. If you're debugging a "why did this fail" report from a teammate, read the raised exception message — it's written to be actionable, not generic.
- Digital PDF → `extract_from_text()`, result tagged `_source: {"file", "method": "text_layer"}`.
- Scanned PDF → `extract_from_image()` per page, returns `{"multi_page": True, "pages": [...]}`, each page tagged `_source: {"file", "method": "vision_ocr", "page": N}`.
- Image file → `extract_from_image()`, tagged `_source: {"file", "method": "vision_ocr"}`.

**Gotcha:** the multi-page-PDF return shape (`{"multi_page": True, "pages": [...]}`) is different from every other return shape (a flat document dict). Anything consuming a list of `process_document()` results must flatten first — that's what `_flatten_documents()` is for. Don't assume `process_document()` always returns a single dict.

### `process_patient_folder(folder_path, model=MODEL) -> List[dict]`

Walks a folder recursively (`rglob("*")`, so subfolders like "Year 1" / "Year 2" are included), processes every supported file (`.pdf .png .jpg .jpeg .webp`), and returns a flat list of whatever `process_document()` returned per file (so this list can still contain `multi_page` dicts mixed with regular ones). Catches and logs per-file failures rather than aborting the whole batch.

## 5. Grouping & timeline

### `group_documents_by_patient(raw_results, drop_demo_documents=True) -> dict[str, list[dict]]`

Run this **before** `build_patient_timeline()` whenever a batch might contain more than one patient or demo/sample data.

- Flattens via `_flatten_documents()`.
- Drops anything `_is_demo_document()` flags (patient name or a medication name containing "DEMO"/"SAMPLE"/"DUMMY") — prevents sample/template pages from polluting a real patient's timeline.
- Groups by `_normalize_patient_key()` (lowercased, stripped `patient_name`; missing/null → `"unknown_patient"`, a deliberate bucket rather than silent merging).
- Prints a warning if more than one real patient shows up in one batch — those get **separate, non-cross-checked** timelines.

Returns `{"amit sharma": [doc, doc, ...], ...}`.

### `build_patient_timeline(raw_results) -> dict`

**Assumes every document belongs to one patient already** — run `group_documents_by_patient()` first if that's not guaranteed. Output:

```python
{
    "visits": [...],                 # one entry per document, sorted by date (undated -> end)
    "medications_timeline": [...],   # every medication across all visits, flattened, each with "date" + "source_file" merged in
    "lab_results_timeline": [...],   # same, for lab results
    "known_allergies": [...],        # deduped, sorted
}
```

This dict is the primary interface between this file and `retrieval.py` — `build_chunks_from_timeline()` there consumes exactly this shape. If you change these key names or the per-entry fields (`date`, `source_file`, `name`, `ingredients`, `dosage`, `frequency`, `duration`, `test_name`, `value`, `unit`, `reference_range`, `flag`, `clinical_notes`), you **must** update the corresponding chunk-builder in `retrieval.py` too.

## 6. Cross-checking

### `cross_check_prescriptions(timeline, model=MODEL) -> dict`

Sends `medications_timeline` + `known_allergies` to the model with `CROSS_CHECK_PROMPT`, gets back:

```jsonc
{
  "potential_drug_interactions": [{"medications_involved", "explanation", "severity", "confidence"}],
  "duplicate_prescriptions": [{"medication", "occurrences", "explanation", "confidence"}],
  "conflicting_dosage_instructions": [{"medication", "conflicting_instructions", "explanation", "confidence"}],
  "allergy_conflicts": [{"medication", "allergy", "explanation", "confidence"}],
  "overall_recommendation": "always defers to a doctor/pharmacist"
}
```

Matches medications by **active ingredient**, not brand name — a duplicate flag between "Panadol" and "Tylenol" is expected behavior, not a bug. Uses `response_format={"type": "json_object"}` (looser than the `strict` schema used for extraction) — if you need guaranteed field presence here too, consider tightening this the same way `EXTRACTION_RESPONSE_FORMAT` is defined.

## 7. `__main__` — CLI flow

```
python medical_extractor.py file1.pdf file2.jpg ...
python medical_extractor.py "C:\path\to\Patient x"        # folder mode, auto-detected when a single dir is passed
python medical_extractor.py <path or files> --chat         # same, then interactive Q&A
```

Per run: extract → `group_documents_by_patient()` → for each patient: `build_patient_timeline()` → `cross_check_prescriptions()` → `index_patient_timeline()` (imported from `retrieval.py`, lazily inside `__main__` to avoid a circular import — `retrieval.py` imports `client`/`MODEL` from this module at its top level) → writes `patient_report_<sanitized_name>.json` (full timeline + cross-check report).

If `--chat` was passed: after all patients are processed, prompts you to pick a patient (if more than one), then loops `input()` → `answer_question()` → prints JSON, carrying `chat_history` forward across turns in that session. See [retrieval.md](retrieval.md) for what that call actually does.

**Note:** indexing failures are caught and logged per-patient (`  Indexing failed (Q&A won't be available for this patient): ...`) rather than aborting the run — a Chroma/embedding problem won't stop the extraction report from being written.
