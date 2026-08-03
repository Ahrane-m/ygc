# Medical Document Pipeline

Two modules:

- [`medical_extractor.py`](../medical_extractor.py) — extraction, timeline building, cross-checking
- [`retrieval.py`](../retrieval.py) — retrieval-augmented Q&A over an already-built timeline (Phase 1)

## 1. Extraction (`medical_extractor.py`)

Turns a PDF or image (prescription, lab report, discharge summary) into structured JSON using an OpenAI vision-capable model (`MODEL = "gpt-5-mini"`, fallback `"gpt-5-nano"`).

- `pdf_has_text_layer()` / `extract_text_from_pdf()` — digital PDFs go straight to text extraction, skipping vision.
- `pdf_pages_to_images()` — scanned PDFs are rasterized per page and sent through vision OCR instead.
- `extract_from_image()` / `extract_from_text()` — call the model with `EXTRACTION_JSON_SCHEMA` (OpenAI Structured Outputs, `strict: True`) so every field is always present: `document_type`, `date`, `provider_or_doctor`, `patient_name`, `medications[]` (with inferred `ingredients`), `lab_results[]`, `allergies_noted[]`, `clinical_notes`, `overall_confidence`.
- `process_document()` — top-level router: detects file type, picks the right path, raises friendly errors for common mistakes (path still inside a `.zip`, missing file, folder passed where a file was expected, unsupported extension).
- `process_patient_folder()` — walks a folder (including subfolders like "Year 1", "Year 2") and processes every supported file in it.

## 2. Grouping and Timeline (`medical_extractor.py`)

- `group_documents_by_patient()` — splits a batch of extracted documents by `patient_name` so unrelated patients' documents never get merged into one timeline. Drops demo/placeholder documents (`_is_demo_document()`) by default and warns if more than one real patient is found in a single batch.
- `build_patient_timeline()` — merges one patient's documents into:
  ```
  {
    "visits": [...],                 # one entry per document, sorted by date
    "medications_timeline": [...],   # every medication, flattened, with date + source_file
    "lab_results_timeline": [...],   # every lab result, flattened, with date + source_file
    "known_allergies": [...]         # deduped, sorted
  }
  ```

## 3. Cross-checking (`medical_extractor.py`)

- `cross_check_prescriptions()` — sends the medication timeline + allergies to the model and gets back a safety report: `potential_drug_interactions`, `duplicate_prescriptions` (matched by active ingredient, not brand name), `conflicting_dosage_instructions`, `allergy_conflicts`, plus an `overall_recommendation` that always defers to a doctor/pharmacist. Never diagnoses or tells the patient to start/stop a medication.

## 4. Retrieval-Augmented Q&A (`retrieval.py`, Phase 1)

Sits on top of the **already-extracted** structured timeline — it does not re-read raw documents.

```
timeline → chunks → embeddings (text-embedding-3-small) → per-patient Chroma collection (./chroma_db)
question → embedding → top_k similarity search → cited context → chat model → structured JSON answer
```

### Chunking

`build_chunks_from_timeline()` produces one chunk per:

| chunk_type       | source                                   |
|-------------------|-------------------------------------------|
| `medication`       | each entry in `medications_timeline`       |
| `lab_result`        | each entry in `lab_results_timeline`        |
| `clinical_note`     | each visit with non-null `clinical_notes`   |
| `allergy`           | one chunk listing all `known_allergies`      |

Each chunk has a natural-language `text` (what gets embedded) and `metadata` (`patient_key`, `date`, `source_file`, `chunk_type`).

### Storage

- One Chroma collection per patient, name = sanitized `patient_key` (`_sanitize_collection_name()`), persisted to `./chroma_db` via `chromadb.PersistentClient`.
- Chunk IDs are deterministic (`sha256(patient_key|source_file|chunk_type|index)`), so `index_patient_timeline()` can be called repeatedly on the same documents — it `upsert()`s instead of duplicating.

### Answering

`answer_question(patient_key, question, chat_history=None, top_k=8)`:

1. Embeds the question.
2. Queries the patient's collection for the `top_k` most similar chunks.
3. Builds a prompt with retrieved chunks (tagged with date/source_file), optional `chat_history`, and the question.
4. Calls the chat model under a system prompt that:
   - answers **only** from retrieved context, saying "I don't have enough information" otherwise
   - never gives a diagnosis
   - forces `recommend_professional_consult: true` for anything touching risk, interactions, or dosage changes
   - requires structured JSON output: `{"answer", "confidence", "sources": [{"date", "source_file"}], "recommend_professional_consult"}`

Returns a graceful "no information" answer (no API calls) if the patient was never indexed or their collection is empty. Raises `ValueError` for a missing `patient_key`/`question`, `RuntimeError` if an OpenAI call fails.

## 5. Wiring (`medical_extractor.py` `__main__`)

```
python medical_extractor.py <file1> <file2> ...      # or a folder path
python medical_extractor.py <path> --chat             # same, then drops into an interactive Q&A loop
```

For each patient found: build timeline → cross-check → `index_patient_timeline()` → write `patient_report_<name>.json`. If `--chat` was passed, prompts for a patient (if more than one was processed) and loops on `input()` → `answer_question()` → prints the JSON result, keeping running `chat_history`.

## Dependencies

```
pip install openai pdfplumber pymupdf pillow chromadb --break-system-packages
```

```
export OPENAI_API_KEY="sk-..."
```

## Status / Next steps

- Phase 1 (this doc) covers Q&A grounded in structured, already-extracted fields only.
- Not yet implemented: retrieval over raw document text/images, multi-patient comparison queries, auth/access control around `./chroma_db`, evaluation harness for answer quality.
