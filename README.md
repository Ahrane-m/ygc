# Medical Records Extraction, Retrieval & Q&A

Turns uploaded medical documents (prescriptions, lab reports, discharge
summaries) into a structured per-patient timeline, cross-checks it for
safety issues, and answers natural-language questions about it — as a
single-shot Q&A call or a real multi-turn conversation. Exposed over HTTP
under `/api/v1/`, scoped per authenticated user.

```
documents --extract--> Cloudinary (file) + MongoDB (structured data)
                |
                +--> timeline --cross-check--> safety report
                        |         |
                        |         +--trend-track--> lab result trends
                        |
                        +--> patient snapshot (MongoDB)
                                        |
                                 question / conversation
                                        |
                          context assembled from the whole record
                                        |
                                    JSON answer
```

There is no vector store. A patient's record is small enough to answer from
whole, and every question this product exists to answer is a completeness
question ("what am I taking?", "did my dose change?") rather than a
similarity one — so retrieval is deterministic assembly of the saved
snapshot, not approximate nearest-neighbour search over chunks. See
[`docs/retrieval.md`](docs/retrieval.md) for the reasoning and the
budget-constrained fallback.

| Module | Responsibility |
|---|---|
| [`medical_extractor.py`](medical_extractor.py) | Extraction, timeline building, cross-checking, on-disk persistence (CLI) |
| [`document_filter.py`](document_filter.py) | Rejects non-medical uploads (post-extraction, no extra API call) |
| [`lab_trends.py`](lab_trends.py) | Tracks each lab test across visits — direction of drift, reference-range crossings, plain-language explanation (deterministic, no LLM call) |
| [`retrieval.py`](retrieval.py) | Cross-document context assembly + single-shot Q&A, with the deterministic consult/confidence guard (Phase 1) |
| [`conversation.py`](conversation.py) | Multi-turn sessions, query rewriting, entity focus carry-over, safety-aware summarization (Phase 2) |
| [`api.py`](api.py) | HTTP API over all of the above (Phase 3) |
| [`auth.py`](auth.py) | Verifies the `Authorization`/`X-User-Id` headers on every API request (Phase 4) |
| [`db.py`](db.py) | MongoDB persistence for uploaded documents, patient snapshots, and conversation sessions, scoped per user (Phase 4) |
| [`storage.py`](storage.py) | Uploads original documents to Cloudinary under `mediscan/<user_id>/` (Phase 4) |
| [`inspect_records.py`](inspect_records.py) | Read-only CLI showing exactly what context a question would be answered from |
| [`generate_lab_test_data.py`](generate_lab_test_data.py) | Generates synthetic, schema-valid lab_report test data — no OCR/API calls needed |

Deeper internals for each module are documented in [`docs/`](docs/).

## Setup

```
pip install -r requirements.txt
```

Create a `.env` file in the project root (already gitignored):

```
OPENAI_API_KEY=sk-...
CLOUDINARY_CLOUD_NAME=...
CLOUDINARY_API_KEY=...
CLOUDINARY_API_SECRET=...
MONGODB_URI=mongodb+srv://...
JWT_SECRET=...          # same secret your auth issuer signs tokens with

# Only needed for the antidote knowledge graph (poisoning_kg.py)
NEO4J_URI=neo4j+s://...
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
NEO4J_DATABASE=neo4j    # optional, defaults to "neo4j"
```

Optional tuning:

| Variable | Default | Effect |
|---|---|---|
| `QA_CONTEXT_BUDGET_CHARS` | `48000` | How much of a patient's record can go into one question's context before retrieval narrows to a planned subset. |
| `MONGODB_TIMEOUT_MS` | `8000` | Server-selection timeout, so an unreachable database fails a request rather than hanging it. |

## Running the API

```
python -m uvicorn api:app --reload

```

- Base URL: `http://127.0.0.1:8000`
- Interactive docs (Swagger UI): `http://127.0.0.1:8000/docs`

All application routes are under the `/api/v1/` prefix.

---

## Deploying to Railway

`requirements.txt` and `Procfile` (`web: uvicorn api:app --host 0.0.0.0 --port $PORT`)
are already set up — Railway's Nixpacks builder detects both automatically,
so a plain "Deploy from GitHub repo" works with no extra build config.

1. **Env vars** — in the Railway service's Variables tab, set everything
   listed under [Setup](#setup) (`OPENAI_API_KEY`, `CLOUDINARY_CLOUD_NAME`,
   `CLOUDINARY_API_KEY`, `CLOUDINARY_API_SECRET`, `MONGODB_URI`,
   `JWT_SECRET`). Don't upload `.env` itself — it's git-ignored and holds
   live secrets.
   If you use the antidote knowledge graph, also set `NEO4J_URI`,
   `NEO4J_USERNAME`, `NEO4J_PASSWORD` (and optionally `NEO4J_DATABASE`).
2. **No volume needed.** The API keeps nothing on the container filesystem —
   patient snapshots and conversation sessions both live in MongoDB, and
   original files live in Cloudinary. Railway rebuilds the filesystem on
   every deploy, so this matters: a restart loses nothing, and the service
   is safe to run with more than one worker.
3. Deploy. Railway assigns `$PORT` automatically; the `Procfile` binds to
   it.

---

## Authentication

Every route except `/health` requires two headers:

```
Authorization: Bearer <jwt>
X-User-Id: <user_id>
```

The JWT is verified locally (HS256, `JWT_SECRET`) — no database round-trip.
The user id claim inside the token (`user_id` / `userId` / `id` / `_id` /
`sub`, whichever is present) must match `X-User-Id`, or the request is
rejected with `401`. There is one patient per user account: the
authenticated `user_id` scopes every read and write, so one user can never
see or modify another user's documents, timeline, or sessions.

## API Reference

### Health

#### `GET /api/v1/health`

```
curl http://127.0.0.1:8000/api/v1/health
```

```json
{"status": "ok"}
```

---

### Documents & Timeline

#### `POST /api/v1/documents`

Uploads one or more files (`multipart/form-data`, field name `files`) for
the authenticated user. Extracts each, archives the original file to
Cloudinary (`mediscan/<user_id>/...`), **merges the structured data with
any documents previously uploaded by this user**, rebuilds the timeline,
re-runs cross-checking, and re-indexes for Q&A. Supported extensions:
`.pdf .png .jpg .jpeg .webp`.

```
curl -X POST http://127.0.0.1:8000/api/v1/documents \
  -H "Authorization: Bearer $TOKEN" -H "X-User-Id: $USER_ID" \
  -F "files=@prescription_march.pdf" \
  -F "files=@lab_report_april.jpg"
```

Response `201`:

```json
{
  "user_id": "6620a1f2...",
  "documents_added": 2,
  "documents_total": 2,
  "timeline": {
    "visits": [
      {
        "document_type": "prescription",
        "date": "2026-03-14",
        "provider_or_doctor": "Dr. Rao",
        "patient_name": "Amit Sharma",
        "medications": [
          {
            "name": "Amoxicillin",
            "ingredients": ["Amoxicillin"],
            "dosage": "500mg",
            "frequency": "3x daily",
            "duration": "7 days",
            "confidence": 0.95
          }
        ],
        "lab_results": [],
        "allergies_noted": ["Penicillin"],
        "clinical_notes": "Patient presented with sinus infection.",
        "illegible_or_low_confidence_fields": [],
        "overall_confidence": 0.93,
        "_source": {"file": "prescription_march.pdf", "method": "text_layer"},
        "document_url": "https://res.cloudinary.com/.../mediscan/6620a1f2.../prescription_march_pdf_a1b2c3d4.pdf",
        "cloudinary_public_id": "mediscan/6620a1f2.../prescription_march_pdf_a1b2c3d4"
      }
    ],
    "medications_timeline": [
      {
        "name": "Amoxicillin",
        "ingredients": ["Amoxicillin"],
        "dosage": "500mg",
        "frequency": "3x daily",
        "duration": "7 days",
        "confidence": 0.95,
        "date": "2026-03-14",
        "source_file": "prescription_march.pdf"
      }
    ],
    "lab_results_timeline": [],
    "known_allergies": ["Penicillin"]
  },
  "cross_check_report": {
    "potential_drug_interactions": [],
    "duplicate_prescriptions": [],
    "conflicting_dosage_instructions": [],
    "allergy_conflicts": [
      {
        "medication": "Amoxicillin",
        "allergy": "Penicillin",
        "explanation": "Amoxicillin is a penicillin-class antibiotic and may trigger a reaction in patients with a penicillin allergy.",
        "confidence": 0.9
      }
    ],
    "overall_recommendation": "Please consult your doctor or pharmacist before continuing this medication given your documented penicillin allergy."
  },
  "lab_trends": {
    "trends": [],
    "insufficient_data": [],
    "note": "This trend analysis is computed directly from the extracted lab values and reference ranges — it is not a diagnosis and does not account for clinical context beyond the numbers shown. Consult the patient's doctor or a pharmacist to interpret what any trend means for their care."
  },
  "indexed": true
}
```

If indexing fails (e.g. embeddings API error), `indexed: false` and an
`index_error` field are included instead — the timeline/cross-check are
still returned and saved.

Errors: `400` no files / unsupported extension, `422` extraction failed for
a given file, `422` a file extracted successfully but doesn't look like a
medical document (see below). A document whose extracted identity doesn't
match this account's other documents is not an error — see "Different-patient
/ identity mismatch detection" below; the request still returns `201`.

**Non-medical document rejection** — passing the `.pdf`/`.png`/`.jpg` file
extension check doesn't mean a file *is* a medical document (a boarding
pass or a receipt still uploads fine as an image). After extraction,
[`document_filter.py`](document_filter.py) checks the result's
`document_type` and clinical content (medications / lab results /
allergies / notes) and rejects it with `422` before any timeline/cross-check/
indexing work happens — no second model call, it just re-uses the
extraction that already ran:

```json
{"detail": "'boarding_pass.jpg' does not appear to be a medical document: classified as 'other' with no medications, lab results, allergies, or clinical notes found (overall_confidence=0.4)."}
```

For multi-page PDFs, each page is checked individually and the page number
is included in the error (`'file.pdf (page 2)'`).

**Different-patient / identity mismatch detection** — this app is one
patient per account, so [`identity_guard.py`](identity_guard.py) checks
each newly-uploaded document's extracted identity (name, age, gender)
against this account's **document history only** — the identity on
documents already on file for this user. It deliberately does **not**
compare against the account holder's registered profile name, since that
name is set at signup and is often not the patient's own name (a
caregiver's account, a nickname, a transliteration choice) — using it as
ground truth produced false positives.

A brand new account has no document history yet, so a first-ever upload is
only checked against *itself*:
- If every document in the batch agrees on one patient, none of them are
  second-guessed, no matter whose name that is.
- If the batch itself disagrees (e.g. one file says "Ramesh", another says
  "Suresh", with no prior history for either), the larger name-group in
  the batch is treated as the baseline and the rest are held for
  confirmation — the same treatment a mismatch against existing history
  gets.

Matching rules:
- **Name** is fuzzy-matched (not exact), since OCR/handwriting reads and
  transliteration introduce spelling variance a genuine same-person upload
  shouldn't be penalized for.
- **Age** is never compared directly (age legitimately differs between
  documents taken years apart) — each document's `patient_age` is combined
  with its `date` to estimate a birth year, and birth-year estimates are
  compared instead, with tolerance for rounding.
- **Gender**, if present on both sides, is compared directly.
- No single weak signal holds a document back alone (e.g. one borderline
  name spelling, or a missing gender field) — it takes either one strong
  signal or two corroborating weaker signals together. See the named
  threshold constants at the top of `identity_guard.py` for the exact
  scoring rule.

**Partial acceptance, not all-or-nothing** — a batch is never fully
rejected over one mismatched file. Documents that match proceed
immediately (uploaded to Cloudinary, merged into the timeline, persisted);
only the documents that don't match are held out of *that* request
entirely. The response is still `201`, with an added
`identity_review_needed` field describing what was held and why:

```json
{
  "user_id": "6620a1f2...",
  "documents_added": 1,
  "documents_total": 3,
  "timeline": { "...": "reflects only the documents that were added" },
  "cross_check_report": { "...": "..." },
  "lab_trends": { "...": "..." },
  "identity_review_needed": {
    "error": "patient_name_mismatch",
    "message": "1 of 2 uploaded document group(s) (Suresh Babu) don't match the patient on your other document(s) and were not added. Confirm to add them anyway, or leave them out.",
    "known_identity": {
      "document_patient_names": ["Ramesh Kumar"],
      "estimated_birth_year": 1975,
      "gender": "male"
    },
    "held_documents": [
      {
        "patient_name": "Suresh Babu",
        "estimated_birth_year": 1995,
        "gender": "male",
        "source_files": ["suresh_report.pdf"],
        "message": "\"Suresh Babu\" doesn't match the patient on your other document(s).",
        "signals": [
          {
            "field": "name",
            "extracted_value": "Suresh Babu",
            "known_value": "Ramesh Kumar",
            "similarity": 0.31,
            "severity": "strong",
            "explanation": "\"Suresh Babu\" is only 31% similar to the patient name on your other document(s)."
          }
        ],
        "score": 2,
        "threshold": 2
      }
    ],
    "documents": [
      {"label": "ramesh_report.pdf", "patient_name": "Ramesh Kumar"},
      {"label": "suresh_report.pdf", "patient_name": "Suresh Babu"}
    ]
  }
}
```

If the held document(s) really do belong to this account, resubmit just
those file(s) (by name, from `held_documents[].source_files`) with
`confirm_name_mismatch=true` added to the form data — that request skips
the check entirely and adds them. There's no need to resend documents that
already succeeded.

#### `GET /api/v1/timeline`

Returns the authenticated user's last saved timeline (same shape as the
`timeline` field above).

```
curl -H "Authorization: Bearer $TOKEN" -H "X-User-Id: $USER_ID" \
  http://127.0.0.1:8000/api/v1/timeline
```

`404` if this user has never uploaded a document.

#### `GET /api/v1/cross-check`

Returns the authenticated user's last saved cross-check report (same shape
as `cross_check_report` above).

```
curl -H "Authorization: Bearer $TOKEN" -H "X-User-Id: $USER_ID" \
  http://127.0.0.1:8000/api/v1/cross-check
```

`404` if this user has never uploaded a document.

#### `GET /api/v1/lab-trends`

Returns the authenticated user's lab result trends (same shape as
`lab_trends` above) — per-test direction of drift across visits, when/if
it crossed out of the reference range, and a plain-language explanation.
Computed by [`lab_trends.py`](lab_trends.py) deterministically from the
numbers already in the timeline (no LLM call).

```
curl -H "Authorization: Bearer $TOKEN" -H "X-User-Id: $USER_ID" \
  http://127.0.0.1:8000/api/v1/lab-trends
```

```json
{
  "trends": [
    {
      "test_name": "Fasting Glucose",
      "unit": "mg/dL",
      "reference_range": "70-99",
      "data_points": [
        {"date": "05 Jan 2026", "value": "91", "flag": "normal", "source_file": "John_Lab_Report_1.pdf"},
        {"date": "20 Apr 2026", "value": "103", "flag": "high", "source_file": "John_Lab_Report_2.pdf"},
        {"date": "30 Aug 2026", "value": "118", "flag": "high", "source_file": "John_Lab_Report_3.pdf"}
      ],
      "direction": "increasing",
      "flag_sequence": "normal → high → high",
      "crossed_into_abnormal_at": {"date": "20 Apr 2026", "flag": "high"},
      "approaching_threshold": false,
      "confidence": 0.95,
      "explanation": "Fasting Glucose has risen across 3 tests (reference range 70-99 mg/dL), from 91 mg/dL to 118 mg/dL ... It moved from within the normal range into the 'high' range starting with the 20 Apr 2026 test, and has stayed there since."
    }
  ],
  "insufficient_data": [
    {"test_name": "TSH", "reason": "only 1 usable data point(s) with a parseable date and numeric value (need at least 2 to establish a trend); 0 entrie(s) were dropped."}
  ],
  "note": "This trend analysis is computed directly from the extracted lab values and reference ranges — it is not a diagnosis and does not account for clinical context beyond the numbers shown. Consult the patient's doctor or a pharmacist to interpret what any trend means for their care."
}
```

A test still flagged `"normal"` can still show `"approaching_threshold": true`
if it's been drifting toward a reference-range boundary across visits (e.g.
Creatinine rising from 0.92 → 1.08 → 1.32 against a 0.74–1.35 range) — this
surfaces that drift before it's officially out of range, not just after.

Tests with fewer than 2 usable (dated + numeric) readings are listed under
`insufficient_data` with a reason, rather than a fabricated single-point
"trend". Reports saved before this feature existed don't have a
`lab_trends` field on disk — this endpoint recomputes it on the fly from
the saved timeline in that case.

`404` if this patient has never been processed.

---

### Single-shot Q&A (Phase 1)

#### `POST /api/v1/qa`

Answers one question grounded in the authenticated user's processed
records. No server-side session — if you want multi-turn context, pass
`chat_history` yourself, or use the conversation endpoints below instead
(they also carry entity focus across turns, which `chat_history` alone
does not).

Request body:

```json
{
  "question": "What was I prescribed for my sinus infection?",
  "chat_history": []
}
```

`chat_history` is optional. `top_k` is still accepted so existing clients
keep working, but it is **ignored** — retrieval assembles the whole record
rather than the *k* nearest chunks.

```
curl -X POST http://127.0.0.1:8000/api/v1/qa \
  -H "Authorization: Bearer $TOKEN" -H "X-User-Id: $USER_ID" \
  -H "Content-Type: application/json" \
  -d '{"question": "What was I prescribed for my sinus infection?"}'
```

Response `200`:

```json
{
  "answer": "You were prescribed Amoxicillin 500mg, three times daily for 7 days, on 2026-03-14.",
  "confidence": 0.9,
  "sources": [
    {
      "date": "2026-03-14",
      "source_file": "prescription_march.pdf",
      "document_type": "prescription",
      "document_url": "https://res.cloudinary.com/..."
    }
  ],
  "cross_document": false,
  "recommend_professional_consult": false,
  "low_confidence": false,
  "retrieval": {"strategy": "full_record", "context_chars": 3184, "plan": null}
}
```

| Field | Meaning |
|---|---|
| `sources` | Every record the answer relied on. `document_type` and `document_url` are filled in from the timeline in code, not by the model. |
| `cross_document` | Whether the answer combined facts from more than one document. |
| `recommend_professional_consult` | Set by the model **and** forced on by a deterministic guard for risk-related questions, low-confidence answers, and partially-shown records. |
| `low_confidence` | `confidence` was at or below `0.6`. |
| `consult_reason` | Present when the guard fired; says plainly why. |
| `retrieval.strategy` | `full_record` (the whole record was shown) or `planned` (too large — a selected subset was shown, and the context says what was left out). |

Errors: `400` empty question, `502` if the chat call fails.

---

### Multi-turn conversation (Phase 2)

A conversation session tracks turn history **in MongoDB** (so follow-ups
still work after a restart, and across multiple workers) and resolves each
follow-up two ways before answering:

1. **Query rewriting** — *"was that safe?"* becomes a self-contained query.
2. **Entity focus** — the session remembers which medications, lab tests and
   documents the conversation is actually about, matched exactly against
   that patient's own record. This is deterministic, so the subject of a
   follow-up survives even if the rewrite call fails or drops a detail.

Because the whole record is normally in context, a follow-up can compare
across every uploaded document at once — *"has this changed since the
discharge summary?"* is answerable without re-uploading or re-querying
anything.

#### `POST /api/v1/sessions`

Starts a new session for the authenticated user. No request body.

```
curl -X POST http://127.0.0.1:8000/api/v1/sessions \
  -H "Authorization: Bearer $TOKEN" -H "X-User-Id: $USER_ID"
```

Response `201`:

```json
{"user_id": "6620a1f2...", "session_id": "29d7891954a543f1a48f19c9e06c7479"}
```

#### `POST /api/v1/sessions/{session_id}/messages`

Asks one question within an existing session. `404`s if `session_id`
doesn't exist, or belongs to a different user.

Request body:

```json
{
  "question": "Was that safe with my allergy?"
}
```

```
curl -X POST http://127.0.0.1:8000/api/v1/sessions/29d7891954a543f1a48f19c9e06c7479/messages \
  -H "Authorization: Bearer $TOKEN" -H "X-User-Id: $USER_ID" \
  -H "Content-Type: application/json" \
  -d '{"question": "Was that safe with my allergy?"}'
```

Response `200` — same shape as `/qa`, plus `rewritten_query` and `focus`:

```json
{
  "answer": "You have a documented Penicillin allergy, and Amoxicillin is a penicillin-class antibiotic — this is a potential allergy conflict. Please consult your doctor or pharmacist before continuing this medication.",
  "confidence": 0.85,
  "sources": [
    {
      "date": "2026-03-14",
      "source_file": "prescription_march.pdf",
      "document_type": "prescription",
      "document_url": "https://res.cloudinary.com/..."
    },
    {
      "date": "2026-01-08",
      "source_file": "discharge_january.pdf",
      "document_type": "discharge_summary",
      "document_url": "https://res.cloudinary.com/..."
    }
  ],
  "cross_document": true,
  "recommend_professional_consult": true,
  "low_confidence": false,
  "consult_reason": "Please confirm this with a doctor or pharmacist, because the question involves safety, interactions, allergies, or a dosage change.",
  "rewritten_query": "Is Amoxicillin, prescribed to the patient on 2026-03-14, safe given the patient's known drug allergies?",
  "focus": {
    "medications": ["Amoxicillin"],
    "lab_tests": [],
    "source_files": ["prescription_march.pdf", "discharge_january.pdf"]
  }
}
```

`focus` is what the next follow-up will inherit — the drug came from the
prescription, the allergy from the discharge summary, and both stay in scope
for the rest of the conversation.

Errors: `404` unknown `session_id` (create one first via `POST /sessions`),
`400` empty question, `502` if an underlying OpenAI call fails.

#### `GET /api/v1/sessions/{session_id}`

Returns the full, untrimmed transcript of a session (for logging/export) —
never summarized or truncated, regardless of how the session compacts
history internally for prompting.

```
curl -H "Authorization: Bearer $TOKEN" -H "X-User-Id: $USER_ID" \
  http://127.0.0.1:8000/api/v1/sessions/29d7891954a543f1a48f19c9e06c7479
```

Response `200`:

```json
{
  "user_id": "6620a1f2...",
  "session_id": "29d7891954a543f1a48f19c9e06c7479",
  "turns": [
    {"role": "user", "content": "What was I prescribed in March?", "timestamp": "2026-08-03T10:15:00+00:00"},
    {"role": "assistant", "content": "In March you were prescribed Amoxicillin 500mg...", "timestamp": "2026-08-03T10:15:02+00:00"},
    {"role": "user", "content": "Was that safe with my allergy?", "timestamp": "2026-08-03T10:16:10+00:00"},
    {"role": "assistant", "content": "You have a documented Penicillin allergy...", "timestamp": "2026-08-03T10:16:13+00:00"}
  ]
}
```

`404` if `session_id` doesn't exist, or belongs to a different user.

#### `DELETE /api/v1/sessions/{session_id}`

Ends a session, deleting its stored turn history and entity focus.

```
curl -X DELETE -H "Authorization: Bearer $TOKEN" -H "X-User-Id: $USER_ID" \
  http://127.0.0.1:8000/api/v1/sessions/29d7891954a543f1a48f19c9e06c7479
```

`204` on success, `404` if `session_id` doesn't exist, or belongs to a
different user.

---

## Test data

[`generate_lab_test_data.py`](generate_lab_test_data.py) produces
synthetic but schema-valid `lab_report` documents — same shape
`process_document()` returns — without any OCR or OpenAI calls, so you can
exercise `build_patient_timeline()`, `document_filter.py`, and
`lab_trends.py` for free:

```
python generate_lab_test_data.py --patient "jane doe" --visits 4 --out test_data/lab_results_fixture.json
```

```python
import json
from medical_extractor import build_patient_timeline
from document_filter import filter_non_medical_documents
from lab_trends import track_lab_trends

docs = json.load(open("test_data/lab_results_fixture.json"))
kept, rejected = filter_non_medical_documents(docs)
timeline = build_patient_timeline(kept)
trends = track_lab_trends(timeline)
```

Note this bypasses OCR — it feeds directly into the pipeline at the
"already extracted" stage, so it's not something you multipart-upload
through `/documents` (that endpoint only accepts real files).

## Inspecting what a question is answered from

Retrieval is deterministic, so the context a question would be answered from
can be printed exactly. [`inspect_records.py`](inspect_records.py) does that,
read-only:

```
python inspect_records.py "<user_id>"                              # full assembled context
python inspect_records.py "<user_id>" --summary                    # inventory + context size
python inspect_records.py "<user_id>" --question "did my dose change?"
```

`<user_id>` for API-ingested records, or the patient name for records
processed through the `medical_extractor.py` CLI. `--summary` also lists the
entity vocabulary a follow-up can be resolved against, and calls out any
medication whose dose or frequency changed across documents. No OpenAI call
is made unless a `--question` is given *and* the record is too large to fit
the context budget.

## Notes / limitations

- Document storage is split two ways: the original uploaded file lives in
  Cloudinary (`mediscan/<user_id>/...`) and its structured extraction lives
  in MongoDB (`documents`, `patient_snapshots`, `conversation_sessions`
  collections). Both are scoped by the authenticated `user_id` (see
  [`auth.py`](auth.py), [`db.py`](db.py), [`storage.py`](storage.py)) — no
  raw file bytes, OpenAI request/response payloads, or access tokens are
  ever persisted, and nothing patient-identifying is written to local disk
  by the API.
- Answering a question sends that patient's whole record to the model when
  it fits the context budget (`QA_CONTEXT_BUDGET_CHARS`, default 48000
  chars). That is the point — completeness is what makes cross-document
  answers correct — but it does mean per-question token cost grows with
  record size, where a top-k vector search would have stayed flat. Very
  large records fall back to a planned subset; `inspect_records.py
  --summary` shows which regime a given patient is in.
- Entity focus is matched against the patient's own record vocabulary, so a
  follow-up naming a drug the patient has never been prescribed resolves to
  nothing and falls back to the rewritten query alone.
- The CLI entry point in `medical_extractor.py` (`python medical_extractor.py ...`)
  is unauthenticated by design (local dev/testing tool) and still writes to
  local `patient_report_*.json` / `patient_docs_*.json` files — it does not
  touch MongoDB or Cloudinary.
- See [`docs/pipeline.md`](docs/pipeline.md), [`docs/medical_extractor.md`](docs/medical_extractor.md),
  and [`docs/retrieval.md`](docs/retrieval.md) for how extraction, timeline
  building, and retrieval work internally.
