# Medical Records Extraction, Retrieval & Q&A

Turns uploaded medical documents (prescriptions, lab reports, discharge
summaries) into a structured per-patient timeline, cross-checks it for
safety issues, and answers natural-language questions about it — as a
single-shot Q&A call or a real multi-turn conversation. Exposed over HTTP
under `/api/v1/`, scoped per authenticated user.

```
documents --extract--> Cloudinary (file) + MongoDB (structured data)
                |                                        |
                +----------> timeline --cross-check-------+
                                        |
                                        +--------> index (Chroma)
                                        |
                                 question / conversation
                                        |
                                    JSON answer
```

| Module | Responsibility |
|---|---|
| [`medical_extractor.py`](medical_extractor.py) | Extraction, timeline building, cross-checking, on-disk persistence (CLI) |
| [`document_filter.py`](document_filter.py) | Rejects non-medical uploads (post-extraction, no extra API call) |
| [`retrieval.py`](retrieval.py) | Embedding + Chroma indexing, single-shot Q&A (Phase 1) |
| [`conversation.py`](conversation.py) | Multi-turn sessions, query rewriting, safety-aware summarization (Phase 2) |
| [`api.py`](api.py) | HTTP API over all of the above (Phase 3) |
| [`auth.py`](auth.py) | Verifies the `Authorization`/`X-User-Id` headers on every API request (Phase 4) |
| [`db.py`](db.py) | MongoDB persistence for uploaded documents + patient snapshots, scoped per user (Phase 4) |
| [`storage.py`](storage.py) | Uploads original documents to Cloudinary under `mediscan/<user_id>/` (Phase 4) |
| [`inspect_chroma.py`](inspect_chroma.py) | Read-only CLI for browsing what's indexed in `./chroma_db` |

Deeper internals for each module are documented in [`docs/`](docs/).

## Setup

```
pip install openai pdfplumber pymupdf pillow chromadb python-dotenv fastapi "uvicorn[standard]" python-multipart pymongo cloudinary pyjwt
```

Create a `.env` file in the project root (already gitignored — copy
`.env.example` and fill in real values):

```
OPENAI_API_KEY=sk-...
CLOUDINARY_CLOUD_NAME=...
CLOUDINARY_API_KEY=...
CLOUDINARY_API_SECRET=...
MONGODB_URI=mongodb+srv://...
JWT_SECRET=...          # same secret your auth issuer signs tokens with
```

## Running the API

```
python -m uvicorn api:app --reload

```

- Base URL: `http://127.0.0.1:8000`
- Interactive docs (Swagger UI): `http://127.0.0.1:8000/docs`

All application routes are under the `/api/v1/` prefix.

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
  "indexed": true
}
```

If indexing fails (e.g. embeddings API error), `indexed: false` and an
`index_error` field are included instead — the timeline/cross-check are
still returned and saved.

Errors: `400` no files / unsupported extension, `422` extraction failed for
a given file, `422` a file extracted successfully but doesn't look like a
medical document (see below).

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

---

### Single-shot Q&A (Phase 1)

#### `POST /api/v1/qa`

Answers one question grounded in the authenticated user's indexed
timeline. No server-side session — if you want multi-turn context, pass
`chat_history` yourself, or use the conversation endpoints below instead.

Request body:

```json
{
  "question": "What was I prescribed for my sinus infection?",
  "chat_history": [],
  "top_k": 8
}
```

`chat_history` and `top_k` are optional (`chat_history` defaults to none,
`top_k` defaults to `8`).

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
    {"date": "2026-03-14", "source_file": "prescription_march.pdf"}
  ],
  "recommend_professional_consult": false
}
```

Errors: `400` empty question, `502` if the embedding/chat call fails.

---

### Multi-turn conversation (Phase 2)

A conversation session tracks turn history server-side (in-memory) and
rewrites each follow-up into a self-contained search query before
retrieval, so ambiguous questions like *"was that safe?"* retrieve well.

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
  "question": "Was that safe with my allergy?",
  "top_k": 8
}
```

```
curl -X POST http://127.0.0.1:8000/api/v1/sessions/29d7891954a543f1a48f19c9e06c7479/messages \
  -H "Authorization: Bearer $TOKEN" -H "X-User-Id: $USER_ID" \
  -H "Content-Type: application/json" \
  -d '{"question": "Was that safe with my allergy?"}'
```

Response `200` — same shape as `/qa`, plus `rewritten_query`:

```json
{
  "answer": "You have a documented Penicillin allergy, and Amoxicillin is a penicillin-class antibiotic — this is a potential allergy conflict. Please consult your doctor or pharmacist before continuing this medication.",
  "confidence": 0.85,
  "sources": [
    {"date": "2026-03-14", "source_file": "prescription_march.pdf"}
  ],
  "recommend_professional_consult": true,
  "rewritten_query": "Is Amoxicillin, prescribed to the patient on 2026-03-14, safe given the patient's known drug allergies?"
}
```

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

Ends a session, freeing its in-memory turn history.

```
curl -X DELETE -H "Authorization: Bearer $TOKEN" -H "X-User-Id: $USER_ID" \
  http://127.0.0.1:8000/api/v1/sessions/29d7891954a543f1a48f19c9e06c7479
```

`204` on success, `404` if `session_id` doesn't exist, or belongs to a
different user.

---

## Inspecting the vector store

`./chroma_db` holds one Chroma collection per user (chunk text +
embeddings + metadata), keyed by `user_id` via the HTTP API (or by
patient name when run through the CLI). [`inspect_chroma.py`](inspect_chroma.py)
is a read-only CLI for browsing it without writing throwaway scripts — it
never modifies the store or calls OpenAI.

```
python inspect_chroma.py                            # list every collection + chunk count
python inspect_chroma.py "<user_id>"                 # show chunks for one user
python inspect_chroma.py "<user_id>" --limit 20      # show more chunks
python inspect_chroma.py "<user_id>" --type medication   # filter by chunk_type
```

`--type` accepts `medication`, `lab_result`, `clinical_note`, or `allergy`
(see [`docs/retrieval.md`](docs/retrieval.md) for what each chunk_type
contains).

## Notes / limitations

- Sessions are held in-memory per process — restarting the API drops all
  active conversations (turn history isn't lost from disk, since it was
  never persisted there; see `conversation.py`).
- Document storage is split: the original uploaded file lives in
  Cloudinary (`mediscan/<user_id>/...`), its structured extraction lives in
  MongoDB (`documents`, `patient_snapshots` collections), and only its
  embeddings live in the local `./chroma_db`. All three are scoped by the
  authenticated `user_id` (see [`auth.py`](auth.py), [`db.py`](db.py),
  [`storage.py`](storage.py)) — no raw file bytes, OpenAI request/response
  payloads, or access tokens are ever persisted.
- The CLI entry point in `medical_extractor.py` (`python medical_extractor.py ...`)
  is unauthenticated by design (local dev/testing tool) and still writes to
  local `patient_report_*.json` / `patient_docs_*.json` files — it does not
  touch MongoDB or Cloudinary.
- See [`docs/pipeline.md`](docs/pipeline.md), [`docs/medical_extractor.md`](docs/medical_extractor.md),
  and [`docs/retrieval.md`](docs/retrieval.md) for how extraction, timeline
  building, and retrieval work internally.
