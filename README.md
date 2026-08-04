# Medical Records Extraction, Retrieval & Q&A

Turns uploaded medical documents (prescriptions, lab reports, discharge
summaries) into a structured per-patient timeline, cross-checks it for
safety issues, and answers natural-language questions about it — as a
single-shot Q&A call or a real multi-turn conversation. Exposed over HTTP
under `/api/v1/`.

```
documents --extract--> timeline --cross-check--> safety report
                |                                        |
                +-----------------> index (Chroma) <-----+
                                        |
                                 question / conversation
                                        |
                                    JSON answer
```

| Module | Responsibility |
|---|---|
| [`medical_extractor.py`](medical_extractor.py) | Extraction, timeline building, cross-checking, on-disk persistence |
| [`retrieval.py`](retrieval.py) | Embedding + Chroma indexing, single-shot Q&A (Phase 1) |
| [`conversation.py`](conversation.py) | Multi-turn sessions, query rewriting, safety-aware summarization (Phase 2) |
| [`api.py`](api.py) | HTTP API over all of the above (Phase 3) |
| [`inspect_chroma.py`](inspect_chroma.py) | Read-only CLI for browsing what's indexed in `./chroma_db` |

Deeper internals for each module are documented in [`docs/`](docs/).

## Setup

```
pip install openai pdfplumber pymupdf pillow chromadb python-dotenv fastapi "uvicorn[standard]" python-multipart
```

Create a `.env` file in the project root (already gitignored):

```
OPENAI_API_KEY=sk-...
```

## Running the API

```
python -m uvicorn api:app --reload

```

- Base URL: `http://127.0.0.1:8000`
- Interactive docs (Swagger UI): `http://127.0.0.1:8000/docs`

All application routes are under the `/api/v1/` prefix.

---

## API Reference

`{patient_key}` is a free-text patient name (e.g. `amit sharma`). It's
normalized internally (lowercased/trimmed), so `Amit Sharma` and
`amit sharma` refer to the same patient.

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

#### `POST /api/v1/patients/{patient_key}/documents`

Uploads one or more files (`multipart/form-data`, field name `files`).
Extracts each, **merges with any documents previously uploaded for this
patient**, rebuilds the timeline, re-runs cross-checking, and re-indexes
for Q&A. Supported extensions: `.pdf .png .jpg .jpeg .webp`.

```
curl -X POST http://127.0.0.1:8000/api/v1/patients/amit%20sharma/documents \
  -F "files=@prescription_march.pdf" \
  -F "files=@lab_report_april.jpg"
```

Response `201`:

```json
{
  "patient_key": "amit sharma",
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
        "_source": {"file": "prescription_march.pdf", "method": "text_layer"}
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
a given file.

#### `GET /api/v1/patients/{patient_key}/timeline`

Returns the patient's last saved timeline (same shape as the `timeline`
field above).

```
curl http://127.0.0.1:8000/api/v1/patients/amit%20sharma/timeline
```

`404` if this patient has never been processed.

#### `GET /api/v1/patients/{patient_key}/cross-check`

Returns the patient's last saved cross-check report (same shape as
`cross_check_report` above).

```
curl http://127.0.0.1:8000/api/v1/patients/amit%20sharma/cross-check
```

`404` if this patient has never been processed.

---

### Single-shot Q&A (Phase 1)

#### `POST /api/v1/patients/{patient_key}/qa`

Answers one question grounded in the patient's indexed timeline. No
server-side session — if you want multi-turn context, pass `chat_history`
yourself, or use the conversation endpoints below instead.

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
curl -X POST http://127.0.0.1:8000/api/v1/patients/amit%20sharma/qa \
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

#### `POST /api/v1/patients/{patient_key}/sessions`

Starts a new session. No request body.

```
curl -X POST http://127.0.0.1:8000/api/v1/patients/amit%20sharma/sessions
```

Response `201`:

```json
{"patient_key": "amit sharma", "session_id": "29d7891954a543f1a48f19c9e06c7479"}
```

#### `POST /api/v1/patients/{patient_key}/sessions/{session_id}/messages`

Asks one question within an existing session.

Request body:

```json
{
  "question": "Was that safe with my allergy?",
  "top_k": 8
}
```

```
curl -X POST http://127.0.0.1:8000/api/v1/patients/amit%20sharma/sessions/29d7891954a543f1a48f19c9e06c7479/messages \
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

#### `GET /api/v1/patients/{patient_key}/sessions/{session_id}`

Returns the full, untrimmed transcript of a session (for logging/export) —
never summarized or truncated, regardless of how the session compacts
history internally for prompting.

```
curl http://127.0.0.1:8000/api/v1/patients/amit%20sharma/sessions/29d7891954a543f1a48f19c9e06c7479
```

Response `200`:

```json
{
  "patient_key": "amit sharma",
  "session_id": "29d7891954a543f1a48f19c9e06c7479",
  "turns": [
    {"role": "user", "content": "What was I prescribed in March?", "timestamp": "2026-08-03T10:15:00+00:00"},
    {"role": "assistant", "content": "In March you were prescribed Amoxicillin 500mg...", "timestamp": "2026-08-03T10:15:02+00:00"},
    {"role": "user", "content": "Was that safe with my allergy?", "timestamp": "2026-08-03T10:16:10+00:00"},
    {"role": "assistant", "content": "You have a documented Penicillin allergy...", "timestamp": "2026-08-03T10:16:13+00:00"}
  ]
}
```

`404` if `session_id` doesn't exist.

#### `DELETE /api/v1/patients/{patient_key}/sessions/{session_id}`

Ends a session, freeing its in-memory turn history.

```
curl -X DELETE http://127.0.0.1:8000/api/v1/patients/amit%20sharma/sessions/29d7891954a543f1a48f19c9e06c7479
```

`204` on success, `404` if `session_id` doesn't exist.

---

## Inspecting the vector store

`./chroma_db` holds one Chroma collection per patient (chunk text +
embeddings + metadata). [`inspect_chroma.py`](inspect_chroma.py) is a
read-only CLI for browsing it without writing throwaway scripts — it never
modifies the store or calls OpenAI.

```
python inspect_chroma.py                            # list every patient collection + chunk count
python inspect_chroma.py "amit sharma"               # show chunks for one patient
python inspect_chroma.py "amit sharma" --limit 20    # show more chunks
python inspect_chroma.py "amit sharma" --type medication   # filter by chunk_type
```

`--type` accepts `medication`, `lab_result`, `clinical_note`, or `allergy`
(see [`docs/retrieval.md`](docs/retrieval.md) for what each chunk_type
contains).

## Notes / limitations

- Sessions are held in-memory per process — restarting the API drops all
  active conversations (turn history isn't lost from disk, since it was
  never persisted there; see `conversation.py`).
- `./chroma_db`, `patient_report_*.json`, and `patient_docs_*.json` are
  local, unauthenticated storage — there's no access control between
  patients or callers. Don't expose this API publicly without adding auth.
- See [`docs/pipeline.md`](docs/pipeline.md), [`docs/medical_extractor.md`](docs/medical_extractor.md),
  and [`docs/retrieval.md`](docs/retrieval.md) for how extraction, timeline
  building, and retrieval work internally.
