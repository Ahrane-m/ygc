# MediScan Records Console — web app

**Live link:** https://claude.ai/code/artifact/5229f6a5-f4ae-4d3f-8a9e-9a9e9296255d

An interactive console for this repo's API. Pick an endpoint in the left rail and it
shows what that endpoint does, the request, and the response it returns — rendered as
a UI rather than raw JSON, with the raw shape behind a toggle.

The page is private by default; share it from the page's share menu if you want others
to see it.

## What it runs on

The page is **static**, with captured responses embedded in it — not live calls to a
running server. Two reasons, both worth knowing before you demo it:

1. The artifact sandbox blocks all outbound network requests (strict CSP), so a hosted
   page cannot call any external host.
2. The API runs on `127.0.0.1:8000`, which nothing outside your machine can reach anyway.

Every response shown is a real one captured from this pipeline, using the synthetic test
corpus (patient "RAMESH") — not a real patient. That is labelled on the page itself so
it can't be mistaken for a real record if the link gets passed on.

To drive the API for real, run it locally and use the `curl` commands shown on each panel:

```bash
python -m uvicorn api:app --reload
# Swagger UI: http://127.0.0.1:8000/docs
```

## Which APIs are used, and how

Four external services. Everything not listed here is deterministic Python — no model
call, no network.

| Service | Used for | How |
|---|---|---|
| **OpenAI** (`gpt-5-mini`) | Extraction, cross-checking, Q&A, specialty naming | Chat Completions with **Structured Outputs** (`strict: true` JSON schema), so every field is always present and parseable. Vision for scans and photos; plain text for PDFs with a text layer. |
| **MongoDB Atlas** | Persistence | `pymongo`. Three collections: `documents`, `patient_snapshots` (one per user), `conversation_sessions`. Every read and write is scoped by `user_id`. |
| **Cloudinary** | Original-file archive | `cloudinary.uploader.upload` into `mediscan/<user_id>/`, so every extracted fact links back to its source document. No file bytes in MongoDB. |
| **Neo4j Aura** | WHO antidote reference graph | Bolt driver over `neo4j+s://`. Shared reference data, not per-patient. Fail-open — unreachable means findings grade as unverified, never a failed upload. |

### Where each OpenAI call happens

| Call | Module | Notes |
|---|---|---|
| Document extraction | `medical_extractor.py` | One call per file (or per page for scanned PDFs). Fixed JSON schema. |
| Safety cross-check | `medical_extractor.py` | One call per upload over the whole medication timeline. |
| Specialty naming | `consult_triage.py` | Only for doctor-routed findings a rule map can't resolve. Falls back to a GP if it fails. |
| Query rewrite | `conversation.py` | Follow-up turns only. Falls back to the raw question on failure. |
| Answer generation | `retrieval.py` | One call per question. |
| Context planner | `retrieval.py` | **Only** when a record exceeds `QA_CONTEXT_BUDGET_CHARS`. The common path makes zero extra calls. |

Deterministic — no API call at all: lab trend tracking, consult routing, duplicate
detection, document de-duplication, the language guard, the identity guard, the
non-medical filter, and evidence grading.

### Notably *not* used

**No vector database, and no embeddings API.** Retrieval is deterministic assembly of the
patient's saved snapshot. The record for one patient fits in a context window whole, and
every question this product answers is a *completeness* question ("what am I taking?",
"did my dose change?") rather than a similarity one — top-k similarity silently drops
exactly the evidence that makes those answers correct, and a dropped medication in a
drug-interaction answer is a safety failure, not a relevance miss. See `retrieval.py`'s
module docstring for the full reasoning.

## Endpoints the console covers

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/documents` | Upload, extract, validate, archive, merge, re-analyse |
| `GET /api/v1/timeline` | Merged chronological record with cross-document rollups |
| `GET /api/v1/cross-check` | Interactions, duplicates, dosage and allergy conflicts — evidence-graded |
| `GET /api/v1/lab-trends` | Per-test movement, range crossings, plain-language explanation |
| `GET /api/v1/consult-triage` | Pharmacist or doctor, how urgently, which specialty |
| `POST /api/v1/qa` | One grounded question with cited sources |
| `POST /api/v1/sessions/{id}/messages` | Multi-turn conversation with pronoun resolution |
| `POST /api/v1/knowledge-graph/antidotes` | Ingest a WHO EML PDF into the reference graph |
| `GET /api/v1/health` | Unauthenticated liveness check |

Auth: every route except `/health` needs `Authorization: Bearer <jwt>` (HS256, verified
locally) plus `X-User-Id`, which must match the token's user-id claim.

## Updating the page

The source is `mediscan_console.html`. Re-publishing the same file keeps the same URL.

## Caveats

- **Not a medical device.** The pipeline extracts and cross-references documents; it does
  not diagnose, and every output defers to a licensed clinician.
- Interaction and allergy findings currently grade as unverified `model_knowledge` — the
  reference graph holds only the WHO antidote list, with no interaction data in it yet.
- Sample data throughout. No real patient records are shown.
