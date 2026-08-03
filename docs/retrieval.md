# `retrieval.py` reference

Retrieval-augmented Q&A layer (Phase 1). Imports `client` and `MODEL` from [`medical_extractor.py`](../medical_extractor.py) — it does **not** create its own `OpenAI` client, so `OPENAI_API_KEY` only needs to be set once. It operates entirely on the already-extracted structured timeline (the dict `build_patient_timeline()` returns) — it never re-reads a PDF/image.

## Install / env

```
pip install chromadb --break-system-packages
export OPENAI_API_KEY="sk-..."   # same key medical_extractor.py uses
```

Module-level constants:
- `EMBEDDING_MODEL = "text-embedding-3-small"`
- `CHAT_MODEL = MODEL` — currently just aliases `medical_extractor.MODEL` (`"gpt-5-mini"`); change this constant, not `medical_extractor.MODEL`, if the QA model should diverge from the extraction model
- `CHROMA_DIR = "./chroma_db"` — local persistent Chroma store, relative to wherever the process is run from. **Not currently gitignored** — check before committing if you run this locally, it'll create a `chroma_db/` folder with binary index files.
- `EMBEDDING_BATCH_SIZE = 100` — chunking safeguard for the embeddings API's per-request item limit

## Data flow

```
build_patient_timeline() dict
        │
        ▼
build_chunks_from_timeline()  →  [{"id", "text", "metadata"}, ...]
        │
        ▼
embed_texts()  →  one vector per chunk (text-embedding-3-small)
        │
        ▼
index_patient_timeline()  →  Chroma collection.upsert(), one collection per patient, persisted to ./chroma_db
```

```
question
        │
        ▼
answer_question()
  1. embed_texts([question])
  2. collection.query(query_embeddings=..., n_results=top_k)
  3. build a context string from the returned chunks (cited by date + source_file)
  4. chat.completions.create(..., response_format=ANSWER_RESPONSE_FORMAT)
  5. return parsed JSON
```

## Chunking

`build_chunks_from_timeline(patient_key, timeline) -> List[dict]`

Reads exactly the shape `build_patient_timeline()` in `medical_extractor.py` produces: `medications_timeline`, `lab_results_timeline`, `visits` (for `clinical_notes` + `_source.file`), `known_allergies`. **If those keys or their per-entry fields change upstream, this function breaks silently** (missing fields just render as "unknown"/"not specified" in chunk text rather than raising) — worth a periodic sanity check if `medical_extractor.py` changes.

Produces one chunk per:

| `chunk_type` | one per... | text helper |
|---|---|---|
| `medication` | entry in `medications_timeline` | `_medication_chunk_text()` |
| `lab_result` | entry in `lab_results_timeline` | `_lab_result_chunk_text()` |
| `clinical_note` | visit with non-null `clinical_notes` | `_clinical_note_chunk_text()` |
| `allergy` | the whole `known_allergies` list (0 or 1 chunk total) | `_allergy_chunk_text()` |

Each chunk: `{"id": <sha256 hex>, "text": <natural-language string, this is what gets embedded>, "metadata": {"patient_key", "date", "source_file", "chunk_type"}}`. Metadata values are always strings (`""` instead of `None`/missing) because Chroma metadata doesn't accept `None`.

**Chunk IDs are deterministic**, not random: `sha256(f"{patient_key}|{source_file}|{chunk_type}|{index}")`. `index` is the position within that entry's list in the timeline (e.g. the 3rd medication overall gets index `2` regardless of which document it came from). This means:
- Re-running `index_patient_timeline()` on the same timeline re-embeds and `upsert()`s the same IDs — no duplicates, safe to call on every pipeline run.
- If the *order* of `medications_timeline`/`lab_results_timeline` changes between runs (e.g. a new document gets inserted earlier chronologically, shifting indices), some IDs will collide with different content and old chunks won't be cleaned up automatically — there's no deletion/reconciliation logic yet. Not a bug per se, just a known Phase-1 limitation: stale/renamed chunks can accumulate if timeline ordering shifts. Worth knowing before treating `./chroma_db` as fully authoritative.

## Embedding

`embed_texts(texts: List[str]) -> List[List[float]]`

Batches into groups of `EMBEDDING_BATCH_SIZE` and calls `client.embeddings.create()`. Empty input returns `[]` without an API call. Raises `RuntimeError` (not the raw `OpenAIError`) on failure, with the batch size in the message — used both for indexing (chunk texts) and for embedding the incoming question in `answer_question()`.

## Storage

- `_sanitize_collection_name(patient_key) -> str` — Chroma collection names must be 3-63 chars, `[a-zA-Z0-9._-]` only, start/end alphanumeric. This lowercases, replaces disallowed chars with `_`, and pads/prefixes as needed so any `patient_key` (e.g. `"amit sharma"` → `"amit_sharma"`) maps to a valid, **stable** name — same patient_key always produces the same collection name, which is what makes `create=False` lookups in `answer_question()` work.
- `_get_chroma_client()` — `chromadb.PersistentClient(path=CHROMA_DIR)`. Called fresh each time rather than cached at module level — fine for CLI/script usage, but if this gets used inside a long-lived server process, consider caching the client instead of reopening it per call.
- `_get_patient_collection(patient_key, create)` — `create=True` does `get_or_create_collection()` (used by indexing); `create=False` does `get_collection()` and returns `None` on any failure (used by querying — a patient who was never indexed simply isn't found, rather than raising).

`index_patient_timeline(patient_key, timeline) -> None` — the indexing entry point. Chunks → embeds → `collection.upsert()`. Raises `ValueError` for an empty/missing `patient_key`. If chunking produces zero chunks (empty timeline), logs and returns without touching Chroma or the embeddings API.

## Q&A

`answer_question(patient_key, question, chat_history=None, top_k=8) -> dict`

- Raises `ValueError` if `patient_key` or `question` is empty.
- If the patient's collection doesn't exist or is empty, returns `_NO_INFO_ANSWER` immediately — **no embedding or chat API call is made**, so asking about an un-indexed patient is free and instant, not just handled gracefully.
- Otherwise: embeds the question, queries `min(top_k, collection.count())` nearest chunks (so a patient with 3 chunks won't error asking for 8), formats each retrieved chunk into a `[date: ... | source_file: ... | type: ...]` tagged block, and sends that plus optional `chat_history` (a list of `{"role", "content"}` dicts, passed straight through as prior turns) plus the question to the chat model.
- `QA_SYSTEM_PROMPT` enforces: answer only from retrieved context ("I don't have enough information" otherwise), never diagnose, force `recommend_professional_consult: true` for anything about risk/interactions/dosage changes, and cite `sources`.
- Response is constrained by `ANSWER_RESPONSE_FORMAT` (OpenAI Structured Outputs, `strict: True`) to exactly `{"answer": str, "confidence": number, "sources": [{"date": str, "source_file": str}], "recommend_professional_consult": bool}` — same pattern as `EXTRACTION_RESPONSE_FORMAT` in `medical_extractor.py`.
- Raises `RuntimeError` if the chat call itself fails (auth/rate-limit/network) — embedding failures inside this call also surface as `RuntimeError` via `embed_texts()`.

## Known Phase-1 limitations (be aware before extending)

- No deletion/reconciliation of stale chunk IDs if a patient's document set changes in a way that shifts list ordering (see chunking section above).
- No access control on `./chroma_db` — anyone with filesystem access to that directory can read any patient's indexed chunk text directly (it's stored as plaintext `documents` alongside the vectors).
- Retrieval is over structured fields only — raw document text/pages are not indexed, so questions about things not captured by the extraction schema (formatting, marginal handwritten notes, etc.) will correctly come back as "I don't have enough information."
- `chat_history` is trusted as-is and inserted directly into the message list — if this is ever exposed to an untrusted caller (e.g. a web API), validate/sanitize `chat_history` contents before passing them through, since they sit alongside the system prompt in the same conversation.

## Quick usage

```python
from medical_extractor import build_patient_timeline
from retrieval import index_patient_timeline, answer_question

timeline = build_patient_timeline(docs)          # docs = one patient's extracted documents
index_patient_timeline("amit sharma", timeline)  # embeds + upserts into ./chroma_db

result = answer_question("amit sharma", "What was I prescribed for headaches?")
# {"answer": "...", "confidence": 0.85, "sources": [...], "recommend_professional_consult": False}
```

Also wired into `medical_extractor.py`'s `__main__` — see [medical_extractor.md](medical_extractor.md#7-main--cli-flow) for the `--chat` CLI flow.
