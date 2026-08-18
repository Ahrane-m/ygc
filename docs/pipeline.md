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

## 3b. Language guard (`language_guard.py`)

Runs right after extraction, before any expensive downstream work, alongside `document_filter.py`. No extra model call — it reads fields `process_document()` already returned.

- Extraction now reports `document_language` and `additional_languages` (mixed-language documents are common and supported — English drug names with Sinhala dosage instructions, for example).
- `assert_supported_language(doc, filename)` raises `UnsupportedLanguageError` when the document's language could not be **normalized**, not when it is merely unfamiliar. Multi-language support works by converting `ingredients` to the English INN at extraction time; `retrieval._med_group_key()` and `detect_exact_duplicate_medications()` both key on that. If normalization silently fails, the same drug under two languages yields two different group keys — so the duplicate is never spotted and the interaction check never sees both halves, with nothing erroring and the record still *looking* complete. That silent gap is what this converts into a loud, actionable `422`.
- **Only positive evidence rejects.** Two signals, both definitive: an `ingredients` entry that isn't Latin script (an INN always is), or a non-Latin drug name with no ingredient resolved at all. A missing `document_language` on a document whose fields all normalized correctly is *not* an error — the normalization is what matters.
- The API rejects the upload; the CLI reports the file and skips it so a folder run still processes everything else.

### Red flag (graduated, never blocks)

Extraction reports `ocr_confidence` (could we read the page?) and `translation_confidence` (did we convert it into English faithfully?) as **independent** scores — they fail differently and are fixed differently: a bad read needs a clearer scan, a bad conversion needs a pharmacist to confirm the generic names. A crisply printed Japanese prescription is high on the first and lower on the second; a blurry English note is the reverse.

- `assess_translation_risk(doc)` grades one document into `flag`: `"none"` / `"review"` / `"high"`, with a plain-language `message`.
- `assess_documents_translation_risk(docs)` rolls a whole record into one banner, returned as `translation_risk` on the upload response.
- The axes are combined by **multiplying** (`effective_confidence`) — a perfect conversion of a misread word is still wrong. This catches documents where each axis clears its own threshold but the pair doesn't (0.65 × 0.75 = 0.49), a real case found in live testing that showed no flag until it was handled.
- `VISION_OCR_CONFIDENCE_CEILING` caps `ocr_confidence` only. Correctly translating a correctly-read word isn't made less certain by the page having been photographed; that coupling belongs in `effective_confidence`, not in a silent cap.
- Silent on documents reporting no language at all (anything extracted before these fields existed) — "we don't know" is not evidence of translation, and flagging the back catalogue would train users to dismiss the flag.
- A flagged non-English document becomes a `translation_uncertain` referral in `consult_triage.py`, routed to a pharmacist.

Run `python language_guard.py` for the offline self-test (no API key or network needed).

## 3c. Neo4j reference graph logging (`graph_db.py`, `poisoning_kg.py`)

The antidote graph is the one dependency that is remote, optional, and **allowed to fail silently** — `api.py` wraps the lookup in a bare `except Exception` so an unreachable Neo4j never fails a patient's upload. That is correct behaviour, but it means a misconfigured or down graph produces no visible symptom. The logging exists so you can tell those apart.

Every interaction logs at three points — **before** the call, at **each step** within it, and on **completion**:

```
neo4j: [lookup_antidote_references] starting (database=neo4j)
neo4j: connecting to neo4j+s://xxxxx.databases.neo4j.io (database=neo4j)
neo4j: connected to neo4j+s://xxxxx.databases.neo4j.io in 1261ms
neo4j: [lookup_antidote_references] -> match 2 medicine name(s)
neo4j: [lookup_antidote_references] <- match 2 medicine name(s) ok in 121ms (2 row(s) returned)
neo4j: [lookup_antidote_references] completed in 1382ms
lookup: 1 of 2 drug name(s) are listed as antidotes: Naloxone
```

- **Successes are logged as loudly as failures.** Both idempotent outcomes are stated plainly — a fresh load reports `created 5 node(s) and 4 relationship(s)`, a re-ingest reports `created no new nodes or relationships because this document was already in the graph`. Logging only the second would make the normal re-ingest look like a silent failure. A no-match lookup says *"the graph was reached — this is a genuine no-match, not a failure"*.
- **Completion logs report what the server actually did** (`summary.counters`), not what was requested. On a MERGE-based load those are very different numbers.
- **Every started operation ends with a terminal line**, `completed` or `FAILED`. `get_driver()` is called *inside* `session_scope`'s `try` specifically so a connection failure still closes out the operation instead of leaving a dangling "starting".
- **Credentials never reach the log** — `_safe_uri()` strips any embedded `user:password@` before a URI is logged.
- **DEBUG adds the per-item detail**: every parsed table row, each subsection/list marker, each matched listing (confirming e.g. that Naloxone came back from *both* the adult EML and the children's EMLc), and confirmation that the singleton driver is being reused rather than reconnecting per request.
- The driver's own `neo4j.notifications` logger is raised to `WARNING`, because it dumps a multi-screen `GqlStatusObject` on every `CREATE CONSTRAINT IF NOT EXISTS` that finds its constraint already present, burying everything above. Genuine server warnings still come through.

`graph_db.close_driver()` closes and resets the singleton. A failed connection also resets it, so one startup blip doesn't disable the graph for the whole process lifetime.

### Stale connections (a real failure this logging caught)

A live upload logged the lookup starting and then:

```
ERROR neo4j.io: <CONNECTION> error: Failed to read from defunct connection ... 
      ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host')
ERROR graph_db: neo4j: [lookup_antidote_references] FAILED after 103ms: Unable to retrieve routing information
```

The diagnostic detail is what is **absent**: there is no `connecting to...` line, so the driver was not reconnecting — it took a long-idle connection out of the pool and found out only on use that Aura had already closed it. The driver is a process-wide singleton created at startup, so between uploads its connections can idle for hours, and a managed instance (or any load balancer in front of it) drops them well before the driver's 1-hour default lifetime.

Three changes, all in `graph_db.py`:

1. **Connections are retired on our schedule** — `max_connection_lifetime=300s` (well under the default), plus `liveness_check_timeout=30s` so the driver pings anything idle longer than that before handing it out, and `keep_alive=True`. All overridable via `NEO4J_*` env vars (see `.env.example`).
2. **Reads and writes go through managed transactions** (`execute_read` / `execute_write`) instead of auto-commit `session.run()`. Auto-commit does **not** retry; a managed transaction re-acquires a connection and runs again, so a connection dying between the liveness check and the query is recovered rather than surfacing as a failed lookup. `max_transaction_retry_time=15s` bounds this — recovering a dropped connection is worth a few seconds, never worth holding up an upload that succeeds without the enrichment.
3. **Retries are logged.** The driver retries silently, which is right operationally and wrong for reading a trace, so the transaction function counts its own invocations and emits `retrying ... (attempt 2)` plus `ok ... after 2 attempts`. A query that "took 4 seconds" reads very differently once you can see it was the third attempt.

## 4. Consultation triage (`consult_triage.py`)

Runs **after** cross-checking and lab-trend tracking, over the findings they already produced. It answers only the routing question — who should the patient talk to — and adds no clinical judgment of its own.

- `triage_consultation(cross_check, lab_trends, timeline)` — returns `consult_needed`, `consult_type` (`"pharmacist"` / `"doctor"`), `urgency` (`routine` / `soon` / `urgent`), a `confidence` score inherited from the finding that triggered the referral, `recommended_specialties`, and the per-finding `referral_items` split into `pharmacist_actions` / `doctor_actions`.
- **Findings about the documents are separated from findings about the patient.** Every item carries a `category` of `"clinical"` or `"data_quality"`. Only clinical items reach `referral_items` and set `consult_needed` / `consult_type` / `urgency`; document-quality items (a scan too poor to trust, `DATA_QUALITY_TRIGGERS`) go to `document_quality_notices` with a one-line `document_quality_note`. A hard-to-read field is a fact about the paperwork, and routing it as "speak to a pharmacist" trains people to ignore the referrals that mean something. The exception is a low-confidence **translation**: those drug names may not be what the page says while the record looks perfectly normal, so a `"high"` translation flag stays a real pharmacist referral (a `"review"` flag does not).
- A non-empty `illegible_or_low_confidence_fields` is no longer enough on its own. The extractor also files interpretation notes there, so a document quality notice additionally requires either genuinely low `overall_confidence` (≤ 0.6) or an unreadable field that is clinically material (`MATERIAL_FIELD_PATTERN`) on a document below `TRUSTED_EXTRACTION_THRESHOLD` (0.8). See `extraction_notes` in `medical_extractor.py` for the field that now absorbs the rest.
- **Routing is deterministic** — a table keyed by finding type and severity (`ROUTING_RULES`), because who resolves a finding is a scope-of-practice question, not a medical one. A prescription clashing with a documented allergy needs the prescription *changed* (doctor); a therapeutic duplication is medication reconciliation (pharmacist); an out-of-range lab needs *interpreting* (doctor). Every item carries a `why_this_route` string, so the recommendation is auditable line by line.
- **Only specialty selection uses the model.** A rule map (`LAB_SPECIALTY_RULES`) covers the common lab tests without an API call; one LLM pass handles the rest; anything still unresolved defaults to a general practitioner. A failed call costs a specialty name, never a referral.
- Two invariants, both covered by the module's self-test: it **never de-escalates** (`consult_needed: false` means "no trigger found", explicitly not a clean bill of health), and **low confidence never lowers urgency** — an uncertain finding gets a `confidence_caveat` telling the reader to verify it against the original document, at unchanged urgency.
- No `emergency` level exists by design: every finding derives from uploaded documents, which describe the past. A standing `emergency_advice` field covers anything happening now.

Run `python consult_triage.py` for the offline self-test (no API key or network needed — it exercises the routing table directly).

## 5. Structured Retrieval + Q&A (`retrieval.py`, Phase 1)

Sits on top of the **already-extracted** structured timeline — it does not re-read raw documents.

```
patient snapshot (Mongo, or the CLI's local JSON report)
  → render document manifest + allergies + medication rollups
    + lab series w/ trends + clinical notes + safety flags + consult routing
  → (only if over budget) plan which entities matter, re-render narrowed
  → answer strictly from that context, with a post-hoc safety guard
```

**There is no vector store, and this is deliberate** — see `retrieval.py`'s module docstring for the full reasoning. In short: the retrieval unit is one patient's own record (tens to a few hundred entries, not a corpus), and every headline question is a *completeness* question ("what am I taking?", "has my dose changed?") rather than a similarity one. Top-k cosine similarity silently drops exactly the evidence that makes those answers correct, and a dropped medication in a drug-interaction answer is a safety failure, not a relevance miss.

So retrieval is deterministic assembly: load the saved snapshot, render it grouped by document *and* rolled up per entity across documents, and hand the whole thing over. **The common path makes zero extra API calls** — a planner LLM narrows the record only when it exceeds `QA_CONTEXT_BUDGET_CHARS` (default 48000).

### Context assembly

`build_full_context(record)` renders the whole record. `_fit_to_budget()` trims only when it must, and never trims the mandatory sections — the document manifest, the allergy list, the safety cross-check, and the consultation routing. Cutting those turns a space problem into a safety problem: a truncated manifest makes "that isn't in your records" a lie, and a truncated allergy list makes the consult recommendation fire on incomplete grounds.

### Answering

`answer_question(patient_key, question, chat_history=None, retrieval_query=None, record=None, focus=None)` calls the chat model under a system prompt that:

- answers **only** from the assembled context, saying "I don't have enough information" otherwise
- never gives a diagnosis
- writes for a reader with no medical background (spells out abbreviations, "the normal range" not "reference range")
- distinguishes VERIFIED / BACKED BY / UNVERIFIED safety findings, never restating unverified model knowledge as fact
- answers "who should I see?" from the computed consultation routing rather than improvising

`_apply_safety_guard()` is a deterministic backstop that forces `recommend_professional_consult` for risk-related or low-confidence answers, and records *why* it fired — the model is already told to do this, but "already told to" is not a control.

Returns a graceful "no information" answer (no API calls) if the patient has no processed records.

## 6. Wiring (`medical_extractor.py` `__main__`)

```
python medical_extractor.py <file1> <file2> ...      # or a folder path
python medical_extractor.py <path> --chat             # same, then drops into an interactive Q&A loop
```

For each patient found: build timeline → cross-check → lab trends → consultation triage → write `patient_report_<name>.json` (which carries `consult_triage` alongside `cross_check_report` and `lab_trends`). If `--chat` was passed, prompts for a patient (if more than one was processed) and loops on `input()` → `answer_question()` → prints the JSON result, keeping running `chat_history`.

## Dependencies

```
pip install -r requirements.txt
```

Environment variables are listed in [`.env.example`](../.env.example).

## Status / Next steps

- Q&A is grounded in structured, already-extracted fields only.
- Not yet implemented: retrieval over raw document text/images, multi-patient comparison queries, evaluation harness for answer quality, and drug-interaction reference data in the knowledge graph (without it, interaction and allergy findings grade as unverified `model_knowledge` — see `evidence_grading.py`).
