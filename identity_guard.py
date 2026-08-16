"""
Different-User Document Detection
=========================================
This app is single-patient-per-account (see auth.py) — every document
uploaded to an account is assumed to belong to that account's one patient.
This module catches the case where that assumption breaks: a user uploads
a document that actually belongs to someone else (e.g. a family member's
prescription uploaded to the wrong account by mistake).

Comparison is against this account's DOCUMENT HISTORY ONLY — the extracted
name/age/gender on documents already stored for this user. It deliberately
does NOT compare against the account holder's registered profile name: that
name is set at signup and is frequently not the patient's own name (a
caregiver's account, a nickname, a transliteration choice), so treating it
as ground truth produced false positives. A brand new account with no
document history yet has nothing to compare against, so:

  - A first-ever upload where every document agrees on one patient is never
    second-guessed, no matter whose name that is.
  - A first-ever upload whose documents themselves DISAGREE (e.g. one file
    says "Ramesh", another says "Suresh") still needs a decision: the
    largest name-group in the batch is treated as the baseline (ties broken
    by whichever name appeared first), and the rest are held for
    confirmation — the same treatment a later mismatch against existing
    history gets.

PARTIAL ACCEPTANCE: a batch is not all-or-nothing. Documents that match the
known/baseline identity proceed immediately; only the documents that don't
are held back pending confirmation — see partition_identity_mismatch(). A
later request resubmitting just the held file(s) with confirm=true adds
them without re-litigating the ones that already went through.

Deliberately tolerant of two real-world sources of noise, called out
explicitly by design:
  - Name spelling varies across documents (OCR errors, transliteration,
    honorifics) — comparison uses fuzzy string similarity, not exact match.
  - Age changes over time — a patient's age differs between documents taken
    years apart, so raw age is never compared directly. Instead each
    (age, document_date) pair is converted to an estimated birth year, and
    birth years are compared with tolerance for rounding/estimation noise.

To avoid false positives from one noisy field (a single garbled OCR name,
a missing gender), the mismatch decision requires CORROBORATION: either one
strong signal alone, or two weaker signals together — see
MISMATCH_SCORE_THRESHOLD. No extra model calls: everything here runs on
fields medical_extractor.py already extracted.
"""

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple

from dateutil import parser as date_parser

# ---------------------------------------------------------------------------
# Name normalization + fuzzy similarity
# ---------------------------------------------------------------------------

_TITLE_PREFIX_RE = re.compile(
    r"^(mr|mrs|ms|miss|dr|prof|master|shri|smt)\.?\s+", re.IGNORECASE
)
_PUNCTUATION_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_name(name: Optional[str]) -> Optional[str]:
    """Lowercase, strip a leading honorific, strip punctuation, collapse
    whitespace. Returns None for missing/blank input (never ""). The
    honorific list is English/South-Asian only — same documented,
    non-exhaustive limitation as DEMO_PLACEHOLDER_MARKERS in
    medical_extractor.py."""
    if not name or not name.strip():
        return None
    cleaned = _TITLE_PREFIX_RE.sub("", name.strip())
    cleaned = _PUNCTUATION_RE.sub(" ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip().lower()
    return cleaned or None


def _token_sorted(name: str) -> str:
    """Joins the name's whitespace-separated tokens in sorted order, so
    'Sharma Amit' and 'Amit Sharma' compare as the same name."""
    return " ".join(sorted(name.split()))


def name_similarity(a: Optional[str], b: Optional[str]) -> Optional[float]:
    """difflib.SequenceMatcher ratio between two names, taking the MAX of
    the direct-order ratio and the token-sorted-order ratio (so word order
    doesn't cause a false mismatch). Returns None if either input
    normalizes to nothing — a missing name is never treated as a mismatch
    signal on its own."""
    na, nb = normalize_name(a), normalize_name(b)
    if na is None or nb is None:
        return None
    direct = SequenceMatcher(None, na, nb).ratio()
    sorted_ratio = SequenceMatcher(None, _token_sorted(na), _token_sorted(nb)).ratio()
    return max(direct, sorted_ratio)


# ---------------------------------------------------------------------------
# Birth-year inference (age drifts over time; birth year doesn't)
# ---------------------------------------------------------------------------

def infer_birth_year(age: Optional[int], document_date: Optional[str]) -> Optional[int]:
    """birth_year ~= year(document_date) - age. None if age or
    document_date is missing, age is outside a plausible human range, or
    document_date doesn't parse. Uses dateutil.parser rather than assuming
    strict ISO YYYY-MM-DD, since nothing in EXTRACTION_SCHEMA_PROMPT
    enforces that format."""
    if age is None or not isinstance(age, (int, float)):
        return None
    if not (0 <= age <= 130):
        return None
    if not document_date:
        return None
    try:
        parsed = date_parser.parse(str(document_date), fuzzy=True)
    except (ValueError, OverflowError):
        return None
    return parsed.year - int(age)


# ---------------------------------------------------------------------------
# Tunable thresholds
# ---------------------------------------------------------------------------

# At/above this similarity, two names are treated as the same person.
# Calibrated against difflib.SequenceMatcher's actual behavior on short
# name strings (measured, not guessed): a single OCR character swap or
# transliteration variant of the SAME name typically scores 0.85-0.96
# ("Amit Sharma"/"Amit Sharrna" -> 0.87, "Mohammed Ali"/"Mohamed Ali" ->
# 0.96), while two DIFFERENT people who happen to share one name token
# (same surname or same first name) typically score 0.70-0.76 ("Amit
# Sharma"/"Amit Verma" -> 0.76, "John Doe"/"Jane Doe" -> 0.75) — there's a
# clear gap around 0.80-0.84 between those two populations.
NAME_SIMILARITY_MATCH_THRESHOLD = 0.82

# Below this, the name gap is a STRONG signal on its own. Two people
# sharing no name token at all score well below this (measured: 0.26-0.55,
# e.g. "Susan Miller"/"John Smith" -> 0.36). Two different people sharing
# one token (0.70-0.76, see above) land in the WEAK band between this and
# NAME_SIMILARITY_MATCH_THRESHOLD — ambiguous enough to need corroboration
# from age/gender rather than triggering alone.
NAME_MISMATCH_THRESHOLD = 0.65

# Birth-year estimates within this many years of each other are treated as
# the same person — absorbs rounding error (a stated "45" could mean
# anywhere in a ~1-year band depending on whether the birthday has passed
# this year) plus manual-entry rounding across two independently
# transcribed documents.
BIRTH_YEAR_TOLERANCE_YEARS = 2

# A birth-year gap this large isn't plausible rounding/estimation noise —
# STRONG signal on its own.
BIRTH_YEAR_MISMATCH_STRONG_YEARS = 5

# There's no per-field confidence for patient_age/patient_gender (only one
# overall_confidence per document) — a whole document below this bar
# contributes no age/gender signal to either side of the comparison, so a
# garbled vision-OCR read can't poison the baseline. Name is exempt from
# this gate since fuzzy matching already absorbs its noise.
MIN_CONFIDENCE_FOR_IDENTITY_SIGNAL = 0.4

# Score contribution per signal. Gender is capped at "weak" always — it's a
# low-cardinality field prone to OCR/honorific misreads and frequently
# absent, so it must never trigger a mismatch by itself.
SIGNAL_WEIGHTS = {
    "name_strong": 2,
    "name_weak": 1,
    "birth_year_strong": 2,
    "birth_year_weak": 1,
    "gender": 1,
}

# Require either ONE strong signal alone (score 2), or TWO corroborating
# weak signals together (score 1+1). A single weak signal (e.g. one
# borderline name variant) never triggers a mismatch by itself.
MISMATCH_SCORE_THRESHOLD = 2


# ---------------------------------------------------------------------------
# Identity aggregation
# ---------------------------------------------------------------------------

def _confident_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Docs whose overall_confidence clears MIN_CONFIDENCE_FOR_IDENTITY_SIGNAL
    (or has no confidence recorded at all, e.g. a doc predating this
    field) — used to gate age/gender signal contribution only."""
    kept = []
    for d in docs:
        confidence = d.get("overall_confidence")
        if confidence is None or not isinstance(confidence, (int, float)):
            kept.append(d)
        elif confidence >= MIN_CONFIDENCE_FOR_IDENTITY_SIGNAL:
            kept.append(d)
    return kept


def _distinct_names(docs: List[Dict[str, Any]]) -> List[str]:
    seen: Dict[str, str] = {}
    for d in docs:
        raw = d.get("patient_name")
        norm = normalize_name(raw)
        if norm and norm not in seen:
            seen[norm] = raw
    return list(seen.values())


def _distinct_birth_years(docs: List[Dict[str, Any]]) -> List[int]:
    years = set()
    for d in _confident_docs(docs):
        year = infer_birth_year(d.get("patient_age"), d.get("date"))
        if year is not None:
            years.add(year)
    return sorted(years)


def _distinct_genders(docs: List[Dict[str, Any]]) -> List[str]:
    genders = set()
    for d in _confident_docs(docs):
        gender = d.get("patient_gender")
        if gender in ("male", "female", "other"):
            genders.add(gender)
    return sorted(genders)


def _identity_from_docs(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregates the distinct name/birth-year/gender signals found across
    `docs` — used both for "this account's known identity" (from stored
    documents) and, on a first-ever upload with no stored history, for
    "this batch's baseline identity" (from the majority name-cluster within
    the batch itself, i.e. docs are that cluster's own pages)."""
    return {
        "document_patient_names": _distinct_names(docs),
        "estimated_birth_years": _distinct_birth_years(docs),
        "genders": _distinct_genders(docs),
    }


def _is_empty_identity(identity: Dict[str, Any]) -> bool:
    return not identity["document_patient_names"] and not identity["estimated_birth_years"] \
        and not identity["genders"]


def _cluster_new_pages(
    labeled_new_pages: List[Tuple[str, Dict[str, Any]]],
) -> Dict[Optional[str], List[Tuple[str, Dict[str, Any]]]]:
    """Groups this upload batch's (label, page) pairs by normalized
    patient_name, preserving first-seen order (relied on by the no-history
    baseline tie-break below). Exact normalized-string clustering is enough
    within a single batch (spelling drift across time is what the fuzzy
    comparison against known_docs is for) — a batch mixing pages for two
    different people must not have the minority page's identity drowned
    out by a majority vote."""
    clusters: Dict[Optional[str], List[Tuple[str, Dict[str, Any]]]] = {}
    for label, page in labeled_new_pages:
        key = normalize_name(page.get("patient_name"))
        clusters.setdefault(key, []).append((label, page))
    return clusters


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _best_name_match(candidate: Optional[str], known_names: List[str]) -> Optional[Dict[str, Any]]:
    """The best (highest-similarity) match between `candidate` and any name
    in `known_names`. None if `candidate` or `known_names` is empty."""
    if not candidate or not known_names:
        return None
    best = None
    for known in known_names:
        sim = name_similarity(candidate, known)
        if sim is None:
            continue
        if best is None or sim > best["similarity"]:
            best = {"known_value": known, "similarity": sim}
    return best


def _score_cluster(
    labeled_cluster_pages: List[Tuple[str, Dict[str, Any]]],
    baseline: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Scores one name-cluster from the new batch against `baseline` (the
    known/baseline identity — see _identity_from_docs). Returns None if
    nothing about this cluster looks mismatched; otherwise a dict
    describing the held document(s) (see module docstring for the shape)."""
    signals: List[Dict[str, Any]] = []
    score = 0
    cluster_pages = [page for _, page in labeled_cluster_pages]

    # -- Name --
    raw_name = next((p.get("patient_name") for p in cluster_pages if p.get("patient_name")), None)
    match = _best_name_match(raw_name, baseline["document_patient_names"])
    if match is not None:
        similarity = match["similarity"]
        if similarity < NAME_MISMATCH_THRESHOLD:
            severity, weight = "strong", SIGNAL_WEIGHTS["name_strong"]
        elif similarity < NAME_SIMILARITY_MATCH_THRESHOLD:
            severity, weight = "weak", SIGNAL_WEIGHTS["name_weak"]
        else:
            severity, weight = None, 0
        if severity:
            score += weight
            signals.append({
                "field": "name",
                "extracted_value": raw_name,
                "known_value": match["known_value"],
                "similarity": round(similarity, 2),
                "severity": severity,
                "explanation": (
                    f"\"{raw_name}\" is only {similarity:.0%} similar to the patient "
                    "name on your other document(s)."
                ),
            })

    # -- Birth year (derived from age + document date) --
    confident_pages = _confident_docs(cluster_pages)
    cluster_years = sorted({
        y for p in confident_pages
        if (y := infer_birth_year(p.get("patient_age"), p.get("date"))) is not None
    })
    if cluster_years and baseline["estimated_birth_years"]:
        # Smallest gap between any new-batch estimate and any known estimate.
        best_gap = min(
            abs(new_y - known_y)
            for new_y in cluster_years for known_y in baseline["estimated_birth_years"]
        )
        if best_gap > BIRTH_YEAR_TOLERANCE_YEARS:
            if best_gap >= BIRTH_YEAR_MISMATCH_STRONG_YEARS:
                severity, weight = "strong", SIGNAL_WEIGHTS["birth_year_strong"]
            else:
                severity, weight = "weak", SIGNAL_WEIGHTS["birth_year_weak"]
            score += weight
            signals.append({
                "field": "age",
                "extracted_value": f"estimated birth year {cluster_years[0]}",
                "known_value": f"estimated birth year {baseline['estimated_birth_years'][0]}",
                "similarity": None,
                "severity": severity,
                "explanation": (
                    f"This document implies a birth year around {cluster_years[0]}, "
                    f"about {best_gap} year(s) off from your other document(s) "
                    f"(around {baseline['estimated_birth_years'][0]})."
                ),
            })

    # -- Gender --
    cluster_genders = _distinct_genders(confident_pages)
    if cluster_genders and baseline["genders"]:
        if set(cluster_genders).isdisjoint(baseline["genders"]):
            score += SIGNAL_WEIGHTS["gender"]
            signals.append({
                "field": "gender",
                "extracted_value": cluster_genders[0],
                "known_value": baseline["genders"][0],
                "similarity": None,
                "severity": "weak",
                "explanation": (
                    f"This document indicates gender '{cluster_genders[0]}', which "
                    f"doesn't match your other document(s) ('{baseline['genders'][0]}')."
                ),
            })

    if score < MISMATCH_SCORE_THRESHOLD:
        return None

    display_name = raw_name or "this document"
    return {
        "patient_name": raw_name,
        "estimated_birth_year": cluster_years[0] if cluster_years else None,
        "gender": cluster_genders[0] if cluster_genders else None,
        "source_files": sorted({label for label, _ in labeled_cluster_pages}),
        "message": (
            f"\"{display_name}\" doesn't match the patient on your other document(s)."
            if raw_name else
            "This document's patient details don't match your other document(s)."
        ),
        "signals": signals,
        "score": score,
        "threshold": MISMATCH_SCORE_THRESHOLD,
    }


def _known_doc_label(doc: Dict[str, Any]) -> str:
    return doc.get("_source", {}).get("file") or "a previously uploaded document"


def partition_identity_mismatch(
    labeled_new_pages: List[Tuple[str, Dict[str, Any]]],
    known_docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Splits `labeled_new_pages` (every kept (label, page) pair from every
    file in one upload batch) into documents that are safe to merge now
    ("proceed") and documents that need explicit user confirmation first
    ("held"), rather than failing the whole batch over one mismatched file.

    Comparison baseline:
      - If `known_docs` (this user's previously stored documents) is
        non-empty, the baseline identity is derived from it — name (fuzzy
        matched), estimated birth year, gender.
      - If `known_docs` is empty (this account's very first upload), there
        is no history to compare against, so the baseline is instead the
        LARGEST name-cluster within `labeled_new_pages` itself (ties broken
        by whichever distinct name appeared first in file order). That
        cluster proceeds unconditionally; other clusters in the same batch
        are held — e.g. uploading one "Ramesh" document and one "Suresh"
        document with no prior history holds the Suresh one back rather
        than silently merging two different patients' data.

    Returns:
      {
        "proceed_labels": set of labels safe to merge now,
        "held_labels": set of labels that need confirmation,
        "details": None if held_labels is empty, else a dict with
          "error", "message", "known_identity", "held_documents" (one
          entry per held name-cluster, each carrying its own signals/
          score), and "documents" (every label + patient_name in this
          batch plus known_docs, for transparency).
      }
    Pure / no side effects.
    """
    if not labeled_new_pages:
        return {"proceed_labels": set(), "held_labels": set(), "details": None}

    clusters = _cluster_new_pages(labeled_new_pages)

    known_identity = _identity_from_docs(known_docs)
    if not _is_empty_identity(known_identity):
        baseline = known_identity
        baseline_cluster_key = None  # baseline comes from history, not from any one batch cluster
    else:
        # No document history at all: pick the largest cluster in this
        # batch as the baseline; ties go to whichever name appeared first.
        cluster_keys_in_order: List[Optional[str]] = []
        for _, page in labeled_new_pages:
            key = normalize_name(page.get("patient_name"))
            if key not in cluster_keys_in_order:
                cluster_keys_in_order.append(key)
        baseline_cluster_key = max(
            cluster_keys_in_order,
            key=lambda k: (len(clusters[k]), -cluster_keys_in_order.index(k)),
        )
        baseline = _identity_from_docs([page for _, page in clusters[baseline_cluster_key]])

    proceed_labels: Set[str] = set()
    held_labels: Set[str] = set()
    held_documents: List[Dict[str, Any]] = []
    for cluster_key, cluster_pages in clusters.items():
        if cluster_key == baseline_cluster_key:
            # This IS the baseline cluster (no-history case) — it always
            # proceeds without needing to be scored against itself.
            proceed_labels.update(label for label, _ in cluster_pages)
            continue
        result = _score_cluster(cluster_pages, baseline)
        if result is None:
            proceed_labels.update(label for label, _ in cluster_pages)
        else:
            held_labels.update(label for label, _ in cluster_pages)
            held_documents.append(result)

    if not held_documents:
        return {"proceed_labels": proceed_labels, "held_labels": set(), "details": None}

    names = sorted({d["patient_name"] for d in held_documents if d["patient_name"]})
    details = {
        "error": "patient_name_mismatch",
        "message": (
            f"{len(held_documents)} of {len(clusters)} uploaded document group(s) "
            f"({', '.join(names)}) don't match the patient on your other document(s) "
            "and were not added. Confirm to add them anyway, or leave them out."
            if names else
            f"{len(held_documents)} of {len(clusters)} uploaded document group(s) don't "
            "match the patient on your other document(s) and were not added. Confirm "
            "to add them anyway, or leave them out."
        ),
        "known_identity": {
            "document_patient_names": baseline["document_patient_names"],
            "estimated_birth_year": baseline["estimated_birth_years"][0] if baseline["estimated_birth_years"] else None,
            "gender": baseline["genders"][0] if baseline["genders"] else None,
        },
        "held_documents": held_documents,
        "documents": [
            {"label": label, "patient_name": page.get("patient_name")}
            for label, page in labeled_new_pages
        ] + [
            {"label": _known_doc_label(doc), "patient_name": doc.get("patient_name")}
            for doc in known_docs
        ],
    }
    return {"proceed_labels": proceed_labels, "held_labels": held_labels, "details": details}


if __name__ == "__main__":
    # Lightweight self-test, no pytest dependency needed.

    def _doc(name, age=None, gender=None, date="2024-01-01", confidence=0.9, file="doc.pdf"):
        return {
            "patient_name": name, "patient_age": age, "patient_gender": gender,
            "date": date, "overall_confidence": confidence,
            "_source": {"file": file},
        }

    def _labeled(*docs):
        return [(d.get("_source", {}).get("file", "doc.pdf"), d) for d in docs]

    # 1. First-ever upload, single name -> proceeds, nothing held, no
    #    comparison against any "account name" (there is none passed in at
    #    all — this module never sees one anymore).
    result = partition_identity_mismatch(
        _labeled(_doc("Whoever Uploaded This", age=40, gender="male")), [],
    )
    assert result["details"] is None
    assert len(result["proceed_labels"]) == 1 and not result["held_labels"]

    # 2. First-ever upload, multiple documents, all the same name -> all proceed.
    result = partition_identity_mismatch(
        _labeled(
            _doc("Ramesh Kumar", age=50, file="a.pdf"),
            _doc("Ramesh Kumar", age=50, file="b.pdf"),
        ), [],
    )
    assert result["details"] is None
    assert result["proceed_labels"] == {"a.pdf", "b.pdf"}

    # 3. First-ever upload, 3 Ramesh docs + 1 Suresh doc -> majority
    #    (Ramesh) proceeds, minority (Suresh) held for confirmation.
    result = partition_identity_mismatch(
        _labeled(
            _doc("Ramesh Kumar", age=50, gender="male", file="r1.pdf"),
            _doc("Ramesh Kumar", age=50, gender="male", file="r2.pdf"),
            _doc("Ramesh Kumar", age=50, gender="male", file="r3.pdf"),
            _doc("Suresh Babu", age=30, gender="male", file="s1.pdf"),
        ), [],
    )
    assert result["proceed_labels"] == {"r1.pdf", "r2.pdf", "r3.pdf"}
    assert result["held_labels"] == {"s1.pdf"}
    assert result["details"] is not None
    assert result["details"]["held_documents"][0]["patient_name"] == "Suresh Babu"

    # 4. First-ever upload, ONE Ramesh doc + ONE Suresh doc (tie) -> the
    #    first-encountered name (Ramesh, uploaded first) is the baseline
    #    and proceeds; Suresh is held.
    result = partition_identity_mismatch(
        _labeled(
            _doc("Ramesh Kumar", age=50, gender="male", file="ramesh.pdf"),
            _doc("Suresh Babu", age=30, gender="male", file="suresh.pdf"),
        ), [],
    )
    assert result["proceed_labels"] == {"ramesh.pdf"}
    assert result["held_labels"] == {"suresh.pdf"}

    # 5. Existing history = Ramesh. New batch = Ramesh (spelling variant,
    #    OCR-plausible) + Suresh -> Ramesh proceeds (fuzzy match against
    #    history), Suresh held.
    known_docs = [_doc("Ramesh Kumar", age=50, gender="male", date="2020-01-01", file="old.pdf")]
    result = partition_identity_mismatch(
        _labeled(
            _doc("Ramesh Kumarr", age=56, gender="male", date="2026-01-01", file="new_ramesh.pdf"),
            _doc("Suresh Babu", age=30, gender="male", date="2026-01-01", file="new_suresh.pdf"),
        ), known_docs,
    )
    assert result["proceed_labels"] == {"new_ramesh.pdf"}
    assert result["held_labels"] == {"new_suresh.pdf"}
    assert result["details"]["known_identity"]["document_patient_names"] == ["Ramesh Kumar"]

    # 6. Existing history = Ramesh. New batch = Suresh only -> held, nothing
    #    proceeds from this batch (existing Ramesh docs are untouched,
    #    that's the caller's concern, not this function's).
    result = partition_identity_mismatch(
        _labeled(_doc("Suresh Babu", age=30, gender="male", file="s.pdf")), known_docs,
    )
    assert not result["proceed_labels"]
    assert result["held_labels"] == {"s.pdf"}

    # 7. Missing fields everywhere -> never held, nothing to compare.
    result = partition_identity_mismatch(_labeled(_doc(None, age=None, gender=None)), [])
    assert result["details"] is None

    # 8. Corroboration boundary: one weak signal alone does not hold a
    #    document back; two weak signals together do.
    known_docs_boundary = [_doc("John Doe", age=50, gender="male", date="2020-01-01")]
    weak_name_only = _labeled(_doc("Jane Doe", age=56, gender="male", date="2026-01-01", file="w.pdf"))
    sim = name_similarity("John Doe", "Jane Doe")
    assert NAME_MISMATCH_THRESHOLD <= sim < NAME_SIMILARITY_MATCH_THRESHOLD, (
        f"test fixture assumption broken: similarity={sim}"
    )
    result = partition_identity_mismatch(weak_name_only, known_docs_boundary)
    assert result["details"] is None, f"a single weak signal must not hold a document, got {result}"
    weak_name_and_gender = _labeled(_doc("Jane Doe", age=56, gender="female", date="2026-01-01", file="w.pdf"))
    result = partition_identity_mismatch(weak_name_and_gender, known_docs_boundary)
    assert result["held_labels"] == {"w.pdf"}, "two corroborating weak signals should hold the document"

    print("All checks passed.")
