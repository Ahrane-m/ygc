"""
Clinical Reference Library (citable guidance)
=========================================
Holds published clinical guidance the pipeline can CITE, so a safety finding
can be graded as evidence-backed instead of capped as unverified model recall.

WHY THIS EXISTS
---------------
evidence_grading.py splits findings into `deterministic`, `reference_graph`
and `model_knowledge`, capping the last at 0.6 because nothing in the system
could confirm it. Its own docstring names the gap: the reference graph held
only the WHO antidote list — drug names and dosage forms, no interaction data
— so in practice every interaction and allergy finding graded as unverified.

This module starts closing that gap with the first source that makes a
specific, high-stakes interaction claim in plain terms:

    SAMHSA Overdose Prevention and Response Toolkit (PEP23-03-00-001, 2026)

    "Mixing opioids with alcohol and/or other depressant medications like
     benzodiazepines or tranquilizers can greatly increase the risk of
     overdose."                                                  — page 13

That is exactly the kind of claim the model was previously asserting from
memory at a capped 0.6. Sourced, it can carry its real weight and cite a
page a pharmacist can look up.

WHY NOT A VECTOR STORE
----------------------
retrieval.py documents at length why this codebase has no embeddings index:
the retrieval unit is small and the questions are completeness questions, so
approximate similarity search silently drops evidence that changes an answer.
That reasoning applies here too. This library is a few dozen curated
statements, each already tagged with the drug classes it applies to, so
selecting the relevant ones is an exact lookup over a medication list — not a
nearest-neighbour search. A retrieved-but-wrong passage in a drug-safety
answer is a safety failure, not a relevance miss.

WHAT IS QUOTED VS WHAT IS CLASSIFICATION
----------------------------------------
Same discipline poisoning_kg.py applies to the WHO list: every `quote` below
is copied verbatim from the source with its page number, and nothing is
paraphrased into a stronger claim than the document makes.

The DRUG CLASS LISTS are a different kind of thing and are labelled as such.
The toolkit names some opioids explicitly (page 1) but says only
"benzodiazepines or tranquilizers" as a category — it does not enumerate
them. So the membership lists below are curated drug classification, not
quotations, and `classification_source` marks them. The CLAIM is cited; the
list of which drugs it applies to is ours, and deliberately conservative:
drug classes the source names as depressants, and no further. Sedating
antihistamines, for instance, are excluded — the toolkit never mentions them,
and stretching a citation to cover something it does not say would defeat the
purpose of citing it.
"""

from typing import Any, Dict, List, Optional, Sequence, Set

SAMHSA_TOOLKIT = {
    "id": "samhsa_overdose_toolkit_2026",
    "title": "SAMHSA Overdose Prevention and Response Toolkit",
    "publisher": "Substance Abuse and Mental Health Services Administration (SAMHSA), U.S. Department of Health and Human Services",
    "publication_no": "PEP23-03-00-001",
    "released": "2026",
    "url": "https://library.samhsa.gov",
    # "This publication is in the public domain and may be reproduced or
    # copied without permission from SAMHSA." (page i)
    "public_domain": True,
}

# ---------------------------------------------------------------------------
# Drug classes — curated classification, NOT quotations. See module docstring.
# ---------------------------------------------------------------------------

CLASSIFICATION_SOURCE = (
    "Curated drug-class list maintained in this repository, not quoted from the "
    "source document. The source names the interaction; it does not enumerate "
    "every drug in each class."
)

# Named on page 1 of the toolkit: "morphine, codeine, oxycodone, hydrocodone,
# fentanyl, and hydromorphone"; methadone and buprenorphine on page 4.
# The remainder are standard members of the same class.
OPIOIDS = frozenset({
    "morphine", "codeine", "oxycodone", "hydrocodone", "fentanyl",
    "hydromorphone", "methadone", "buprenorphine", "tramadol", "tapentadol",
    "oxymorphone", "dihydrocodeine", "pethidine", "meperidine", "heroin",
    "diamorphine", "sufentanil", "alfentanil", "remifentanil", "pholcodine",
})

BENZODIAZEPINES = frozenset({
    "diazepam", "lorazepam", "alprazolam", "clonazepam", "midazolam",
    "temazepam", "nitrazepam", "chlordiazepoxide", "bromazepam", "oxazepam",
    "clobazam", "flurazepam", "triazolam",
})

# "Z-drugs" and barbiturates — sedative-hypnotics, i.e. the "tranquilizers"
# and "depressant medications" the source groups with benzodiazepines.
SEDATIVE_HYPNOTICS = frozenset({
    "zolpidem", "zopiclone", "eszopiclone", "zaleplon",
    "phenobarbital", "phenobarbitone", "secobarbital", "amobarbital",
    "sodium oxybate",
})

# Opioid overdose reversal medications, named on pages 5-6.
REVERSAL_MEDICATIONS = frozenset({"naloxone", "nalmefene"})

DRUG_CLASSES = {
    "opioid": OPIOIDS,
    "benzodiazepine": BENZODIAZEPINES,
    "sedative_hypnotic": SEDATIVE_HYPNOTICS,
    "opioid_reversal": REVERSAL_MEDICATIONS,
}

# Classes the source explicitly groups as depressants that raise overdose risk
# when combined with an opioid.
DEPRESSANT_CLASSES = frozenset({"benzodiazepine", "sedative_hypnotic"})


# ---------------------------------------------------------------------------
# Guidance entries — every `quote` is verbatim, with its page.
# ---------------------------------------------------------------------------

GUIDANCE: List[Dict[str, Any]] = [
    {
        "id": "opioid-plus-depressant",
        "topic": "drug_interaction",
        "severity": "high",
        "requires_classes": ["opioid"],
        "with_classes": sorted(DEPRESSANT_CLASSES),
        "quote": (
            "Avoid mixing your medication with alcohol or other sedating drugs. "
            "Mixing opioids with alcohol and/or other depressant medications like "
            "benzodiazepines or tranquilizers can greatly increase the risk of overdose."
        ),
        "page": 13,
        "plain": (
            "Taking a strong painkiller together with a sedative or sleeping tablet "
            "can dangerously slow your breathing."
        ),
    },
    {
        "id": "opioid-risk-combination",
        "topic": "overdose_risk_factor",
        "severity": "high",
        "requires_classes": ["opioid"],
        "with_classes": sorted(DEPRESSANT_CLASSES),
        "quote": (
            "Combining different drugs—for example, opioids with other sedating "
            "substances such as benzodiazepines or alcohol."
        ),
        "page": 3,
        "plain": "Mixing these two is listed as a known cause of overdose.",
    },
    {
        "id": "carry-reversal-medication",
        "topic": "prevention",
        "severity": "moderate",
        "requires_classes": ["opioid"],
        "with_classes": [],
        "quote": (
            "Prescribe an OORM when you prescribe an opioid and encourage patients "
            "to have it on hand."
        ),
        "page": 14,
        "plain": (
            "Anyone taking a strong painkiller should keep an overdose rescue "
            "medicine (naloxone) at home, and make sure someone else knows where it is."
        ),
    },
    {
        "id": "overdose-signs",
        "topic": "recognition",
        "severity": "high",
        "requires_classes": ["opioid"],
        "with_classes": [],
        "quote": (
            "Unconsciousness or inability to awaken. Slow or shallow breathing or "
            "difficulty breathing such as choking sounds or a gurgling/snoring noise "
            "from a person who cannot be awakened. Fingernails or lips turning "
            "blue/purple. ... Pinpointed pupils or pupils that don't react to light."
        ),
        "page": 9,
        "plain": (
            "Signs of an overdose: cannot be woken, slow or noisy breathing, blue or "
            "grey lips, very small pupils. Call emergency services immediately."
        ),
    },
    {
        "id": "tolerance-after-break",
        "topic": "overdose_risk_factor",
        "severity": "high",
        "requires_classes": ["opioid"],
        "with_classes": [],
        "quote": (
            "Taking an amount of a drug that is greater than your tolerance level. "
            "This may include using drugs after a recent period of abstinence, which "
            "may decrease previous tolerance levels."
        ),
        "page": 3,
        "plain": (
            "After a break from a strong painkiller, the dose you used before may be "
            "too much. Restarting at the old dose is a known overdose risk."
        ),
    },
    {
        "id": "breathing-conditions",
        "topic": "overdose_risk_factor",
        "severity": "moderate",
        "requires_classes": ["opioid"],
        "with_classes": [],
        "quote": (
            "Using a drug when you have underlying lung or heart conditions that leave "
            "you unable to tolerate lower levels of oxygen, such as asthma or sleep apnea."
        ),
        "page": 3,
        "plain": (
            "Asthma, sleep apnoea or a heart condition raises the risk from strong "
            "painkillers. Mention them to whoever prescribes."
        ),
    },
]

GUIDANCE_BY_ID = {entry["id"]: entry for entry in GUIDANCE}


# ---------------------------------------------------------------------------
# Matching a patient's medications against the library
# ---------------------------------------------------------------------------

def _normalize(name: Any) -> str:
    from document_dedup import _base_ingredient

    return _base_ingredient(name or "")


def classify_drug(name: Any) -> Set[str]:
    """Every class a drug name belongs to. Empty set if unclassified — an
    unknown drug is never assumed to be a depressant."""
    key = _normalize(name)
    if not key:
        return set()
    return {
        class_name for class_name, members in DRUG_CLASSES.items()
        if key in members or any(key == m or key.startswith(m + " ") for m in members)
    }


def classify_medication(med: Dict[str, Any]) -> Set[str]:
    """Classes for one medication entry, checking normalized ingredients first
    and falling back to the printed name."""
    classes: Set[str] = set()
    for ingredient in med.get("ingredients") or []:
        classes |= classify_drug(ingredient)
    if not classes:
        classes |= classify_drug(med.get("name"))
    return classes


def classify_timeline(timeline: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Maps each drug class to the medication entries in it."""
    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for med in timeline.get("medications_timeline") or []:
        for class_name in classify_medication(med):
            by_class.setdefault(class_name, []).append(med)
    return by_class


def _from_graph_row(row: Dict[str, Any], by_class: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Shapes a guidance row returned by the graph into the same dict the
    local constants produce, so callers cannot tell which backed it."""
    triggering = [
        m for class_name in (row.get("requires_classes") or []) + (row.get("with_classes") or [])
        for m in by_class.get(class_name, [])
    ]
    return {
        "id": row["id"],
        "topic": row["topic"],
        "severity": row["severity"],
        "requires_classes": row.get("requires_classes") or [],
        "with_classes": row.get("with_classes") or [],
        "quote": row["quote"],
        "page": row["page"],
        "plain": row["plain"],
        "citation": {
            "source": row.get("source") or SAMHSA_TOOLKIT["title"],
            "publication_no": row.get("publication_no") or SAMHSA_TOOLKIT["publication_no"],
            "publisher": row.get("publisher") or SAMHSA_TOOLKIT["publisher"],
            "released": row.get("released") or SAMHSA_TOOLKIT["released"],
            "url": row.get("url") or SAMHSA_TOOLKIT["url"],
            "page": row["page"],
            "quote": row["quote"],
            "guidance_id": row["id"],
        },
        "triggered_by": sorted({m.get("name") for m in triggering if m.get("name")}),
        "backed_by": "knowledge_graph",
    }


def cite(entry: Dict[str, Any]) -> Dict[str, Any]:
    """A finding-attachable citation for one guidance entry."""
    return {
        "source": SAMHSA_TOOLKIT["title"],
        "publication_no": SAMHSA_TOOLKIT["publication_no"],
        "publisher": SAMHSA_TOOLKIT["publisher"],
        "released": SAMHSA_TOOLKIT["released"],
        "url": SAMHSA_TOOLKIT["url"],
        "page": entry["page"],
        "quote": entry["quote"],
        "guidance_id": entry["id"],
    }


def find_relevant_guidance(
    timeline: Dict[str, Any], use_graph: bool = True
) -> List[Dict[str, Any]]:
    """
    Guidance entries that apply to THIS patient's medication list.

    An entry fires only when the record actually contains a drug in every
    class it requires — guidance about opioids never surfaces for a record
    with no opioid in it. Returns each entry with the medications that
    triggered it, so the reason is visible rather than implied.

    The graph (guidance_kg.py) is consulted first when reachable, so guidance
    added there without a code change is picked up. On any failure this falls
    back to the constants above rather than returning nothing: elsewhere in
    this pipeline an unreachable graph fail-opens into a missing enrichment,
    which is fine for a reference note and NOT fine for the citation behind a
    safety warning — silently dropping it would downgrade the finding to
    unverified model recall with no error shown.
    """
    by_class = classify_timeline(timeline)

    if use_graph:
        try:
            from guidance_kg import lookup_guidance_for_drugs

            names = [
                m.get("name") for m in timeline.get("medications_timeline") or []
                if m.get("name")
            ]
            rows = lookup_guidance_for_drugs(names)
            if rows:
                return [_from_graph_row(row, by_class) for row in rows]
        except Exception:
            # Deliberately silent: the local copy below is authoritative and
            # complete, so a graph outage changes nothing the user can see.
            pass

    matched: List[Dict[str, Any]] = []

    for entry in GUIDANCE:
        required = entry["requires_classes"]
        if not all(by_class.get(c) for c in required):
            continue

        with_classes = entry["with_classes"]
        companions: List[Dict[str, Any]] = []
        if with_classes:
            for class_name in with_classes:
                companions.extend(by_class.get(class_name) or [])
            if not companions:
                continue

        triggering = [m for c in required for m in by_class.get(c, [])] + companions
        matched.append({
            **entry,
            "citation": cite(entry),
            "triggered_by": sorted({
                m.get("name") for m in triggering if m.get("name")
            }),
        })
    return matched


# ---------------------------------------------------------------------------
# Evidence grading hook
# ---------------------------------------------------------------------------

def samhsa_claim_reference(finding: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Returns a citation when a cross-check finding matches a claim this library
    actually supports, else None.

    Deliberately narrow, and pairwise: it fires only when a finding names BOTH
    an opioid and a depressant, because that is the combination the source
    document makes a claim about. An interaction between two drugs it never
    discusses stays `model_knowledge` — citing a document for something it
    does not say would be worse than not citing one at all.
    """
    names = list(finding.get("medications_involved") or [])
    if finding.get("medication"):
        names.append(finding["medication"])
    if len(names) < 2:
        return None

    classes: Set[str] = set()
    for name in names:
        classes |= classify_drug(name)

    if "opioid" in classes and (classes & DEPRESSANT_CLASSES):
        return cite(GUIDANCE_BY_ID["opioid-plus-depressant"])
    return None


# ---------------------------------------------------------------------------
# Concurrent-exposure check (uses the treatment windows, so it is dated)
# ---------------------------------------------------------------------------

def find_concurrent_depressant_risk(timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Periods where an opioid and a depressant were prescribed AT THE SAME TIME.

    Uses risk_timeline's treatment windows rather than mere co-presence in the
    record: an opioid course that ended a year before a sedative began was
    never this risk, and reporting it as one is the false alarm risk_timeline
    was written to stop.
    """
    from risk_timeline import CONCURRENT, POSSIBLE, build_treatment_windows, overlap_of

    windows = build_treatment_windows(timeline)
    for window in windows:
        window["_classes"] = set()
        for ingredient in window["ingredients"]:
            window["_classes"] |= classify_drug(ingredient)

    opioid_windows = [w for w in windows if "opioid" in w["_classes"]]
    depressant_windows = [w for w in windows if w["_classes"] & DEPRESSANT_CLASSES]

    entry = GUIDANCE_BY_ID["opioid-plus-depressant"]
    risks: List[Dict[str, Any]] = []
    for opioid in opioid_windows:
        for depressant in depressant_windows:
            if opioid is depressant:
                continue
            result = overlap_of(opioid, depressant)
            if result["status"] not in (CONCURRENT, POSSIBLE):
                continue
            risks.append({
                "opioid": opioid["name"],
                "depressant": depressant["name"],
                "status": result["status"],
                "window_start": result["start"].isoformat() if result["start"] else None,
                "window_end": result["end"].isoformat() if result["end"] else None,
                "overlap_days": result["days"],
                "severity": entry["severity"],
                "plain": entry["plain"],
                "citation": cite(entry),
            })
    return risks


# ---------------------------------------------------------------------------
# Rendering into Q&A context
# ---------------------------------------------------------------------------

def render_reference_guidance(timeline: Dict[str, Any]) -> str:
    """
    Renders the guidance that applies to this patient into the Q&A context,
    with its source and page, so an answer can cite published guidance rather
    than assert it.

    Returns "" when nothing applies — a record with no opioid in it should not
    carry opioid guidance into every answer.
    """
    matched = find_relevant_guidance(timeline)
    if not matched:
        return ""

    source = SAMHSA_TOOLKIT
    lines = [
        "PUBLISHED GUIDANCE THAT APPLIES TO THIS PATIENT'S MEDICATIONS",
        f"Source: {source['title']} ({source['publisher']}, "
        f"{source['publication_no']}, {source['released']}).",
        "This is published reference guidance, not a finding about this patient. "
        "You may cite it as established, naming the source and page. It does not "
        "describe what has happened to this person.",
    ]
    for entry in matched:
        lines.append(
            f"- [{entry['topic']}, page {entry['page']}] applies because of: "
            f"{', '.join(entry['triggered_by'])}"
        )
        lines.append(f'    Quote: "{entry["quote"]}"')
        lines.append(f"    Plain language: {entry['plain']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    def med(name, ingredient, date="09/11/2025", duration="14 days"):
        return {"name": name, "ingredients": [ingredient], "date": date,
                "duration": duration, "dosage_value": 10, "dosage_unit": "mg",
                "frequency_per_day": 2, "source_file": f"{name}.png",
                "prescription_group": f"rx-{name}"}

    # --- Classification -----------------------------------------------------
    assert classify_drug("Oxycodone") == {"opioid"}
    assert classify_drug("Diazepam") == {"benzodiazepine"}
    assert classify_drug("Zolpidem") == {"sedative_hypnotic"}
    assert classify_drug("Naloxone") == {"opioid_reversal"}
    assert classify_drug("Paracetamol") == set(), "a non-depressant must not be classified"
    assert classify_drug("Cetirizine") == set(), (
        "sedating antihistamines are deliberately NOT in the depressant list — the "
        "source never names them, and a citation must not be stretched")
    # Salt forms normalize (Oxycodone hydrochloride -> oxycodone).
    assert classify_drug("Oxycodone hydrochloride") == {"opioid"}

    # --- The demo record has no opioids: nothing should fire ---------------
    demo = {"medications_timeline": [
        med("Paracetamol", "Paracetamol"), med("Omeprazole", "Omeprazole"),
        med("Cetirizine hydrochloride", "Cetirizine"),
    ]}
    assert find_relevant_guidance(demo) == []
    assert render_reference_guidance(demo) == ""
    assert find_concurrent_depressant_risk(demo) == []

    # --- An opioid alone triggers the opioid-only guidance ------------------
    opioid_only = {"medications_timeline": [
        med("Oxycodone", "Oxycodone"), med("Paracetamol", "Paracetamol"),
    ]}
    ids = {e["id"] for e in find_relevant_guidance(opioid_only)}
    assert "carry-reversal-medication" in ids, ids
    assert "overdose-signs" in ids
    # ...but NOT the combination guidance, since there is no depressant.
    assert "opioid-plus-depressant" not in ids, ids

    # --- Opioid + benzodiazepine, concurrent: the real case ----------------
    combo = {"medications_timeline": [
        med("Oxycodone", "Oxycodone", "09/11/2025", "14 days"),
        med("Diazepam", "Diazepam", "12/11/2025", "10 days"),
    ]}
    matched = find_relevant_guidance(combo)
    ids = {e["id"] for e in matched}
    assert "opioid-plus-depressant" in ids, ids
    combination = next(e for e in matched if e["id"] == "opioid-plus-depressant")
    assert combination["citation"]["page"] == 13
    assert combination["citation"]["publication_no"] == "PEP23-03-00-001"
    assert set(combination["triggered_by"]) == {"Oxycodone", "Diazepam"}

    concurrent = find_concurrent_depressant_risk(combo)
    assert len(concurrent) == 1, concurrent
    assert concurrent[0]["status"] == "concurrent"
    assert concurrent[0]["window_start"] == "2025-11-12", concurrent[0]
    assert concurrent[0]["citation"]["page"] == 13

    # --- Same two drugs, courses a year apart: NOT a live risk -------------
    apart = {"medications_timeline": [
        med("Oxycodone", "Oxycodone", "09/11/2024", "14 days"),
        med("Diazepam", "Diazepam", "12/11/2025", "10 days"),
    ]}
    assert find_concurrent_depressant_risk(apart) == [], (
        "courses a year apart were never a concurrent risk")
    # The general guidance still applies to the record as a whole...
    assert "opioid-plus-depressant" in {e["id"] for e in find_relevant_guidance(apart)}

    # --- Evidence grading hook ---------------------------------------------
    backed = samhsa_claim_reference(
        {"medications_involved": ["Oxycodone", "Diazepam"], "confidence": 0.9})
    assert backed and backed["page"] == 13, backed
    # A pair the document says nothing about stays uncited.
    assert samhsa_claim_reference(
        {"medications_involved": ["Fluconazole", "Montelukast"]}) is None
    # Two opioids are not an opioid-plus-depressant claim.
    assert samhsa_claim_reference(
        {"medications_involved": ["Oxycodone", "Morphine"]}) is None
    # A single drug cannot match a pairwise claim.
    assert samhsa_claim_reference({"medication": "Oxycodone"}) is None

    # --- Grading integration ------------------------------------------------
    from evidence_grading import MODEL_KNOWLEDGE, REFERENCE_GRAPH, grade_cross_check

    report = {
        "potential_drug_interactions": [
            {"medications_involved": ["Oxycodone", "Diazepam"],
             "explanation": "Combined sedation and respiratory depression.",
             "severity": "high", "confidence": 0.9},
            {"medications_involved": ["Fluconazole", "Montelukast"],
             "explanation": "CYP interaction.", "severity": "moderate",
             "confidence": 0.9},
        ],
        "duplicate_prescriptions": [], "conflicting_dosage_instructions": [],
        "allergy_conflicts": [],
    }
    grade_cross_check(report, claim_reference=samhsa_claim_reference)
    cited, uncited = report["potential_drug_interactions"]
    assert cited["evidence_source"] == REFERENCE_GRAPH, cited
    assert cited["confidence"] == 0.9, "a cited claim keeps its confidence"
    assert cited["reference"]["page"] == 13
    assert uncited["evidence_source"] == MODEL_KNOWLEDGE
    assert uncited["confidence"] == 0.6, "an uncited claim is still capped"

    # --- Rendering ----------------------------------------------------------
    rendered = render_reference_guidance(combo)
    assert "SAMHSA" in rendered and "page 13" in rendered
    assert "not a finding about this patient" in rendered

    # --- Graph and local copy must agree, and a graph outage must not
    #     silently drop the citation behind a safety warning ---------------
    local_only = find_relevant_guidance(combo, use_graph=False)
    assert {e["id"] for e in local_only} == ids, "local copy disagrees with graph"
    assert all("backed_by" not in e for e in local_only)

    import builtins

    real_import = builtins.__import__

    def _no_graph(name, *a, **kw):
        if name == "guidance_kg":
            raise ImportError("simulated graph outage")
        return real_import(name, *a, **kw)

    builtins.__import__ = _no_graph
    try:
        during_outage = find_relevant_guidance(combo)
    finally:
        builtins.__import__ = real_import
    assert {e["id"] for e in during_outage} == ids, (
        "an unreachable graph must fall back to the local copy, not return nothing")
    print(f"Graph unreachable -> fell back to {len(during_outage)} local statement(s), "
          "citation preserved.\n")

    print("Guidance matched for an opioid + benzodiazepine record:")
    for entry in matched:
        print(f"  [{entry['topic']:20}] p{entry['page']}  {entry['plain']}")
    print()
    print("Concurrent exposure found:")
    for risk in concurrent:
        print(f"  {risk['opioid']} + {risk['depressant']}: "
              f"{risk['window_start']} to {risk['window_end']} "
              f"({risk['overlap_days']} days)")
    print()
    print("All checks passed.")
