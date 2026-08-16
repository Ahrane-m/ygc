"""
Evidence Grading for Safety Findings
=========================================
Separates safety findings the system can actually EVIDENCE from ones that are
only the language model's recollection, and scores them accordingly.

THE PROBLEM THIS FIXES
----------------------
cross_check_prescriptions() returns interactions, allergy conflicts and dosage
conflicts with confidence scores the model assigns itself. Those scores read
as if they were all the same kind of claim, and they are not:

  * "Cetirizine 10 mg appears on two separate prescriptions" is arithmetic
    over the patient's own extracted records. It is checkable, repeatable,
    and either true or false.
  * "Fluconazole inhibits CYP2C9, raising montelukast levels" is the model
    recalling pharmacology from training. It may well be right — but nothing
    in this system verifies it, and the model has no way to distinguish a
    fact it knows from one it has confabulated.

Both used to arrive at 0.65, 0.95, whatever the model felt, with no way for a
reader to tell which was which. This module makes the difference explicit and
caps ungrounded claims, so a confident-sounding interaction cannot outrank a
finding that was actually verified.

That is not a hedge, it is what the pipeline already tells users: the
cross-check prompt itself states it is "a reasoning layer over extracted
text, NOT a validated clinical drug-interaction database". This grading makes
the scores honour that sentence instead of contradicting it.

THREE GRADES
------------
  deterministic     Computed in code from the patient's own extracted data
                    (e.g. detect_exact_duplicate_medications). Highest trust —
                    no model judgment involved.
  reference_graph   Backed by an ingested reference source in Neo4j (WHO EML
                    today). Trusted, and cites the source document.
  model_knowledge   The model's own pharmacological knowledge, with nothing
                    behind it. Capped, flagged, and told to be confirmed.

NOTE ON TODAY'S COVERAGE: the reference graph currently holds only the WHO
"Antidotes and other substances used in poisonings" list. It contains no
drug-interaction or allergy cross-reactivity data, so in practice nearly every
interaction and allergy finding grades as `model_knowledge`. That is the
honest answer, not a gap in this module — and `graph_backed_findings` below is
the hook to raise findings to `reference_graph` as real interaction reference
data is ingested.
"""

from typing import Any, Dict, Iterable, List, Optional, Set

# Ungrounded findings are capped here. Matches retrieval.LOW_CONFIDENCE_THRESHOLD
# so "low confidence" means one thing across the product — and so the existing
# consult guard, which already escalates at or below this value, fires on them.
MODEL_KNOWLEDGE_CONFIDENCE_CEILING = 0.6

DETERMINISTIC = "deterministic"
REFERENCE_GRAPH = "reference_graph"
MODEL_KNOWLEDGE = "model_knowledge"

# Finding lists in a cross_check report, and whether each is about drugs whose
# names can be matched against the reference graph.
FINDING_LISTS = (
    "potential_drug_interactions",
    "duplicate_prescriptions",
    "conflicting_dosage_instructions",
    "allergy_conflicts",
)

MODEL_KNOWLEDGE_NOTE = (
    "This comes from the language model's own pharmacological knowledge — no "
    "reference source in this system confirms it. It may be correct, but it has "
    "not been checked against a drug-interaction database. Treat it as a prompt "
    "to ask a pharmacist, not as an established fact."
)

DETERMINISTIC_NOTE = (
    "Computed directly from the patient's own extracted records, not from model "
    "knowledge — this one is checkable against the source documents."
)

REFERENCE_GRAPH_NOTE = (
    "Backed by a reference document ingested into the knowledge graph, cited below."
)


def _finding_drug_names(finding: Dict[str, Any]) -> Set[str]:
    """Every drug name a finding refers to, lowercased."""
    names: Set[str] = set()
    for key in ("medication", "allergy"):
        value = finding.get(key)
        if isinstance(value, str) and value.strip():
            names.add(value.strip().lower())
    for value in finding.get("medications_involved") or []:
        if isinstance(value, str) and value.strip():
            names.add(value.strip().lower())
    return names


def grade_finding(
    finding: Dict[str, Any],
    graph_backed_findings: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Grades one finding in place and returns it.

    Adds:
      evidence_source : "deterministic" | "reference_graph" | "model_knowledge"
      grounded        : bool  (False only for model_knowledge)
      evidence_note   : str   plain-language explanation of the grade
      reference       : dict  present only for reference_graph findings

    A finding that already declares `evidence_source` (as the deterministic
    duplicate detector does) keeps it — this never downgrades a finding that
    code produced into one that looks like model output.
    """
    existing = finding.get("evidence_source")
    if existing == DETERMINISTIC:
        finding["grounded"] = True
        finding.setdefault("evidence_note", DETERMINISTIC_NOTE)
        return finding

    graph_backed_findings = graph_backed_findings or {}
    matched = {
        name: graph_backed_findings[name]
        for name in _finding_drug_names(finding)
        if name in graph_backed_findings
    }

    if matched:
        finding["evidence_source"] = REFERENCE_GRAPH
        finding["grounded"] = True
        finding["reference"] = matched
        finding["evidence_note"] = REFERENCE_GRAPH_NOTE
        return finding

    finding["evidence_source"] = MODEL_KNOWLEDGE
    finding["grounded"] = False
    finding["evidence_note"] = MODEL_KNOWLEDGE_NOTE

    raw = finding.get("confidence")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        capped = min(float(raw), MODEL_KNOWLEDGE_CONFIDENCE_CEILING)
        if capped < raw:
            # Keep what the model claimed. Silently overwriting it would hide
            # that the cap was applied, and the gap between the two is useful
            # signal — a model asserting 0.95 for something nothing can verify
            # is worth being able to see.
            finding["model_reported_confidence"] = round(float(raw), 2)
        finding["confidence"] = round(capped, 2)
    else:
        finding["confidence"] = MODEL_KNOWLEDGE_CONFIDENCE_CEILING

    return finding


def grade_cross_check(
    cross_check: Dict[str, Any],
    graph_backed_findings: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Grades every finding in a cross_check report, in place, and adds an
    `evidence_summary` describing the mix.

    `graph_backed_findings` maps a lowercased drug name to whatever reference
    record supports it (e.g. the WHO listing from poisoning_kg). Any finding
    naming one of those drugs is graded `reference_graph` and keeps its
    confidence uncapped.
    """
    counts = {DETERMINISTIC: 0, REFERENCE_GRAPH: 0, MODEL_KNOWLEDGE: 0}

    for list_name in FINDING_LISTS:
        for finding in cross_check.get(list_name) or []:
            if not isinstance(finding, dict):
                continue
            grade_finding(finding, graph_backed_findings)
            counts[finding["evidence_source"]] += 1

    total = sum(counts.values())
    cross_check["evidence_summary"] = {
        "total_findings": total,
        "deterministic": counts[DETERMINISTIC],
        "reference_graph": counts[REFERENCE_GRAPH],
        "model_knowledge": counts[MODEL_KNOWLEDGE],
        "model_knowledge_confidence_ceiling": MODEL_KNOWLEDGE_CONFIDENCE_CEILING,
        "note": (
            f"{counts[DETERMINISTIC] + counts[REFERENCE_GRAPH]} of {total} finding(s) "
            "are backed by the patient's own records or by an ingested reference "
            f"document. The other {counts[MODEL_KNOWLEDGE]} come from the language "
            "model's own knowledge with nothing in this system confirming them, so "
            f"their confidence is capped at {MODEL_KNOWLEDGE_CONFIDENCE_CEILING} and "
            "each is flagged. A pharmacist can confirm any of them."
        ) if total else (
            "No safety findings were reported, so there is nothing to grade."
        ),
    }
    return cross_check


def graph_backed_findings_from_antidotes(
    antidote_references: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Adapts poisoning_kg.lookup_antidote_references() output into the
    `graph_backed_findings` shape.

    Deliberately narrow: being WHO-listed as an antidote evidences that the
    DRUG is in a reference document, not that any particular interaction claim
    about it is true. It is used here to cite a real source alongside a finding
    — never to imply the graph confirmed an interaction it has no data about.
    """
    backed: Dict[str, Dict[str, Any]] = {}
    for name, ref in (antidote_references or {}).items():
        if not name:
            continue
        backed[name.strip().lower()] = {
            "source": "WHO Model List of Essential Medicines (antidotes section)",
            "display_name": ref.get("display_name"),
            "listings": ref.get("listings", []),
        }
    return backed


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Findings taken from the user's real cross_check_report.
    report = {
        "potential_drug_interactions": [
            {"medications_involved": ["Fluconazole", "Montelukast"],
             "explanation": "Fluconazole inhibits CYP enzymes...",
             "severity": "moderate", "confidence": 0.65},
            {"medications_involved": ["Cetirizine", "Chlorpheniramine"],
             "explanation": "Additive sedation.",
             "severity": "moderate", "confidence": 0.95},
            {"medications_involved": ["Fluconazole", "Omeprazole"],
             "explanation": "Both affect CYP2C19; clinical impact usually small.",
             "severity": "low", "confidence": 0.45},
        ],
        "duplicate_prescriptions": [
            {"medication": "Cetirizine", "occurrences": [], "confidence": 0.95,
             "explanation": "Deterministic check: identical active ingredient(s)...",
             "evidence_source": DETERMINISTIC},
            {"medication": "Paracetamol", "occurrences": [], "confidence": 0.9,
             "explanation": "Model spotted these look similar."},
        ],
        "conflicting_dosage_instructions": [],
        "allergy_conflicts": [
            {"medication": "Amoxicillin", "allergy": "Penicillin",
             "explanation": "Amoxicillin is a penicillin-class antibiotic.",
             "confidence": 0.93},
        ],
    }

    grade_cross_check(report)

    # Model-knowledge findings are capped and flagged, and what the model
    # originally claimed is preserved.
    interaction = report["potential_drug_interactions"][1]
    assert interaction["evidence_source"] == MODEL_KNOWLEDGE
    assert interaction["grounded"] is False
    assert interaction["confidence"] == 0.6, interaction["confidence"]
    assert interaction["model_reported_confidence"] == 0.95
    assert "pharmacist" in interaction["evidence_note"]

    # 0.65 sits just above the ceiling, so it is capped too — the cap is not
    # only for the loudly-overconfident ones.
    just_over = report["potential_drug_interactions"][0]
    assert just_over["confidence"] == 0.6, just_over["confidence"]
    assert just_over["model_reported_confidence"] == 0.65

    # A model finding already BELOW the ceiling keeps its own value and gains
    # no misleading "was capped" marker — capping must never inflate a score.
    low = report["potential_drug_interactions"][2]
    assert low["confidence"] == 0.45, low["confidence"]
    assert "model_reported_confidence" not in low

    # The deterministic duplicate is NOT capped or downgraded.
    deterministic = report["duplicate_prescriptions"][0]
    assert deterministic["evidence_source"] == DETERMINISTIC
    assert deterministic["grounded"] is True
    assert deterministic["confidence"] == 0.95, "verifiable findings must keep their score"
    assert "model_reported_confidence" not in deterministic

    # An LLM-authored duplicate IS capped, even in the same list.
    assert report["duplicate_prescriptions"][1]["confidence"] == 0.6

    # The allergy claim is model knowledge (nothing in this system holds
    # penicillin cross-reactivity data) and is capped accordingly.
    allergy = report["allergy_conflicts"][0]
    assert allergy["evidence_source"] == MODEL_KNOWLEDGE
    assert allergy["confidence"] == 0.6
    assert allergy["model_reported_confidence"] == 0.93

    summary = report["evidence_summary"]
    assert summary["total_findings"] == 6
    assert summary["deterministic"] == 1
    assert summary["model_knowledge"] == 5
    assert summary["reference_graph"] == 0

    # --- Reference-graph grounding raises a finding above the cap ----------
    backed = graph_backed_findings_from_antidotes({
        "Naloxone": {"display_name": "naloxone",
                     "listings": [{"population": "adult", "source_document": "WHO.pdf"}]},
    })
    graph_report = {
        "potential_drug_interactions": [
            {"medications_involved": ["Naloxone", "Morphine"],
             "explanation": "Naloxone reverses opioid effects.",
             "severity": "high", "confidence": 0.9},
        ],
        "duplicate_prescriptions": [], "conflicting_dosage_instructions": [],
        "allergy_conflicts": [],
    }
    grade_cross_check(graph_report, backed)
    grounded = graph_report["potential_drug_interactions"][0]
    assert grounded["evidence_source"] == REFERENCE_GRAPH
    assert grounded["grounded"] is True
    assert grounded["confidence"] == 0.9, "a graph-backed finding is not capped"
    assert "naloxone" in grounded["reference"]

    # --- An empty report grades cleanly ------------------------------------
    empty = grade_cross_check({"potential_drug_interactions": [], "duplicate_prescriptions": [],
                               "conflicting_dosage_instructions": [], "allergy_conflicts": []})
    assert empty["evidence_summary"]["total_findings"] == 0
    assert "nothing to grade" in empty["evidence_summary"]["note"]

    print("Graded findings from the real report:")
    for name in FINDING_LISTS:
        for f in report.get(name) or []:
            label = f.get("medication") or " + ".join(f.get("medications_involved") or [])
            was = f.get("model_reported_confidence")
            print(f"  [{f['evidence_source']:16}] {label:28} confidence={f['confidence']}"
                  + (f"  (model claimed {was})" if was else ""))
    print("\n" + report["evidence_summary"]["note"])
    print("\nAll checks passed.")
