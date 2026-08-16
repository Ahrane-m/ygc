"""
Retrieval Inspection CLI
=========================================
Read-only debug/demo helper: shows exactly what context retrieval.py would
assemble for a patient, and how — without calling the answering model.

Replaces the old inspect_chroma.py. That tool answered "what chunks are in
the vector store?"; the equivalent question now is "what would the model
actually be shown for this question?", which this prints verbatim. Since
retrieval is deterministic, what you see here IS what the model sees.

Never modifies anything. Calls OpenAI only when a question is supplied AND
the record is too large for the context budget (the planner call) — printing
a patient's context with no question makes no API calls at all.

Usage:
    python inspect_records.py "<patient_key>"
        # full assembled context for this patient

    python inspect_records.py "<patient_key>" --question "did my dose change?"
        # context as it would be assembled for that question

    python inspect_records.py "<patient_key>" --summary
        # entity vocabulary + section sizes only, not the full text

patient_key is the user_id for API-ingested records, or the patient name for
records processed through the medical_extractor.py CLI.
"""

import argparse
import sys

import retrieval

# Assembled context contains non-ASCII characters (the "→" in lab_trends'
# flag sequences, the "…" in truncation markers). A Windows console defaults
# to cp1252 and would raise UnicodeEncodeError partway through printing.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def print_summary(record: dict) -> None:
    """Entity inventory and context size — enough to tell whether a record
    will fit the budget without printing the whole thing."""
    timeline = record.get("patient_timeline") or {}
    vocabulary = retrieval.build_record_vocabulary(record)
    med_groups = retrieval.group_medications(timeline)
    lab_groups = retrieval.group_lab_results(timeline)
    context = retrieval.build_full_context(record)

    print(f"Documents on file:      {len(timeline.get('visits') or [])}")
    print(f"Medication entries:     {len(timeline.get('medications_timeline') or [])} "
          f"({len(med_groups)} distinct after cross-document rollup)")
    print(f"Lab result entries:     {len(timeline.get('lab_results_timeline') or [])} "
          f"({len(lab_groups)} distinct tests)")
    print(f"Known allergies:        {len(timeline.get('known_allergies') or [])}")
    print(f"Cross-check computed:   {bool(record.get('cross_check_report'))}")
    print(f"Lab trends computed:    {len((record.get('lab_trends') or {}).get('trends') or [])}")
    print()
    print(f"Full context size:      {len(context)} chars "
          f"(budget {retrieval.DEFAULT_CONTEXT_BUDGET_CHARS})")
    print(f"Fits whole:             {len(context) <= retrieval.DEFAULT_CONTEXT_BUDGET_CHARS} "
          "(if False, questions are answered from a planned subset)")
    print()

    # Medications that changed across documents are the highest-value signal
    # for cross-document follow-ups, so call them out here directly.
    changed = [g for g in med_groups if g["dose_changes"] or g["frequency_changes"]]
    if changed:
        print("Medications that CHANGED across documents:")
        for group in changed:
            if group["dose_changes"]:
                print(f"  {group['display_name']}: dose {' -> '.join(group['dose_changes'])}")
            if group["frequency_changes"]:
                print(f"  {group['display_name']}: frequency {' -> '.join(group['frequency_changes'])}")
        print()

    print("Entity vocabulary (what a follow-up can be resolved against):")
    for field in ("medications", "lab_tests", "source_files", "allergies"):
        values = vocabulary.get(field) or []
        print(f"  {field} ({len(values)}): {', '.join(values) if values else '(none)'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the context retrieval.py assembles for a patient (read-only)."
    )
    parser.add_argument("patient_key", help="user_id (API records) or patient name (CLI records)")
    parser.add_argument(
        "--question",
        default=None,
        help="Assemble context as it would be for this question (may invoke the planner)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print the entity inventory and context size instead of the full context text",
    )
    args = parser.parse_args()

    record = retrieval.load_patient_record(args.patient_key)
    if record is None:
        print(
            f"No processed records found for '{args.patient_key}'.\n"
            "Checked MongoDB (patient_snapshots) and the local "
            f"patient_report_*.json written by the medical_extractor.py CLI."
        )
        return

    print(f"=== Patient '{args.patient_key}' ===\n")

    if args.summary:
        print_summary(record)
        return

    if args.question:
        context, diagnostics = retrieval.assemble_context(record, args.question)
        print(f"Question:  {args.question}")
        print(f"Strategy:  {diagnostics['strategy']} ({diagnostics['context_chars']} chars)")
        if diagnostics.get("plan"):
            print(f"Plan:      {diagnostics['plan']}")
        print("\n" + "-" * 70 + "\n")
        print(context)
        return

    print(retrieval.build_full_context(record))


if __name__ == "__main__":
    main()
