"""
Clinical guidance in the knowledge graph
=========================================
Loads the curated guidance in reference_library.py into Neo4j, so published
clinical statements live in the same reference store as the WHO antidote list
and can be queried the same way.

    (:Guidance)-[:PUBLISHED_IN]->(:GuidanceSource)
    (:Guidance)-[:REQUIRES]->(:DrugClass)
    (:Guidance)-[:COMBINED_WITH]->(:DrugClass)
    (:DrugClass)-[:INCLUDES]->(:Medicine)

`:Medicine` is the SAME node the WHO ingest already creates, keyed on the
lowercased drug name. So loading this connects the two sources rather than
building a parallel island: naloxone is a WHO-listed antidote AND a member of
the opioid_reversal class, and a single query can now walk from a patient's
drug name to the guidance that applies to it.

WHY THE LOCAL COPY STAYS
------------------------
The graph is fail-open everywhere else in this pipeline — an unreachable
Neo4j must never fail a patient's upload — and that is the right call for an
enrichment. It would be the wrong call here. If this guidance lived ONLY in
the graph, an outage would silently drop the citation behind an
opioid-plus-sedative warning, and the finding would quietly fall back to
capped, unverified model recall. The user would see a weaker warning and no
error.

So reference_library.py remains the source of truth: versioned in git,
self-tested, available offline. The graph is a queryable projection of it,
which is what makes the data joinable with the WHO list and extensible
without a code change. Where they disagree, the local copy wins.

Usage:
    python guidance_kg.py --ingest      # load reference_library into Neo4j
    python guidance_kg.py               # show what is currently loaded
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

from graph_db import run_read, run_write, session_scope

logger = logging.getLogger("guidance_kg")


def ingest_guidance() -> Dict[str, int]:
    """
    Idempotent (MERGE-based) load of reference_library's guidance, source
    metadata and drug-class membership.

    Re-running updates in place rather than duplicating, so it is safe to call
    on every deploy. Returns counts of what was submitted.
    """
    from reference_library import (
        DRUG_CLASSES, GUIDANCE, SAMHSA_TOOLKIT, CLASSIFICATION_SOURCE,
    )

    source = SAMHSA_TOOLKIT
    guidance_rows = [
        {
            "id": entry["id"],
            "topic": entry["topic"],
            "severity": entry["severity"],
            "page": entry["page"],
            "quote": entry["quote"],
            "plain": entry["plain"],
            "requires": entry["requires_classes"],
            "combined_with": entry["with_classes"],
        }
        for entry in GUIDANCE
    ]
    class_rows = [
        {"name": class_name, "members": sorted(members)}
        for class_name, members in DRUG_CLASSES.items()
    ]

    logger.info(
        "ingest: loading %d guidance statement(s) and %d drug class(es) from '%s'",
        len(guidance_rows), len(class_rows), source["title"],
    )

    with session_scope("ingest_guidance") as session:
        run_write(
            session, "ingest_guidance", f"MERGE source '{source['publication_no']}'",
            """
            MERGE (s:GuidanceSource {id: $source.id})
              SET s.title = $source.title,
                  s.publisher = $source.publisher,
                  s.publication_no = $source.publication_no,
                  s.released = $source.released,
                  s.url = $source.url
            """,
            source=source,
        )

        run_write(
            session, "ingest_guidance",
            f"MERGE {len(class_rows)} drug class(es) and their members",
            """
            UNWIND $classes AS row
            MERGE (c:DrugClass {name: row.name})
              SET c.classification_source = $classification_source
            WITH c, row
            UNWIND row.members AS member
            // Same :Medicine node the WHO antidote ingest uses, so the two
            // sources join instead of sitting in parallel.
            MERGE (m:Medicine {name: toLower(member)})
              ON CREATE SET m.display_name = member
            MERGE (c)-[:INCLUDES]->(m)
            """,
            classes=class_rows,
            classification_source=CLASSIFICATION_SOURCE,
        )

        run_write(
            session, "ingest_guidance",
            f"MERGE {len(guidance_rows)} guidance statement(s)",
            """
            MATCH (s:GuidanceSource {id: $source_id})
            UNWIND $rows AS row
            MERGE (g:Guidance {id: row.id})
              SET g.topic = row.topic,
                  g.severity = row.severity,
                  g.page = row.page,
                  g.quote = row.quote,
                  g.plain = row.plain
            MERGE (g)-[:PUBLISHED_IN]->(s)
            """,
            source_id=source["id"],
            rows=guidance_rows,
        )

        # The two class links are written as their own statements rather than
        # as CALL subqueries: the scopeless `CALL { WITH ... }` form is
        # deprecated in current Neo4j and warns on every ingest, and the
        # scoped replacement would not parse on older servers. Plain UNWIND
        # works on both and reads more clearly.
        for step, relationship, field in (
            ("link REQUIRES classes", "REQUIRES", "requires"),
            ("link COMBINED_WITH classes", "COMBINED_WITH", "combined_with"),
        ):
            run_write(
                session, "ingest_guidance", step,
                f"""
                UNWIND $rows AS row
                UNWIND row.{field} AS class_name
                MATCH (g:Guidance {{id: row.id}})
                MERGE (c:DrugClass {{name: class_name}})
                MERGE (g)-[:{relationship}]->(c)
                """,
                rows=[r for r in guidance_rows if r[field]],
            )

    logger.info(
        "ingest: '%s' complete — %d statement(s), %d class(es)",
        source["title"], len(guidance_rows), len(class_rows),
    )
    return {"guidance": len(guidance_rows), "drug_classes": len(class_rows)}


def lookup_guidance_for_drugs(drug_names: Sequence[str]) -> List[Dict[str, Any]]:
    """
    Guidance that applies to a list of drug names, resolved entirely in the
    graph: name -> :Medicine -> :DrugClass -> :Guidance.

    A statement is returned only when EVERY class it requires is represented
    in the supplied drugs, and — where it names companion classes — at least
    one of those is present too. Same rule the local matcher applies, so the
    two cannot drift into disagreeing about when guidance fires.
    """
    names = [n for n in drug_names if n]
    if not names:
        return []

    with session_scope("lookup_guidance") as session:
        return run_read(
            session, "lookup_guidance", f"match guidance for {len(names)} drug name(s)",
            """
            UNWIND $names AS wanted
            MATCH (c:DrugClass)-[:INCLUDES]->(m:Medicine {name: toLower(wanted)})
            WITH collect(DISTINCT c.name) AS present
            MATCH (g:Guidance)-[:PUBLISHED_IN]->(s:GuidanceSource)
            WITH g, s, present,
                 [(g)-[:REQUIRES]->(rc) | rc.name] AS required,
                 [(g)-[:COMBINED_WITH]->(wc) | wc.name] AS companions
            WHERE all(r IN required WHERE r IN present)
              AND (size(companions) = 0 OR any(w IN companions WHERE w IN present))
            RETURN g.id AS id, g.topic AS topic, g.severity AS severity,
                   g.page AS page, g.quote AS quote, g.plain AS plain,
                   required AS requires_classes, companions AS with_classes,
                   s.title AS source, s.publication_no AS publication_no,
                   s.publisher AS publisher, s.released AS released, s.url AS url
            ORDER BY g.severity DESC, g.page
            """,
            names=names,
        )


def loaded_summary() -> Dict[str, Any]:
    """What guidance is currently in the graph — for the CLI and for checking
    an ingest actually landed."""
    with session_scope("guidance_summary") as session:
        sources = run_read(
            session, "guidance_summary", "sources",
            """
            MATCH (s:GuidanceSource)
            RETURN s.title AS title, s.publication_no AS publication_no,
                   s.released AS released,
                   count{(s)<-[:PUBLISHED_IN]-()} AS statements
            ORDER BY s.title
            """,
        )
        classes = run_read(
            session, "guidance_summary", "drug classes",
            """
            MATCH (c:DrugClass)
            RETURN c.name AS name, count{(c)-[:INCLUDES]->()} AS members
            ORDER BY name
            """,
        )
    return {"sources": sources, "drug_classes": classes}


if __name__ == "__main__":
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)-12s %(message)s")

    parser = argparse.ArgumentParser(
        description="Load or inspect published clinical guidance in the Neo4j graph."
    )
    parser.add_argument("--ingest", action="store_true",
                        help="load reference_library.py's guidance into Neo4j (idempotent)")
    args = parser.parse_args()

    if args.ingest:
        counts = ingest_guidance()
        print(f"\nSubmitted {counts['guidance']} guidance statement(s), "
              f"{counts['drug_classes']} drug class(es).")

    summary = loaded_summary()
    print("\nGuidance sources in the graph:")
    for source in summary["sources"]:
        print(f"  {source['title']} ({source['publication_no']}, {source['released']})"
              f" — {source['statements']} statement(s)")
    if not summary["sources"]:
        print("  (none — run with --ingest)")

    print("\nDrug classes:")
    for entry in summary["drug_classes"]:
        print(f"  {entry['name']:20} {entry['members']} member(s)")
