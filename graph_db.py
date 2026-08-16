"""
Neo4j connection (drug/poisoning reference knowledge graph)
=========================================
Lazy singleton driver, same shape as db.py's MongoClient singleton.
This graph is a shared reference knowledge base (e.g. WHO EML antidote
data via poisoning_kg.py) — not per-patient data, so unlike db.py there
is no user_id scoping here.

Env:
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE (optional,
    defaults to "neo4j")
"""

import os
from typing import Optional

from neo4j import Driver, GraphDatabase

_driver: Optional[Driver] = None


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        )
    return _driver


def database_name() -> str:
    return os.environ.get("NEO4J_DATABASE", "neo4j")


def ensure_constraints() -> None:
    """Called once at API startup, next to db.ensure_indexes()."""
    with get_driver().session(database=database_name()) as session:
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Medicine) REQUIRE m.name IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:SourceDocument) "
            "REQUIRE s.filename IS UNIQUE"
        )
