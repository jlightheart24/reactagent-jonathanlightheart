"""Catalog of the five supplied source documents and their roles.

Expected page counts and SHA-256 digests come from U0-HW-09-materials/README.md. Ingestion
checks them so an incomplete or substituted corpus is caught before indexing.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    source_id: str
    filename: str
    title: str
    role: str
    pages: int
    sha256: str


SOURCES: dict[str, Source] = {
    s.source_id: s
    for s in (
        Source(
            "food-code-2026",
            "fda-food-code-2026.pdf",
            "FDA Food Code 2026",
            "FDA model code, 2026 edition. A recommendation that is legally binding only where a "
            "jurisdiction adopts it; adoption requires a separate authority.",
            732,
            "a608605db1a44d07fc9255db281175a933b4290597b4dc4cf58e035158fc8e06",
        ),
        Source(
            "employee-health-handbook",
            "fda-employee-health-personal-hygiene-handbook.pdf",
            "Employee Health and Personal Hygiene Handbook",
            "FDA operational guidance (2020 publication) based on the earlier 2017 Food Code edition; "
            "not the current model-code text.",
            45,
            "d83321e689eab26958f27ee5dd8886d231b9140fdc9cbcca3e30613da80c74e8",
        ),
        Source(
            "bad-bug-book",
            "fda-bad-bug-book-second-edition.pdf",
            "Bad Bug Book, Second Edition",
            "FDA pathogen and natural-toxin reference information (2012); not a regulation.",
            292,
            "b76d65edc9cce029a8e864f51eee8f7ae124157d55532019c6ea9de529ff21d7",
        ),
        Source(
            "appendix-3",
            "fda-appendix-3-bacterial-pathogen-growth-inactivation.pdf",
            "Appendix 3: Bacterial Pathogen Growth and Inactivation",
            "Draft, nonbinding FDA scientific guidance for food processing.",
            23,
            "ba818268cc7d6894f41c0200f14b064ad254da8f649cf7f75d62ca03733207c1",
        ),
        Source(
            "utah-risk-factor-study",
            "utah-risk-factor-study-2016-2017.pdf",
            "Utah Risk Factor Study, 2016-2017",
            "Historical Utah inspection observations (2016-2017); not a statement of current legal "
            "requirements.",
            18,
            "ea1f93b84c18e73d9ffa2f7909705ea0012eef08a440384254e4f5bc7c516737",
        ),
    )
}

SOURCE_IDS: tuple[str, ...] = tuple(SOURCES)


def catalog_text() -> str:
    """One line per source for the system prompt."""
    return "\n".join(f"- {s.source_id}: {s.title}. {s.role}" for s in SOURCES.values())
