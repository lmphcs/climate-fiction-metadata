"""Final work-form/relevance decision rule and evaluation statistics."""

from __future__ import annotations
import math
from collections import Counter
from typing import Dict, List, Sequence, Tuple

POSITIVE_RELEVANCE = {"IMPLICIT_OR_ENVIRONMENTAL", "CLIMATE_EXPLICIT"}


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Invalid Boolean value: {value!r}")


def derived_status(fiction: str, relevance: str) -> str:
    if fiction in {"NONFICTION", "OTHER_LITERARY"}:
        return "NEGATIVE"
    if fiction == "UNCLEAR" or relevance in {"", "UNCLEAR"}:
        return "UNRESOLVED"
    if fiction == "FICTION" and relevance in POSITIVE_RELEVANCE:
        return "POSITIVE"
    if fiction == "FICTION" and relevance == "NONE":
        return "NEGATIVE"
    raise ValueError(
        f"Invalid v1.0 judgment: fiction={fiction!r}, relevance={relevance!r}"
    )


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> Tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires total > 0")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def route_signature(row: Dict[str, str], semantic_top_k: int = 100) -> str:
    explicit = parse_bool(row["explicit_route"])
    motif = parse_bool(row["motif_route"])
    semantic = int(row["semantic_rank"]) <= semantic_top_k
    if semantic and (explicit or motif):
        return "route_overlap"
    if explicit and not motif and not semantic:
        return "explicit_only"
    if motif and not explicit and not semantic:
        return "motif_only"
    if semantic and not explicit and not motif:
        return "semantic_only"
    if not semantic and not explicit and not motif:
        return "outside_pool"
    raise ValueError(
        "Record does not match a frozen mutually exclusive stratum: "
        f"{row.get('annotation_id')}"
    )


def kappa(left: Sequence[str], right: Sequence[str]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("Kappa requires two non-empty equal-length sequences")
    total = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / total
    left_counts, right_counts = Counter(left), Counter(right)
    expected = sum(
        left_counts[label] / total * right_counts[label] / total
        for label in set(left_counts) | set(right_counts)
    )
    if expected == 1.0:
        return None
    return (observed - expected) / (1.0 - expected)
