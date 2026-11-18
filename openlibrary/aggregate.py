#!/usr/bin/env python3
"""Aggregate parsed Open Library editions and create the eligible work corpus."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import pyarrow as pa
import pyarrow.parquet as pq

FICTION_RE = re.compile(
    r"\bfiction\b|\bnovels\b|\bshort stor(?:y|ies)\b|" r"\bnovel\b(?=$|\s*[-–—,:;/)])",
    re.IGNORECASE,
)
NONFICTION_RE = re.compile(r"\bnon[\s-]?fiction(?:al)?\b", re.IGNORECASE)

YEAR_CONFLICT_REASONS = (
    "edition_more_than_2_years_earlier_supported",
    "edition_more_than_2_years_earlier_singleton_ignored",
    "edition_more_than_2_years_later_than_work",
)


@dataclass(frozen=True)
class YearResolution:
    year: Optional[int]
    source: str
    raw_min_year: Optional[int]
    difference: Optional[int]
    conflict: bool
    conflict_reason: str
    edition_only: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def unique(values: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        text = value.strip() if isinstance(value, str) else ""
        folded = text.casefold()
        if text and folded not in seen:
            seen.add(folded)
            result.append(text)
    return result


def stable_split(work_key: str) -> tuple[int, str]:
    bucket = int(hashlib.sha256(work_key.encode("utf-8")).hexdigest()[:8], 16) % 100
    return bucket, "development" if bucket < 20 else "evaluation"


class EditionAggregate:
    __slots__ = (
        "total_count",
        "english_count",
        "earliest_year",
        "earliest_year_key",
        "earliest_year_raw_date",
        "earliest_year_support_keys",
        "english_keys",
        "best_description",
        "best_description_edition",
        "fallback_title",
        "fallback_subtitle",
        "subjects",
        "genres",
        "publishers",
        "isbn_10",
        "isbn_13",
    )

    def __init__(self) -> None:
        self.total_count = 0
        self.english_count = 0
        self.earliest_year: Optional[int] = None
        self.earliest_year_key = ""
        self.earliest_year_raw_date = ""
        self.earliest_year_support_keys: Set[str] = set()
        self.english_keys: List[str] = []
        self.best_description = ""
        self.best_description_edition = ""
        self.fallback_title = ""
        self.fallback_subtitle = ""
        self.subjects: Set[str] = set()
        self.genres: Set[str] = set()
        self.publishers: Set[str] = set()
        self.isbn_10: Set[str] = set()
        self.isbn_13: Set[str] = set()

    def add(self, row: Dict[str, Any]) -> None:
        self.total_count += 1
        edition_key = row.get("edition_key") or ""
        year = row.get("publish_year")
        if isinstance(year, int) and not isinstance(year, bool):
            raw_date = row.get("publish_date_raw") or ""
            support_key = edition_key or f"<missing-key>:{raw_date}"
            if self.earliest_year is None or year < self.earliest_year:
                self.earliest_year = year
                self.earliest_year_key = edition_key
                self.earliest_year_raw_date = raw_date
                self.earliest_year_support_keys = {support_key}
            elif year == self.earliest_year:
                self.earliest_year_support_keys.add(support_key)
                candidate = (edition_key or "\uffff", raw_date)
                current = (
                    self.earliest_year_key or "\uffff",
                    self.earliest_year_raw_date,
                )
                if candidate < current:
                    self.earliest_year_key = edition_key
                    self.earliest_year_raw_date = raw_date
        if not row.get("is_english"):
            return
        self.english_count += 1
        if edition_key:
            self.english_keys.append(edition_key)
        description = row.get("description") or ""
        if len(description) > len(self.best_description):
            self.best_description = description
            self.best_description_edition = edition_key
        if not self.fallback_title and row.get("title"):
            self.fallback_title = row["title"]
        if not self.fallback_subtitle and row.get("subtitle"):
            self.fallback_subtitle = row["subtitle"]
        self.subjects.update(row.get("subjects") or [])
        self.genres.update(row.get("genres") or [])
        self.publishers.update(row.get("publishers") or [])
        self.isbn_10.update(row.get("isbn_10") or [])
        self.isbn_13.update(row.get("isbn_13") or [])

    @property
    def earliest_year_support_count(self) -> int:
        return len(self.earliest_year_support_keys)


ELIGIBLE_SCHEMA = pa.schema(
    [
        ("work_key", pa.string()),
        ("title", pa.string()),
        ("subtitle", pa.string()),
        ("author_keys", pa.list_(pa.string())),
        ("authors", pa.list_(pa.string())),
        ("raw_min_year", pa.int32()),
        ("year", pa.int32()),
        ("year_source", pa.string()),
        ("work_first_publish_year", pa.int32()),
        ("first_publish_date_raw", pa.string()),
        ("edition_earliest_year", pa.int32()),
        ("edition_earliest_year_key", pa.string()),
        ("edition_earliest_date_raw", pa.string()),
        ("edition_earliest_year_support_count", pa.int32()),
        ("year_difference", pa.int32()),
        ("year_conflict", pa.bool_()),
        ("year_conflict_reason", pa.string()),
        ("year_edition_only", pa.bool_()),
        ("has_english_edition", pa.bool_()),
        ("edition_count", pa.int32()),
        ("english_edition_count", pa.int32()),
        ("english_edition_keys", pa.list_(pa.string())),
        ("work_description", pa.large_string()),
        ("edition_description", pa.large_string()),
        ("edition_description_source", pa.string()),
        ("best_description", pa.large_string()),
        ("description_source", pa.string()),
        ("work_subjects", pa.list_(pa.string())),
        ("edition_subjects", pa.list_(pa.string())),
        ("genres", pa.list_(pa.string())),
        ("subjects", pa.list_(pa.string())),
        ("publishers", pa.list_(pa.string())),
        ("isbn_10", pa.list_(pa.string())),
        ("isbn_13", pa.list_(pa.string())),
        ("fiction_likely", pa.bool_()),
        ("fiction_signals", pa.list_(pa.string())),
        ("fiction_signal_provenance", pa.list_(pa.string())),
        ("nonfiction_signals", pa.list_(pa.string())),
        ("nonfiction_signal_provenance", pa.list_(pa.string())),
        ("fiction_signal_conflict", pa.bool_()),
        ("metadata_text", pa.large_string()),
        ("metadata_characters", pa.int32()),
        ("split", pa.string()),
        ("split_bucket", pa.int32()),
        ("source_snapshot", pa.string()),
        ("work_last_modified", pa.string()),
    ]
)


def load_authors(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=100_000, columns=["author_key", "name"]
    ):
        keys = batch.column(0).to_pylist()
        names = batch.column(1).to_pylist()
        for key, name in zip(keys, names):
            if key and name:
                result[key] = name
    return result


def aggregate_editions(path: Path) -> Dict[str, EditionAggregate]:
    result: Dict[str, EditionAggregate] = {}
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=20_000):
        for row in batch.to_pylist():
            key = row["work_key"]
            aggregate = result.get(key)
            if aggregate is None:
                aggregate = EditionAggregate()
                result[key] = aggregate
            aggregate.add(row)
    return result


def valid_year(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def choose_year(
    work_year: Any,
    edition_year: Optional[int],
    edition_year_support_count: int = 1,
) -> YearResolution:
    """Resolve catalogue years without trusting one anomalously early edition.

    The signed difference is ``edition_year - work_year``. Work metadata is the
    default authority. Dates within two years are treated as ordinary catalogue
    variation and their minimum is retained. An edition more than two years
    earlier overrides the work only when at least two distinct editions support
    that earliest year.
    """

    valid_work_year = valid_year(work_year)
    valid_edition_year = valid_year(edition_year)
    raw_years = [
        value for value in (valid_work_year, valid_edition_year) if value is not None
    ]
    raw_min_year = min(raw_years) if raw_years else None

    if valid_work_year is not None and valid_edition_year is not None:
        difference = valid_edition_year - valid_work_year
        if abs(difference) <= 2:
            selected = min(valid_work_year, valid_edition_year)
            if difference == 0:
                source = "work_and_edition_agree"
            elif difference < 0:
                source = "edition_within_2_years_min"
            else:
                source = "work_within_2_years_min"
            return YearResolution(
                selected, source, raw_min_year, difference, False, "", False
            )
        if difference < -2:
            if edition_year_support_count >= 2:
                return YearResolution(
                    valid_edition_year,
                    "edition_earlier_supported_override",
                    raw_min_year,
                    difference,
                    True,
                    "edition_more_than_2_years_earlier_supported",
                    False,
                )
            return YearResolution(
                valid_work_year,
                "work_preferred_edition_singleton_ignored",
                raw_min_year,
                difference,
                True,
                "edition_more_than_2_years_earlier_singleton_ignored",
                False,
            )
        return YearResolution(
            valid_work_year,
            "work_preferred_edition_later",
            raw_min_year,
            difference,
            True,
            "edition_more_than_2_years_later_than_work",
            False,
        )

    if valid_work_year is not None:
        return YearResolution(
            valid_work_year,
            "work_first_publish_date",
            raw_min_year,
            None,
            False,
            "",
            False,
        )
    if valid_edition_year is not None:
        return YearResolution(
            valid_edition_year,
            "edition_only_earliest_date",
            raw_min_year,
            None,
            False,
            "",
            True,
        )
    return YearResolution(None, "missing", None, None, False, "", False)


def catalog_fiction_signals(
    fields: Sequence[tuple[str, Sequence[str]]],
) -> tuple[List[str], List[str], List[str], List[str]]:
    positive: List[str] = []
    positive_provenance: List[str] = []
    negative: List[str] = []
    negative_provenance: List[str] = []
    for field, values in fields:
        for value in values:
            if not isinstance(value, str) or not value.strip():
                continue
            text = value.strip()
            if NONFICTION_RE.search(text):
                negative.append(text)
                negative_provenance.append(f"{field}::{text}")
            without_nonfiction = NONFICTION_RE.sub(" ", text)
            if FICTION_RE.search(without_nonfiction):
                positive.append(text)
                positive_provenance.append(f"{field}::{text}")
    return (
        unique(positive),
        unique(positive_provenance),
        unique(negative),
        unique(negative_provenance),
    )


def make_eligible_row(
    work: Dict[str, Any],
    editions: EditionAggregate,
    authors_by_key: Dict[str, str],
    snapshot: str,
):
    year_resolution = choose_year(
        work.get("first_publish_year"),
        editions.earliest_year,
        editions.earliest_year_support_count,
    )
    year = year_resolution.year
    if editions.english_count == 0 or year is None or not (1990 <= year <= 2025):
        return None, year_resolution

    title = (work.get("title") or editions.fallback_title or "").strip()
    subtitle = (work.get("subtitle") or editions.fallback_subtitle or "").strip()
    author_keys = unique(work.get("author_keys") or [])
    author_names = unique(authors_by_key.get(key, "") for key in author_keys)
    work_description = work.get("description") or ""
    edition_description = editions.best_description
    if len(work_description) >= len(edition_description):
        best_description = work_description
        description_source = "work" if work_description else "missing"
    else:
        best_description = edition_description
        description_source = "english_edition"

    work_subjects = unique(work.get("subjects") or [])
    edition_subjects = sorted(unique(editions.subjects), key=str.casefold)
    genres = sorted(unique(editions.genres), key=str.casefold)
    combined_subjects = unique([*work_subjects, *edition_subjects, *genres])
    (
        fiction_signals,
        fiction_signal_provenance,
        nonfiction_signals,
        nonfiction_signal_provenance,
    ) = catalog_fiction_signals(
        [
            ("work_subject", work_subjects),
            ("edition_subject", edition_subjects),
            ("edition_genre", genres),
        ]
    )
    metadata_parts = [
        title,
        subtitle,
        "; ".join(author_names),
        " | ".join(combined_subjects),
        best_description,
    ]
    metadata_text = "\n".join(part for part in metadata_parts if part).strip()
    split_bucket, split = stable_split(work["work_key"])
    return (
        {
            "work_key": work["work_key"],
            "title": title,
            "subtitle": subtitle,
            "author_keys": author_keys,
            "authors": author_names,
            "raw_min_year": year_resolution.raw_min_year,
            "year": year,
            "year_source": year_resolution.source,
            "work_first_publish_year": valid_year(work.get("first_publish_year")),
            "first_publish_date_raw": work.get("first_publish_date_raw") or "",
            "edition_earliest_year": editions.earliest_year,
            "edition_earliest_year_key": editions.earliest_year_key,
            "edition_earliest_date_raw": editions.earliest_year_raw_date,
            "edition_earliest_year_support_count": (
                editions.earliest_year_support_count
            ),
            "year_difference": year_resolution.difference,
            "year_conflict": year_resolution.conflict,
            "year_conflict_reason": year_resolution.conflict_reason,
            "year_edition_only": year_resolution.edition_only,
            "has_english_edition": True,
            "edition_count": editions.total_count,
            "english_edition_count": editions.english_count,
            "english_edition_keys": unique(editions.english_keys),
            "work_description": work_description,
            "edition_description": edition_description,
            "edition_description_source": editions.best_description_edition,
            "best_description": best_description,
            "description_source": description_source,
            "work_subjects": work_subjects,
            "edition_subjects": edition_subjects,
            "genres": genres,
            "subjects": combined_subjects,
            "publishers": sorted(unique(editions.publishers), key=str.casefold),
            "isbn_10": sorted(unique(editions.isbn_10)),
            "isbn_13": sorted(unique(editions.isbn_13)),
            "fiction_likely": bool(fiction_signals),
            "fiction_signals": fiction_signals,
            "fiction_signal_provenance": fiction_signal_provenance,
            "nonfiction_signals": nonfiction_signals,
            "nonfiction_signal_provenance": nonfiction_signal_provenance,
            "fiction_signal_conflict": bool(fiction_signals and nonfiction_signals),
            "metadata_text": metadata_text,
            "metadata_characters": len(metadata_text),
            "split": split,
            "split_bucket": split_bucket,
            "source_snapshot": snapshot,
            "work_last_modified": work.get("last_modified") or "",
        },
        year_resolution,
    )


def read_json_marker(path: Path) -> tuple[Dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"Cannot read completion marker {path}: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid JSON completion marker {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Completion marker must be a JSON object: {path}")
    return value, digest


def require_nonnegative_int(marker: Mapping[str, Any], field: str, path: Path) -> int:
    value = marker.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError(f"Invalid {field!r} in completion marker {path}: {value!r}")
    return value


def validate_parse_inputs(parsed_root: Path, buckets: int) -> Dict[str, Any]:
    expected_bucket_counts = {"works": buckets, "editions": buckets, "authors": 1}
    recorded: Dict[str, Any] = {}
    for kind, expected_bucket_count in expected_bucket_counts.items():
        directory = parsed_root / kind
        marker_path = directory / "COMPLETE.json"
        if not marker_path.is_file():
            raise RuntimeError(f"Missing parse completion marker: {marker_path}")
        if (directory / ".CROW_INPROGRESS").exists():
            raise RuntimeError(
                f"Parse output is still marked in progress despite COMPLETE: {directory}"
            )
        marker, marker_sha256 = read_json_marker(marker_path)
        if marker.get("kind") != kind:
            raise RuntimeError(
                f"Parse marker kind mismatch in {marker_path}: "
                f"expected {kind!r}, found {marker.get('kind')!r}"
            )
        marker_buckets = require_nonnegative_int(marker, "buckets", marker_path)
        if marker_buckets != expected_bucket_count:
            raise RuntimeError(
                f"Parse marker bucket mismatch in {marker_path}: expected "
                f"{expected_bucket_count}, found {marker_buckets}"
            )
        line_limit = require_nonnegative_int(marker, "line_limit", marker_path)
        if line_limit != 0:
            raise RuntimeError(
                f"Refusing partial parse marker {marker_path}: line_limit={line_limit}"
            )
        rows_emitted = require_nonnegative_int(marker, "rows_emitted", marker_path)
        rows_written = require_nonnegative_int(marker, "rows_written", marker_path)
        if rows_written != rows_emitted:
            raise RuntimeError(
                f"Parse marker row mismatch in {marker_path}: rows_emitted="
                f"{rows_emitted}, rows_written={rows_written}"
            )
        if (
            not isinstance(marker.get("completed_utc"), str)
            or not marker["completed_utc"].strip()
        ):
            raise RuntimeError(f"Parse marker lacks completed_utc: {marker_path}")

        expected_files = {
            f"bucket-{bucket:03d}.parquet" for bucket in range(expected_bucket_count)
        }
        actual_files = {
            path.name for path in directory.glob("bucket-*.parquet") if path.is_file()
        }
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unexpected:
                details.append(f"unexpected={unexpected}")
            raise RuntimeError(
                f"Parse bucket files do not match marker {marker_path}: "
                + "; ".join(details)
            )

        recorded[kind] = {
            "path": str(marker_path.resolve()),
            "sha256": marker_sha256,
            "kind": kind,
            "buckets": marker_buckets,
            "line_limit": line_limit,
            "input": marker.get("input") or "",
            "rows_emitted": rows_emitted,
            "rows_written": rows_written,
            "completed_utc": marker["completed_utc"],
        }
    return recorded


def aggregate_args_record(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "parsed_root": str(args.parsed_root.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "report": str(args.report.resolve()),
        "snapshot": args.snapshot,
        "buckets": args.buckets,
        "flush_rows": args.flush_rows,
    }


def aggregate_binding(
    args: argparse.Namespace, input_markers: Mapping[str, Any]
) -> tuple[Dict[str, Any], str]:
    code_sha256 = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    binding = {
        "aggregate_args": aggregate_args_record(args),
        "input_marker_sha256": {
            kind: details["sha256"] for kind, details in sorted(input_markers.items())
        },
        "code_sha256": code_sha256,
    }
    encoded = json.dumps(
        binding, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return binding, hashlib.sha256(encoded).hexdigest()


def validate_aggregate_reuse(
    marker: Mapping[str, Any],
    marker_path: Path,
    expected_binding: Mapping[str, Any],
    expected_binding_sha256: str,
) -> None:
    if marker.get("marker_version") != 2:
        raise RuntimeError(
            f"Aggregate completion marker is not safely reusable (marker_version): "
            f"{marker_path}"
        )
    actual_binding = marker.get("aggregate_binding")
    actual_binding_sha256 = marker.get("aggregate_binding_sha256")
    if (
        actual_binding != expected_binding
        or actual_binding_sha256 != expected_binding_sha256
    ):
        raise RuntimeError(
            "Aggregate completion marker dependency mismatch; refusing reuse: "
            f"{marker_path}"
        )
    if (
        not isinstance(marker.get("completed_utc"), str)
        or not marker["completed_utc"].strip()
    ):
        raise RuntimeError(f"Aggregate marker lacks completed_utc: {marker_path}")


def prepare_output(out_dir: Path, overwrite_incomplete: bool):
    temporary = out_dir.parent / f".{out_dir.name}.inprogress"
    if (out_dir / "COMPLETE.json").exists():
        return temporary, True
    if out_dir.exists():
        raise RuntimeError(f"Output exists without marker: {out_dir}")
    if temporary.exists():
        marker = temporary / ".CROW_INPROGRESS"
        if not overwrite_incomplete or not marker.exists():
            raise RuntimeError(f"Incomplete output exists: {temporary}")
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    (temporary / ".CROW_INPROGRESS").write_text(
        f"created_utc={utc_now()}\n", encoding="utf-8"
    )
    return temporary, False


def write_report(path: Path, stats: Dict[str, Any]) -> None:
    eligible = stats["eligible_works"]

    def percentage(value: int) -> str:
        return f"{100.0 * value / eligible:.2f}%" if eligible else "n/a"

    lines = [
        "# Open Library coverage audit v0",
        "",
        f"Generated: {stats['completed_utc']}",
        f"Snapshot: {stats['source_snapshot']}",
        "",
        "## Population construction",
        "",
        f"- Parsed work records: {stats['work_rows']:,}",
        f"- Works with any edition link: {stats['works_with_edition_links']:,}",
        f"- Works with an explicitly English edition: {stats['works_with_english']:,}",
        f"- English-edition works dated 1990--2025: {eligible:,}",
        f"- Fiction-likely eligible works: {stats['fiction_likely']:,} "
        f"({percentage(stats['fiction_likely'])})",
        "",
        "The fiction flag is a permissive cataloguing signal, not a validated",
        "genre label or an eligibility rule. Its source field and literal value",
        "are retained for audit. Main evaluation still requires human judgments.",
        "",
        "## Metadata coverage among eligible works",
        "",
        f"- Work-level description: {stats['work_description']:,} "
        f"({percentage(stats['work_description'])})",
        f"- Any selected description after English-edition enrichment: "
        f"{stats['best_description']:,} ({percentage(stats['best_description'])})",
        f"- Any subject/genre metadata: {stats['subjects']:,} "
        f"({percentage(stats['subjects'])})",
        f"- Resolved author name: {stats['author_names']:,} "
        f"({percentage(stats['author_names'])})",
        "",
        "## Year provenance",
        "",
    ]
    lines.extend(
        f"- {key}: {value:,}" for key, value in sorted(stats["year_sources"].items())
    )
    lines.extend(
        [
            f"- Edition-only resolved years: {stats['year_edition_only']:,}",
            f"- Raw minimum differs from resolved year: "
            f"{stats['raw_min_year_differs']:,}",
            "",
            "## Year conflicts among eligible works",
            "",
            f"- More-than-two-year work/edition conflicts: "
            f"{stats['year_conflicts']:,}",
        ]
    )
    conflict_counts = stats["year_conflict_counts"]
    lines.extend(
        f"- {reason}: {conflict_counts.get(reason, 0):,}"
        for reason in YEAR_CONFLICT_REASONS
    )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Coverage describes the frozen Open Library catalogue. It does not",
            "establish representativeness of contemporary fiction, self-publishing,",
            "or reader reception. English means at least one edition explicitly",
            "catalogued with `/languages/eng`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def aggregate(args: argparse.Namespace) -> Dict[str, Any]:
    works_dir = args.parsed_root / "works"
    editions_dir = args.parsed_root / "editions"
    authors_path = args.parsed_root / "authors" / "bucket-000.parquet"
    input_markers = validate_parse_inputs(args.parsed_root, args.buckets)
    binding, binding_sha256 = aggregate_binding(args, input_markers)

    temporary, complete = prepare_output(args.out_dir, args.overwrite_incomplete)
    if complete:
        marker_path = args.out_dir / "COMPLETE.json"
        existing, _ = read_json_marker(marker_path)
        validate_aggregate_reuse(existing, marker_path, binding, binding_sha256)
        write_report(args.report, existing)
        return existing

    started = time.monotonic()
    print(f"loading_authors={authors_path}", flush=True)
    authors_by_key = load_authors(authors_path)
    print(f"authors_loaded={len(authors_by_key)}", flush=True)
    stats: Dict[str, Any] = {
        "marker_version": 2,
        "source_snapshot": args.snapshot,
        "started_utc": utc_now(),
        "buckets": args.buckets,
        "input_markers": input_markers,
        "aggregate_binding": binding,
        "aggregate_binding_sha256": binding_sha256,
        "authors_loaded": len(authors_by_key),
        "work_rows": 0,
        "works_with_edition_links": 0,
        "works_with_english": 0,
        "eligible_works": 0,
        "fiction_likely": 0,
        "work_description": 0,
        "best_description": 0,
        "subjects": 0,
        "author_names": 0,
        "year_sources": Counter(),
        "year_conflicts": 0,
        "year_conflict_counts": Counter(),
        "year_edition_only": 0,
        "raw_min_year_differs": 0,
        "bucket_stats": [],
    }

    for bucket in range(args.buckets):
        work_path = works_dir / f"bucket-{bucket:03d}.parquet"
        edition_path = editions_dir / f"bucket-{bucket:03d}.parquet"
        edition_aggregates = aggregate_editions(edition_path)
        output_path = temporary / f"bucket-{bucket:03d}.parquet"
        writer: Optional[pq.ParquetWriter] = None
        buffer: List[Dict[str, Any]] = []
        bucket_counts = Counter()

        for batch in pq.ParquetFile(work_path).iter_batches(batch_size=10_000):
            for work in batch.to_pylist():
                stats["work_rows"] += 1
                bucket_counts["work_rows"] += 1
                edition_agg = edition_aggregates.get(work["work_key"])
                if edition_agg is None:
                    continue
                stats["works_with_edition_links"] += 1
                bucket_counts["with_editions"] += 1
                if edition_agg.english_count:
                    stats["works_with_english"] += 1
                    bucket_counts["with_english"] += 1
                row, year_resolution = make_eligible_row(
                    work, edition_agg, authors_by_key, args.snapshot
                )
                if row is None:
                    continue
                stats["eligible_works"] += 1
                bucket_counts["eligible"] += 1
                stats["year_sources"][year_resolution.source] += 1
                stats["year_conflicts"] += int(year_resolution.conflict)
                if year_resolution.conflict_reason:
                    stats["year_conflict_counts"][year_resolution.conflict_reason] += 1
                stats["year_edition_only"] += int(year_resolution.edition_only)
                stats["raw_min_year_differs"] += int(
                    year_resolution.raw_min_year != year_resolution.year
                )
                stats["fiction_likely"] += int(row["fiction_likely"])
                stats["work_description"] += int(bool(row["work_description"]))
                stats["best_description"] += int(bool(row["best_description"]))
                stats["subjects"] += int(bool(row["subjects"]))
                stats["author_names"] += int(bool(row["authors"]))
                buffer.append(row)
                if len(buffer) >= args.flush_rows:
                    if writer is None:
                        writer = pq.ParquetWriter(
                            output_path,
                            ELIGIBLE_SCHEMA,
                            compression="zstd",
                            compression_level=3,
                            use_dictionary=True,
                            write_statistics=True,
                        )
                    writer.write_table(
                        pa.Table.from_pylist(buffer, schema=ELIGIBLE_SCHEMA),
                        row_group_size=len(buffer),
                    )
                    buffer.clear()

        if buffer:
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    ELIGIBLE_SCHEMA,
                    compression="zstd",
                    compression_level=3,
                    use_dictionary=True,
                    write_statistics=True,
                )
            writer.write_table(
                pa.Table.from_pylist(buffer, schema=ELIGIBLE_SCHEMA),
                row_group_size=len(buffer),
            )
        if writer is not None:
            writer.close()
        stats["bucket_stats"].append({"bucket": bucket, **bucket_counts})
        print(
            f"bucket={bucket:03d} works={bucket_counts['work_rows']} "
            f"eligible={bucket_counts['eligible']}",
            flush=True,
        )
        del edition_aggregates
        gc.collect()
        pa.default_memory_pool().release_unused()

    stats["year_sources"] = dict(stats["year_sources"])
    stats["year_conflict_counts"] = dict(stats["year_conflict_counts"])
    stats["completed_utc"] = utc_now()
    stats["elapsed_seconds"] = round(time.monotonic() - started, 3)
    (temporary / "COMPLETE.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # Finalize the report before making a COMPLETE corpus visible. A report
    # failure therefore leaves only the explicitly incomplete temporary corpus.
    write_report(args.report, stats)
    (temporary / ".CROW_INPROGRESS").unlink()
    os.replace(temporary, args.out_dir)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--snapshot", default="2026-06-30")
    parser.add_argument("--buckets", type=int, default=128)
    parser.add_argument("--flush-rows", type=int, default=5000)
    parser.add_argument("--overwrite-incomplete", action="store_true")
    args = parser.parse_args()
    if args.buckets <= 0 or args.flush_rows <= 0:
        parser.error("buckets and flush-rows must be positive")
    try:
        args.report.resolve().relative_to(args.out_dir.resolve())
    except ValueError:
        pass
    else:
        parser.error("report must be outside out-dir for atomic finalization")
    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        stats = aggregate(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
