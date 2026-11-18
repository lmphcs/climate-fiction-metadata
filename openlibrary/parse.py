#!/usr/bin/env python3
"""Stream one Open Library gzip dump into deterministic Parquet buckets."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

YEAR_RE = re.compile(r"(?<!\d)(1[0-9]{3}|20[0-9]{2}|2100)(?!\d)")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def text_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        nested = value.get("value")
        return nested.strip() if isinstance(nested, str) else ""
    return ""


def string_list(value: Any) -> List[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    result: List[str] = []
    seen = set()
    for item in values:
        text = text_value(item)
        folded = text.casefold()
        if text and folded not in seen:
            seen.add(folded)
            result.append(text)
    return result


def key_list(value: Any, nested: Optional[str] = None) -> List[str]:
    values = value if isinstance(value, list) else []
    result: List[str] = []
    seen = set()
    for item in values:
        current: Any = item
        if nested and isinstance(current, Mapping):
            current = current.get(nested)
        key = current.get("key") if isinstance(current, Mapping) else None
        if isinstance(key, str) and key not in seen:
            seen.add(key)
            result.append(key)
    return result


def parse_year(value: Any) -> Optional[int]:
    text = text_value(value)
    for match in YEAR_RE.finditer(text):
        year = int(match.group(1))
        if 1000 <= year <= 2100:
            return year
    return None


def bucket_for(key: str, bucket_count: int) -> int:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % bucket_count


WORK_SCHEMA = pa.schema(
    [
        ("work_key", pa.string()),
        ("title", pa.string()),
        ("subtitle", pa.string()),
        ("description", pa.large_string()),
        ("subjects", pa.list_(pa.string())),
        ("subject_places", pa.list_(pa.string())),
        ("subject_times", pa.list_(pa.string())),
        ("subject_people", pa.list_(pa.string())),
        ("author_keys", pa.list_(pa.string())),
        ("original_language_keys", pa.list_(pa.string())),
        ("first_publish_date_raw", pa.string()),
        ("first_publish_year", pa.int32()),
        ("revision", pa.int32()),
        ("last_modified", pa.string()),
    ]
)

EDITION_SCHEMA = pa.schema(
    [
        ("work_key", pa.string()),
        ("edition_key", pa.string()),
        ("title", pa.string()),
        ("subtitle", pa.string()),
        ("description", pa.large_string()),
        ("subjects", pa.list_(pa.string())),
        ("genres", pa.list_(pa.string())),
        ("publishers", pa.list_(pa.string())),
        ("isbn_10", pa.list_(pa.string())),
        ("isbn_13", pa.list_(pa.string())),
        ("language_keys", pa.list_(pa.string())),
        ("is_english", pa.bool_()),
        ("publish_date_raw", pa.string()),
        ("publish_year", pa.int32()),
        ("revision", pa.int32()),
        ("last_modified", pa.string()),
    ]
)

AUTHOR_SCHEMA = pa.schema(
    [
        ("author_key", pa.string()),
        ("name", pa.string()),
        ("alternate_names", pa.list_(pa.string())),
        ("birth_date", pa.string()),
        ("death_date", pa.string()),
        ("remote_ids_json", pa.string()),
        ("revision", pa.int32()),
        ("last_modified", pa.string()),
    ]
)


class BucketWriters:
    def __init__(
        self,
        directory: Path,
        schema: pa.Schema,
        bucket_count: int,
        flush_rows: int,
        max_buffered_rows: int = 50_000,
    ) -> None:
        self.directory = directory
        self.schema = schema
        self.bucket_count = bucket_count
        self.flush_rows = flush_rows
        self.max_buffered_rows = max_buffered_rows
        self.buffers: List[List[Dict[str, Any]]] = [[] for _ in range(bucket_count)]
        self.writers: Dict[int, pq.ParquetWriter] = {}
        self.rows_written = 0
        self.buffered_rows = 0
        self.peak_buffered_rows = 0

    def add(self, bucket: int, row: Dict[str, Any]) -> None:
        buffer = self.buffers[bucket]
        buffer.append(row)
        self.buffered_rows += 1
        self.peak_buffered_rows = max(self.peak_buffered_rows, self.buffered_rows)
        if len(buffer) >= self.flush_rows:
            self.flush(bucket)
        while self.buffered_rows >= self.max_buffered_rows:
            largest_bucket = max(
                range(self.bucket_count), key=lambda index: len(self.buffers[index])
            )
            if not self.buffers[largest_bucket]:
                raise RuntimeError("Buffered-row accounting is inconsistent")
            self.flush(largest_bucket)

    def new_writer(self, bucket: int) -> pq.ParquetWriter:
        path = self.directory / f"bucket-{bucket:03d}.parquet"
        return pq.ParquetWriter(
            path,
            self.schema,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
            write_statistics=True,
        )

    def flush(self, bucket: int) -> None:
        rows = self.buffers[bucket]
        if not rows:
            return
        writer = self.writers.get(bucket)
        if writer is None:
            writer = self.new_writer(bucket)
            self.writers[bucket] = writer
        table = pa.Table.from_pylist(rows, schema=self.schema)
        writer.write_table(table, row_group_size=len(rows))
        row_count = len(rows)
        self.rows_written += row_count
        self.buffered_rows -= row_count
        rows.clear()

    def close(self) -> None:
        for bucket in range(self.bucket_count):
            self.flush(bucket)
        for bucket in range(self.bucket_count):
            if bucket not in self.writers:
                self.new_writer(bucket).close()
        for writer in self.writers.values():
            writer.close()


def code_sha256() -> str:
    script_path = Path(__file__).resolve()
    return hashlib.sha256(script_path.read_bytes()).hexdigest()


def build_run_identity(args: argparse.Namespace, bucket_count: int) -> Dict[str, Any]:
    input_path = args.input.resolve(strict=True)
    input_size = input_path.stat().st_size
    expected_size = getattr(args, "expected_size", None)
    input_md5 = getattr(args, "input_md5", None)
    if expected_size is not None and input_size != expected_size:
        raise RuntimeError(
            f"Input size mismatch for {input_path}: expected "
            f"{expected_size}, found {input_size}"
        )
    return {
        "input_path": str(input_path),
        "input_size": input_size,
        "kind": args.kind,
        "bucket_count": bucket_count,
        "line_limit": args.limit,
        "flush_rows": args.flush_rows,
        "max_buffered_rows": getattr(args, "max_buffered_rows", 50_000),
        "code_sha256": code_sha256(),
        "expected_size": expected_size,
        "input_md5": input_md5,
    }


def assert_matching_identity(
    complete_stats: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    found = complete_stats.get("run_identity")
    if found == expected:
        return
    found_mapping = found if isinstance(found, Mapping) else {}
    keys = sorted(set(expected) | set(found_mapping))
    differences = {
        key: {"expected": expected.get(key), "found": found_mapping.get(key)}
        for key in keys
        if expected.get(key) != found_mapping.get(key)
    }
    raise RuntimeError(
        "Completed output identity mismatch; refusing reuse: "
        + json.dumps(differences, ensure_ascii=False, sort_keys=True)
    )


def work_rows(obj: Mapping[str, Any], revision: int, modified: str):
    key = obj.get("key")
    if not isinstance(key, str) or not key.startswith("/works/"):
        return []
    raw_date = text_value(obj.get("first_publish_date"))
    return [
        {
            "work_key": key,
            "title": text_value(obj.get("title")),
            "subtitle": text_value(obj.get("subtitle")),
            "description": text_value(obj.get("description")),
            "subjects": string_list(obj.get("subjects")),
            "subject_places": string_list(obj.get("subject_places")),
            "subject_times": string_list(obj.get("subject_times")),
            "subject_people": string_list(obj.get("subject_people")),
            "author_keys": key_list(obj.get("authors"), nested="author"),
            "original_language_keys": key_list(obj.get("original_languages")),
            "first_publish_date_raw": raw_date,
            "first_publish_year": parse_year(raw_date),
            "revision": revision,
            "last_modified": modified,
        }
    ]


def edition_rows(obj: Mapping[str, Any], revision: int, modified: str):
    edition_key = obj.get("key")
    work_keys = key_list(obj.get("works"))
    if not isinstance(edition_key, str) or not work_keys:
        return []
    languages = key_list(obj.get("languages"))
    raw_date = text_value(obj.get("publish_date"))
    base = {
        "edition_key": edition_key,
        "title": text_value(obj.get("title")),
        "subtitle": text_value(obj.get("subtitle")),
        "description": text_value(obj.get("description")),
        "subjects": string_list(obj.get("subjects")),
        "genres": string_list(obj.get("genres")),
        "publishers": string_list(obj.get("publishers")),
        "isbn_10": string_list(obj.get("isbn_10")),
        "isbn_13": string_list(obj.get("isbn_13")),
        "language_keys": languages,
        "is_english": "/languages/eng" in languages,
        "publish_date_raw": raw_date,
        "publish_year": parse_year(raw_date),
        "revision": revision,
        "last_modified": modified,
    }
    return [{"work_key": work_key, **base} for work_key in work_keys]


def author_rows(obj: Mapping[str, Any], revision: int, modified: str):
    key = obj.get("key")
    if not isinstance(key, str) or not key.startswith("/authors/"):
        return []
    remote_ids = obj.get("remote_ids")
    return [
        {
            "author_key": key,
            "name": text_value(obj.get("name")),
            "alternate_names": string_list(obj.get("alternate_names")),
            "birth_date": text_value(obj.get("birth_date")),
            "death_date": text_value(obj.get("death_date")),
            "remote_ids_json": json.dumps(
                remote_ids if isinstance(remote_ids, Mapping) else {},
                ensure_ascii=False,
                sort_keys=True,
            ),
            "revision": revision,
            "last_modified": modified,
        }
    ]


KIND_CONFIG = {
    "works": (WORK_SCHEMA, work_rows, "work_key"),
    "editions": (EDITION_SCHEMA, edition_rows, "work_key"),
    "authors": (AUTHOR_SCHEMA, author_rows, "author_key"),
}


def prepare_output(
    out_root: Path,
    kind: str,
    overwrite_incomplete: bool,
    run_identity: Mapping[str, Any],
):
    final_dir = out_root / kind
    temporary_dir = out_root / f".{kind}.inprogress"
    if (final_dir / "COMPLETE.json").exists():
        complete_stats = json.loads(
            (final_dir / "COMPLETE.json").read_text(encoding="utf-8")
        )
        assert_matching_identity(complete_stats, run_identity)
        print(f"already_complete={final_dir}")
        return final_dir, temporary_dir, complete_stats
    if final_dir.exists():
        raise RuntimeError(f"Final directory exists without marker: {final_dir}")
    if temporary_dir.exists():
        marker = temporary_dir / ".CROW_INPROGRESS"
        if not overwrite_incomplete or not marker.exists():
            raise RuntimeError(
                f"Incomplete directory exists: {temporary_dir}; inspect it or "
                "rerun with --overwrite-incomplete"
            )
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)
    (temporary_dir / ".CROW_INPROGRESS").write_text(
        f"kind={kind}\ncreated_utc={utc_now()}\n", encoding="utf-8"
    )
    return final_dir, temporary_dir, None


def parse_dump(args: argparse.Namespace) -> Dict[str, Any]:
    schema, row_builder, key_field = KIND_CONFIG[args.kind]
    bucket_count = 1 if args.kind == "authors" else args.buckets
    max_buffered_rows = getattr(args, "max_buffered_rows", 50_000)
    run_identity = build_run_identity(args, bucket_count)
    final_dir, temporary_dir, complete_stats = prepare_output(
        args.out_root,
        args.kind,
        args.overwrite_incomplete,
        run_identity,
    )
    if complete_stats is not None:
        return complete_stats

    writers = BucketWriters(
        temporary_dir,
        schema,
        bucket_count,
        args.flush_rows,
        max_buffered_rows,
    )
    stats: Dict[str, Any] = {
        "run_identity": run_identity,
        "kind": args.kind,
        "input": run_identity["input_path"],
        "started_utc": utc_now(),
        "buckets": bucket_count,
        "line_limit": args.limit,
        "max_buffered_rows": max_buffered_rows,
        "lines_seen": 0,
        "records_parsed": 0,
        "rows_emitted": 0,
        "records_without_output_rows": 0,
        "malformed_tsv": 0,
        "json_errors": 0,
        "error_examples": [],
    }
    started = time.monotonic()
    try:
        with gzip.open(args.input, "rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if args.limit and line_number > args.limit:
                    break
                stats["lines_seen"] += 1
                parts = raw_line.rstrip(b"\n").split(b"\t", 4)
                if len(parts) != 5:
                    stats["malformed_tsv"] += 1
                    if len(stats["error_examples"]) < 20:
                        stats["error_examples"].append(
                            {"line": line_number, "error": "malformed_tsv"}
                        )
                    continue
                try:
                    obj = json.loads(parts[4])
                    revision = int(parts[2])
                    modified = parts[3].decode("utf-8", errors="replace")
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                    stats["json_errors"] += 1
                    if len(stats["error_examples"]) < 20:
                        stats["error_examples"].append(
                            {
                                "line": line_number,
                                "error": type(exc).__name__,
                                "message": str(exc)[:300],
                            }
                        )
                    continue
                stats["records_parsed"] += 1
                rows = row_builder(obj, revision, modified)
                if not rows:
                    stats["records_without_output_rows"] += 1
                    continue
                for row in rows:
                    bucket = bucket_for(row[key_field], bucket_count)
                    writers.add(bucket, row)
                    stats["rows_emitted"] += 1
                if line_number % args.progress_every == 0:
                    elapsed = time.monotonic() - started
                    rate = line_number / elapsed if elapsed else 0.0
                    print(
                        f"progress kind={args.kind} lines={line_number} "
                        f"rows={stats['rows_emitted']} rate_lines_s={rate:.1f}",
                        flush=True,
                    )
        writers.close()
    except Exception:
        for writer in writers.writers.values():
            writer.close()
        raise

    stats["completed_utc"] = utc_now()
    stats["elapsed_seconds"] = round(time.monotonic() - started, 3)
    stats["rows_written"] = writers.rows_written
    stats["peak_buffered_rows"] = writers.peak_buffered_rows
    (temporary_dir / "COMPLETE.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (temporary_dir / ".CROW_INPROGRESS").unlink()
    os.replace(temporary_dir, final_dir)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=sorted(KIND_CONFIG), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--buckets", type=int, default=128)
    parser.add_argument("--flush-rows", type=int, default=5000)
    parser.add_argument("--max-buffered-rows", type=int, default=50_000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument("--expected-size", type=int)
    parser.add_argument("--input-md5")
    parser.add_argument("--overwrite-incomplete", action="store_true")
    args = parser.parse_args()

    if (
        args.buckets <= 0
        or args.flush_rows <= 0
        or args.max_buffered_rows <= 0
        or args.limit < 0
    ):
        parser.error(
            "buckets/flush-rows/max-buffered-rows must be positive and "
            "limit non-negative"
        )
    if args.expected_size is not None and args.expected_size < 0:
        parser.error("expected-size must be non-negative")
    if args.input_md5 is not None:
        args.input_md5 = args.input_md5.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{32}", args.input_md5):
            parser.error("input-md5 must be exactly 32 hexadecimal characters")
    if not args.input.is_file():
        parser.error(f"input not found: {args.input}")
    args.input = args.input.resolve(strict=True)
    args.out_root.mkdir(parents=True, exist_ok=True)

    try:
        stats = parse_dump(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
