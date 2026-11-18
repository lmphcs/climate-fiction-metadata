"""Annotate final-schema metadata packets; save each completed record atomically."""

from pathlib import Path
import argparse
import csv
from openlibrary.evaluation import derived_status


def choose(prompt, values):
    while True:
        print(prompt + ": " + ", ".join(f"{i+1}={v}" for i, v in enumerate(values)))
        value = input("> ").strip()
        if value.isdigit() and 1 <= int(value) <= len(values):
            return values[int(value) - 1]
        if value in values:
            return value


def run(packet, output):
    source = output if output.exists() else packet
    with source.open(newline="") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0])
    for k in [
        "fiction_status",
        "environment_climate_relevance",
        "derived_status",
        "annotation_comment",
    ]:
        if k not in fields:
            fields.append(k)
    for r in rows:
        if r.get("derived_status"):
            continue
        for k in [
            "annotation_id",
            "title",
            "subtitle",
            "authors",
            "first_publish_year",
            "subjects",
            "description",
        ]:
            print(f'\n{k}: {r.get(k,"")}')
        fiction = choose("Form", ["FICTION", "NONFICTION", "OTHER_LITERARY", "UNCLEAR"])
        relevance = (
            choose(
                "Relevance",
                ["NONE", "IMPLICIT_OR_ENVIRONMENTAL", "CLIMATE_EXPLICIT", "UNCLEAR"],
            )
            if fiction in {"FICTION", "UNCLEAR"}
            else ""
        )
        r.update(
            fiction_status=fiction,
            environment_climate_relevance=relevance,
            derived_status=derived_status(fiction, relevance),
            annotation_comment=input("Optional comment: "),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix(".tmp")
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        tmp.replace(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.packet, a.output)
