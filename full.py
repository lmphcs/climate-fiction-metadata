#!/usr/bin/env python3
"""Portable stages for source acquisition, corpus reconstruction and evaluation."""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import yaml

ROOT = Path(__file__).resolve().parent
STAGES = ["acquire", "prepare", "retrieve", "temporal", "evaluate"]


def read(path, sep=","):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f, delimiter=sep))


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def json_write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def yaml_write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False))


def record(path):
    return dict(path=str(path.resolve()), sha256=digest(path))


def csv_write(path, rows, sep=","):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=list(rows[0]), delimiter=sep, lineterminator="\n"
        )
        w.writeheader()
        w.writerows(rows)


def call(module, *args):
    subprocess.run(
        [sys.executable, "-m", module, *map(str, args)], cwd=ROOT, check=True
    )


def format_paths(value, workdir):
    if isinstance(value, dict):
        return {k: format_paths(v, workdir) for k, v in value.items()}
    if isinstance(value, list):
        return [format_paths(v, workdir) for v in value]
    return value.replace("{workdir}", str(workdir)) if isinstance(value, str) else value


def acquire(work, config):
    manifest = json.loads((ROOT / config["sources"]).read_text())
    for name in ["openlibrary"]:
        folder = work / "sources" / name
        folder.mkdir(parents=True, exist_ok=True)
        checks = []
        for row in manifest[name]["files"]:
            path = folder / row["filename"]
            if not path.exists():
                partial = path.with_suffix(path.suffix + ".part")
                print("Downloading", row["url"], flush=True)
                urllib.request.urlretrieve(row["url"], partial)
                verify_source(partial, row)
                partial.replace(path)
            verify_source(path, row)
            checks.append(digest(path) + "  " + path.name)
        (folder / "CHECKSUMS.sha256").write_text("\n".join(checks) + "\n")


def verify_source(path, row):
    if path.stat().st_size != int(row["size_bytes"]):
        raise ValueError(f"Source size mismatch: {path}")
    algorithm = "sha256" if "sha256" in row else "md5"
    if digest(path, algorithm) != row[algorithm]:
        raise ValueError(f"Source checksum mismatch: {path}")


def prepare(work, config):
    manifest = json.loads((ROOT / config["sources"]).read_text())
    for row in manifest["openlibrary"]["files"]:
        kind = next(
            k
            for k in ["works", "editions", "authors"]
            if "ol_dump_" + k + "_" in row["filename"]
        )
        path = work / "sources/openlibrary" / row["filename"]
        verify_source(path, row)
        call(
            "openlibrary.parse",
            "--kind",
            kind,
            "--input",
            path,
            "--out-root",
            work / "openlibrary/parsed",
            "--buckets",
            config["buckets"],
            "--expected-size",
            row["size_bytes"],
            "--input-md5",
            row["md5"],
        )
    call(
        "openlibrary.aggregate",
        "--parsed-root",
        work / "openlibrary/parsed",
        "--out-dir",
        work / "openlibrary/eligible",
        "--report",
        work / "openlibrary/corpus.md",
        "--snapshot",
        manifest["openlibrary"]["snapshot"],
        "--buckets",
        config["buckets"],
    )
    marker = json.loads((work / "openlibrary/eligible/COMPLETE.json").read_text())
    fields = [
        "eligible_works",
        "fiction_likely",
        "work_description",
        "best_description",
        "subjects",
        "author_names",
        "year_sources",
        "year_conflicts",
    ]
    json_write(work / "openlibrary/data/corpus.json", {k: marker[k] for k in fields})


def retrieve(work, config):
    environment = work / "environment.txt"
    environment.parent.mkdir(parents=True, exist_ok=True)
    environment.write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    call(
        "openlibrary.retrieval",
        "--eligible-dir",
        work / "openlibrary/eligible",
        "--semantic-config",
        ROOT / config["evaluation"],
        "--terms",
        ROOT / config["terms"],
        "--codebook",
        ROOT / "openlibrary/codebook.md",
        "--model-dir",
        work / "models/e5",
        "--model-manifest",
        work / "models/e5.json",
        "--environment-freeze",
        environment,
        "--requirements",
        ROOT / "requirements-full.txt",
        "--score-output",
        work / "openlibrary/scores.parquet",
        "--packet-dir",
        work / "openlibrary/packet",
        "--exclude-key",
        Path(config.get("references", ROOT)) / "openlibrary/data/exclusions.csv",
    )


def temporal(work, config):
    for name, module in [
        ("temporal", "openlibrary.temporal"),
        ("vocabulary", "openlibrary.term_trends"),
    ]:
        cfg = format_paths(yaml.safe_load((ROOT / config[name]).read_text()), work)
        marker = json.loads((work / "openlibrary/eligible/COMPLETE.json").read_text())
        cfg["input"]["expected_rows"] = marker["eligible_works"]
        cfg["input"]["expected_shards"] = marker["buckets"]
        cfg["output"]["directory"] = str(work / "openlibrary/data" / name)
        path = work / "config" / f"{name}.yaml"
        yaml_write(path, cfg)
        call(module, "--config", path)


VISIBLE_FIELDS = [
    "title",
    "subtitle",
    "authors",
    "first_publish_year",
    "subjects",
    "description",
]
LABEL_FIELDS = ["fiction_status", "environment_climate_relevance", "derived_status"]


def evaluate(work, config):
    from collections import Counter
    from openlibrary.evaluation import derived_status, wilson_interval

    references = Path(config.get("references", ROOT))
    evaluation_cfg = yaml.safe_load((ROOT / config["evaluation"]).read_text())
    packet = read(
        work / "openlibrary/packet" / evaluation_cfg["evaluation"]["key_filename"]
    )
    custom = config.get("annotations")
    if custom:
        labelled = read(Path(custom))
        if len(labelled) != len(packet) or len(
            {r["annotation_id"] for r in labelled}
        ) != len(packet):
            raise ValueError(
                "Annotation IDs must be unique and cover the generated packet exactly."
            )
        gold = {r["annotation_id"]: r for r in labelled}
        if set(gold) != {r["annotation_id"] for r in packet}:
            raise ValueError("Annotation IDs do not match the generated packet.")
    else:
        labelled = read(references / "openlibrary/data/evaluation.csv")
        gold = {r["work_key"]: r for r in labelled}
        if (
            len(gold) != len(labelled)
            or len(packet) != len(gold)
            or {r["work_key"] for r in packet} != set(gold)
        ):
            raise ValueError(
                "Evaluation sample changed; annotate the new blinded packet and pass --annotations. No labels transferred."
            )
    for row in packet:
        g = gold[row["annotation_id"] if custom else row["work_key"]]
        fields = (
            VISIBLE_FIELDS
            if custom
            else VISIBLE_FIELDS
            + ["evaluation_stratum", "annotation_id", "retest_annotation_id"]
        )
        if any(row[k] != g[k] for k in fields):
            raise ValueError(
                "Evaluation metadata or assignment changed; human annotation is required. No labels transferred."
            )
        status = derived_status(g["fiction_status"], g["environment_climate_relevance"])
        if g.get("derived_status") != status:
            raise ValueError(
                "Invalid derived annotation status: " + row["annotation_id"]
            )
        for k in LABEL_FIELDS:
            row[k] = g[k]
    csv_write(work / "openlibrary/data/evaluation.csv", packet)
    if custom or not config.get("verify_paper", True):
        groups = []
        for stratum in sorted({r["evaluation_stratum"] for r in packet}):
            rows = [r for r in packet if r["evaluation_stratum"] == stratum]
            counts = Counter(r["derived_status"] for r in rows)
            low, high = wilson_interval(counts["POSITIVE"], len(rows))
            groups.append(
                dict(
                    stratum=stratum,
                    records=len(rows),
                    positive=counts["POSITIVE"],
                    negative=counts["NEGATIVE"],
                    unresolved=counts["UNRESOLVED"],
                    yield_fraction=counts["POSITIVE"] / len(rows),
                    wilson_low=low,
                    wilson_high=high,
                )
            )
        csv_write(work / "results/evaluation.csv", groups)
        json_write(
            work / "results/evaluation.json",
            dict(
                records=len(packet),
                strata=dict(Counter(r["evaluation_stratum"] for r in packet)),
                judgments=dict(Counter(r["derived_status"] for r in packet)),
            ),
        )
        return
    shutil.copyfile(
        references / "openlibrary/data/retest.csv", work / "openlibrary/data/retest.csv"
    )
    # The optional GPU benchmark is regenerated by --stage llm; otherwise its
    # frozen predictions remain the input to the published benchmark statistics.
    secondary = work / "openlibrary/data/secondary.csv"
    if not secondary.exists():
        shutil.copyfile(references / "openlibrary/data/secondary.csv", secondary)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "reproduce.py"),
            "--data-root",
            str(work),
            "--output",
            str(work / "results"),
        ],
        check=True,
    )


def run_pipeline(work, config, stage="all", before=None, after=None):
    for name in STAGES if stage == "all" else [stage]:
        print("Stage:", name, flush=True)
        if before:
            before(name, work, config)
        globals()[name](work, config)
        if after:
            after(name, work, config)


def check(config):
    for key in ["sources", "evaluation", "terms", "temporal", "vocabulary"]:
        if not (ROOT / config[key]).is_file():
            raise FileNotFoundError(config[key])
    for path in sorted((ROOT / "openlibrary").glob("*.py")):
        compile(path.read_text(), str(path), "exec")
    for name in ["acquire", "prepare", "retrieve", "temporal", "evaluate"]:
        if name not in globals():
            raise ValueError("Missing pipeline stage: " + name)
    manifest = json.loads((ROOT / config["sources"]).read_text())
    for row in manifest["openlibrary"]["files"]:
        if not all(row.get(k) for k in ["filename", "url", "size_bytes", "md5"]):
            raise ValueError("Incomplete dated source manifest")
    for name in ["temporal", "vocabulary"]:
        cfg = yaml.safe_load((ROOT / config[name]).read_text())
        terms_path = ROOT / cfg["lexical_contract"]["source_terms_file"]
        terms = yaml.safe_load(terms_path.read_text())[
            cfg["lexical_contract"]["source_group"]
        ]
        if terms != cfg["lexical_contract"]["explicit_terms"]:
            raise ValueError("Terms differ from frozen lexical contract")
    cfg = yaml.safe_load((ROOT / config["evaluation"]).read_text())
    if (
        cfg["model"]["id"] != manifest["models"]["e5"]["model_id"]
        or cfg["model"]["revision"] != manifest["models"]["e5"]["revision"]
    ):
        raise ValueError("E5 configuration differs from pinned model")
    secondary_cfg = ROOT / "openlibrary/config/secondary.yaml"
    secondary = yaml.safe_load(secondary_cfg.read_text())
    for task in ["fiction", "relevance", "evidence"]:
        if not (ROOT / secondary["prompts"][task]).is_file():
            raise FileNotFoundError(secondary["prompts"][task])
    for name in [
        "requirements.txt",
        "requirements-full.txt",
        "requirements-secondary.txt",
    ]:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(name)
    print(
        "PASS: full-workflow code, sources and configurations resolve; no scan or inference executed."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "pipeline.json")
    p.add_argument("--stage", choices=STAGES + ["all", "llm"], default="all")
    p.add_argument("--check", action="store_true")
    p.add_argument(
        "--annotations",
        type=Path,
        help="Human labels for a new blinded packet; evaluate without paper-result assertions",
    )
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    if a.annotations:
        cfg["annotations"] = str(a.annotations.resolve())
    if a.check:
        check(cfg)
    else:
        import socket

        if (
            a.stage != "evaluate"
            and "roihu" in socket.gethostname().lower()
            and not os.environ.get("SLURM_JOB_ID")
        ):
            p.error(
                "Full stages require a Slurm allocation on Roihu; use --check for local validation."
            )
        work = a.workdir.resolve()
        if work == ROOT:
            p.error("Use a separate working directory, not the package root.")
        work.mkdir(parents=True, exist_ok=True)
        if a.stage == "llm":
            from openlibrary.secondary import run

            run(work)
        else:
            run_pipeline(work, cfg, a.stage)
