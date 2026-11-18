#!/usr/bin/env python3
"""Regenerate compact final tables, models and figures; fail on a numerical mismatch."""

from pathlib import Path
from collections import Counter
import argparse
import csv
import importlib
import json
import math
import yaml

from openlibrary.common import atomic_csv, atomic_json, file_record
from openlibrary.evaluation import derived_status, route_signature, wilson_interval
from openlibrary.statistics import period_rows, trend_rows

ROOT = Path(__file__).resolve().parent
DATA = ROOT


def read(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def load(path):
    return json.loads(Path(path).read_text())


def table(path, records):
    atomic_csv(path, records, list(records[0]))


def verify_subset(observed, expected, path="result"):
    if isinstance(expected, dict):
        for key, value in expected.items():
            verify_subset(observed[key], value, path + "." + key)
    elif observed != expected:
        raise ValueError(f"{path}: observed {observed!r}, expected {expected!r}")


def evaluation(out):
    rows = read(DATA / "openlibrary/data/evaluation.csv")
    for r in rows:
        assert route_signature(r) == r["evaluation_stratum"]
        assert (
            derived_status(r["fiction_status"], r["environment_climate_relevance"])
            == r["derived_status"]
        )
    assert len(rows) == len({r["work_key"] for r in rows}) == 100
    expected = load(ROOT / "openlibrary/data/expected.json")
    totals = dict(
        records=len(rows),
        fiction_status=dict(Counter(r["fiction_status"] for r in rows)),
        derived_status=dict(Counter(r["derived_status"] for r in rows)),
        positive_composition={
            "climate_explicit": sum(
                r["derived_status"] == "POSITIVE"
                and r["environment_climate_relevance"] == "CLIMATE_EXPLICIT"
                for r in rows
            ),
            "implicit_or_environmental": sum(
                r["derived_status"] == "POSITIVE"
                and r["environment_climate_relevance"] == "IMPLICIT_OR_ENVIRONMENTAL"
                for r in rows
            ),
        },
    )
    verify_subset(totals, expected["overall"])
    groups = []
    for s, e in expected["strata"].items():
        group = [r for r in rows if r["evaluation_stratum"] == s]
        n = len(group)
        c = Counter(r["derived_status"] for r in group)
        forms = Counter(r["fiction_status"] for r in group)
        observed = dict(
            records=n,
            positive_count=c["POSITIVE"],
            fiction_status=dict(forms),
            derived_status=dict(c),
        )
        verify_subset(observed, e, s)
        low, high = wilson_interval(c["POSITIVE"], n)
        groups.append(
            dict(
                stratum=s,
                records=n,
                climate_related_fiction=c["POSITIVE"],
                unrelated_fiction=sum(
                    r["fiction_status"] == "FICTION"
                    and r["derived_status"] == "NEGATIVE"
                    for r in group
                ),
                nonfiction_or_other=forms["NONFICTION"] + forms["OTHER_LITERARY"],
                unclear=c["UNRESOLVED"],
                yield_fraction=c["POSITIVE"] / n,
                wilson_low=low,
                wilson_high=high,
            )
        )
    retest = {r["annotation_id"]: r for r in read(DATA / "openlibrary/data/retest.csv")}
    paired = [
        (r, retest[r["retest_annotation_id"]])
        for r in rows
        if r["retest_annotation_id"]
    ]
    exact = sum(
        all(a[k] == b[k] for k in ["fiction_status", "environment_climate_relevance"])
        for a, b in paired
    )
    assert f"{exact}/{len(paired)}" == expected["retest_exact"]
    table(out / "evaluation.csv", groups)
    table(
        out / "queries.csv",
        yaml.safe_load((ROOT / "openlibrary/config/evaluation.yaml").read_text())[
            "queries"
        ],
    )
    corpus = load(DATA / "openlibrary/data/corpus.json")
    table(
        out / "year_provenance.csv",
        [dict(decision=k, works=v) for k, v in corpus["year_sources"].items()],
    )
    table(
        out / "coverage.csv",
        [
            dict(
                field=k,
                works=corpus[k],
                percent=100 * corpus[k] / corpus["eligible_works"],
            )
            for k in [
                "best_description",
                "work_description",
                "subjects",
                "author_names",
                "fiction_likely",
            ]
        ],
    )
    examples = [
        r
        for r in rows
        if r["title"].casefold()
        in {"tore and the town on thin ice", "drought", "growing up in the south"}
    ]
    assert len(examples) == 3
    table(
        out / "annotation_examples.csv",
        [
            {
                k: r[k]
                for k in [
                    "annotation_id",
                    "work_key",
                    "title",
                    "description",
                    "fiction_status",
                    "environment_climate_relevance",
                ]
            }
            for r in examples
        ],
    )
    semantic = [r for r in rows if r["evaluation_stratum"] == "semantic_only"]
    assert all(
        r["fiction_likely"] == "True" and r["sampling_tier"] == "fiction_clean"
        for r in semantic
    )
    table(out / "semantic_sample.csv", semantic)
    return dict(**totals, retest_exact=f"{exact}/{len(paired)}", corpus=corpus)


def secondary(out):
    rows = read(DATA / "openlibrary/data/secondary.csv")
    labelled = [r for r in rows if r["gold_fiction_status"]]
    if not rows or not labelled:
        raise ValueError(
            "Secondary benchmark requires predictions and labelled development cases."
        )
    positive = lambda r: r["gold_fiction_status"] == "FICTION" and r[
        "gold_relevance"
    ] in {"CLIMATE_EXPLICIT", "IMPLICIT_OR_ENVIRONMENTAL"}
    pred = lambda r: r["model_positive"] == "True"
    yes = [r for r in labelled if positive(r)]
    no = [r for r in labelled if not positive(r)]
    relevance = [
        r for r in labelled if r["gold_fiction_status"] in {"FICTION", "UNCLEAR"}
    ]
    sorted_rows = sorted(
        labelled, key=lambda r: (-float(r["target_rank_score"]), r["work_key"])
    )
    hits = 0
    ap = 0.0
    for rank, r in enumerate(sorted_rows, 1):
        if positive(r):
            hits += 1
            ap += hits / rank
    summary = dict(
        rows=len(rows),
        model_positive_rows=sum(pred(r) for r in rows),
        grounded_model_positive_rows=sum(
            pred(r) and r["evidence_status"] == "grounded" for r in rows
        ),
        wave_b=dict(
            rows=len(labelled),
            fiction_exact_agreement=sum(
                r["fiction_status"] == r["gold_fiction_status"] for r in labelled
            )
            / len(labelled),
            relevance_rows=len(relevance),
            relevance_exact_agreement=sum(
                r["relevance_label_raw"] == r["gold_relevance"] for r in relevance
            )
            / len(relevance),
            binary_balanced_accuracy=(
                sum(pred(r) for r in yes) / len(yes)
                + sum(not pred(r) for r in no) / len(no)
            )
            / 2,
            average_precision=ap / len(yes),
        ),
    )
    expected = load(ROOT / "openlibrary/data/secondary_expected.json")
    for key in ["rows", "model_positive_rows", "grounded_model_positive_rows"]:
        assert summary[key] == expected[key]
    for key, value in summary["wave_b"].items():
        assert math.isclose(value, expected["wave_b"][key], rel_tol=1e-12), (key, value)
    atomic_json(out / "secondary_benchmark.json", summary)
    return summary


def temporal(out):
    data = DATA / "openlibrary/data/temporal"
    annual = read(data / "annual_counts.csv")
    for r in annual:
        for key in ["year", "works", "field_available", "explicit_count"]:
            r[key] = int(r[key])
        for key in [
            "rate_per_10000_works",
            "rate_per_10000_available",
            "coverage_percent",
        ]:
            r[key] = float(r[key]) if r[key] else None
    config = yaml.safe_load((ROOT / "openlibrary/config/temporal.yaml").read_text())
    periods = period_rows(annual, config["scope"]["periods"])
    estimates = trend_rows(annual, config["trend_tests"])
    expected = load(ROOT / "openlibrary/data/expected.json")["trends"]
    assert len(estimates) == len(
        expected
    ), "Temporal model count differs from the final study"
    for a, b in zip(estimates, expected):
        assert a["id"] == b["id"]
        for key, value in b.items():
            if isinstance(a.get(key), (int, float)) and value:
                assert math.isclose(
                    a[key], float(value), rel_tol=1e-8, abs_tol=1e-12
                ), (a["id"], key)
    table(out / "period_counts.csv", periods)
    table(out / "trends.csv", estimates)
    for module in ["plot_temporal", "plot_fiction", "plot_coverage"]:
        plot = importlib.import_module("openlibrary." + module)
        plot.render(plot.load_bound_rows(data), out / "figures")
    plot = importlib.import_module("openlibrary.plot_vocabulary")
    plot.render(
        plot.load_verified_rows(DATA / "openlibrary/data/vocabulary"), out / "figures"
    )
    # Keep PDF and PNG exports; the manuscript uses both formats.
    for p in (out / "figures").glob("*absolute*"):
        p.unlink()
    all_rows = [
        r for r in annual if r["population"] == "all_works" and r["field"] == "any"
    ]
    assert sum(r["works"] for r in all_rows) == 13754720
    assert sum(r["explicit_count"] for r in all_rows) == 16791
    series = [
        dict(
            year=r["year"],
            works=r["works"],
            matches=r["explicit_count"],
            percent=100 * r["explicit_count"] / r["works"],
        )
        for r in all_rows
    ]
    table(out / "annual_series.csv", series)
    return dict(
        works=sum(r["works"] for r in all_rows),
        explicit_matches=sum(r["explicit_count"] for r in all_rows),
        peak_year=max(series, key=lambda r: r["percent"])["year"],
        trend_estimates=estimates,
    )


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    results = dict(
        openlibrary=evaluation(out), secondary=secondary(out), temporal=temporal(out)
    )
    atomic_json(out / "statistics.json", results)
    print(f"PASS: Open Library results written to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "results")
    p.add_argument(
        "--data-root",
        type=Path,
        default=ROOT,
        help="Root of compact or fully regenerated input data",
    )
    args = p.parse_args()
    DATA = args.data_root.resolve()
    run(args.output.resolve())
