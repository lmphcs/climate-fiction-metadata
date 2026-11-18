"""Compact-count temporal statistics; no corpus or Arrow dependency."""

from __future__ import annotations
import math
from typing import Any, Mapping, Sequence
import numpy as np


def rate(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else 10_000.0 * numerator / denominator


def period_rows(
    annual: Sequence[Mapping[str, Any]], periods: Sequence[Sequence[int]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = sorted({(row["population"], row["field"]) for row in annual})
    for population, field in groups:
        selected = [
            row
            for row in annual
            if row["population"] == population and row["field"] == field
        ]
        for start, stop in periods:
            block = [row for row in selected if int(start) <= row["year"] <= int(stop)]
            works = sum(int(row["works"]) for row in block)
            available = sum(int(row["field_available"]) for row in block)
            explicit = sum(int(row["explicit_count"]) for row in block)
            rows.append(
                {
                    "period": f"{start}-{stop}",
                    "first_year": int(start),
                    "last_year": int(stop),
                    "population": population,
                    "field": field,
                    "works": works,
                    "field_available": available,
                    "explicit_count": explicit,
                    "rate_per_10000_works": rate(explicit, works),
                    "rate_per_10000_available": rate(explicit, available),
                    "coverage_percent": (
                        None if works == 0 else 100.0 * available / works
                    ),
                }
            )
    return rows


def grouped_binomial_trend(
    years: np.ndarray, successes: np.ndarray, totals: np.ndarray
) -> dict[str, Any]:
    keep = totals > 0
    years = years[keep].astype(float)
    successes = successes[keep].astype(float)
    totals = totals[keep].astype(float)
    if len(years) < 3 or successes.sum() <= 0 or successes.sum() >= totals.sum():
        return {
            "status": "not_estimable",
            "years": int(len(years)),
            "successes": int(successes.sum()),
            "trials": int(totals.sum()),
        }
    center = float(np.average(years, weights=totals))
    x = (years - center) / 10.0
    design = np.column_stack([np.ones(len(x)), x])
    initial_p = np.clip(successes.sum() / totals.sum(), 1e-9, 1 - 1e-9)
    beta = np.array([math.log(initial_p / (1 - initial_p)), 0.0])
    for _iteration in range(100):
        eta = np.clip(design @ beta, -35.0, 35.0)
        probabilities = 1.0 / (1.0 + np.exp(-eta))
        gradient = design.T @ (successes - totals * probabilities)
        weights = totals * probabilities * (1.0 - probabilities)
        information = design.T @ (weights[:, None] * design)
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            return {
                "status": "not_estimable",
                "years": int(len(years)),
                "successes": int(successes.sum()),
                "trials": int(totals.sum()),
            }
        beta += step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    eta = np.clip(design @ beta, -35.0, 35.0)
    probabilities = 1.0 / (1.0 + np.exp(-eta))
    weights = totals * probabilities * (1.0 - probabilities)
    information = design.T @ (weights[:, None] * design)
    covariance = np.linalg.inv(information)
    standard_error = math.sqrt(float(covariance[1, 1]))
    z_value = float(beta[1] / standard_error)
    p_value = math.erfc(abs(z_value) / math.sqrt(2.0))
    odds_ratio = math.exp(float(beta[1]))
    lower = math.exp(float(beta[1]) - 1.96 * standard_error)
    upper = math.exp(float(beta[1]) + 1.96 * standard_error)

    def fitted(target_year: float) -> float:
        value = beta[0] + beta[1] * ((target_year - center) / 10.0)
        return 10_000.0 / (1.0 + math.exp(-max(-35.0, min(35.0, value))))

    return {
        "status": "estimated",
        "years": int(len(years)),
        "successes": int(successes.sum()),
        "trials": int(totals.sum()),
        "odds_ratio_per_decade": odds_ratio,
        "wald_95_percent_low": lower,
        "wald_95_percent_high": upper,
        "wald_z": z_value,
        "wald_p_two_sided": p_value,
        "fitted_rate_per_10000_first_year": fitted(float(years.min())),
        "fitted_rate_per_10000_last_year": fitted(float(years.max())),
        "model": "grouped_binomial_logit_on_calendar_year",
    }


def trend_rows(
    annual: Sequence[Mapping[str, Any]], tests: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for specification in tests:
        selected = [
            row
            for row in annual
            if row["population"] == specification["population"]
            and row["field"] == specification["field"]
            and row["year"] <= int(specification["through_year"])
        ]
        denominator_field = str(specification["denominator"])
        if denominator_field not in {"works", "field_available"}:
            raise ValueError(f"Unsupported trend denominator: {denominator_field}")
        estimate = grouped_binomial_trend(
            np.asarray([row["year"] for row in selected], dtype=np.int64),
            np.asarray([row["explicit_count"] for row in selected], dtype=np.int64),
            np.asarray([row[denominator_field] for row in selected], dtype=np.int64),
        )
        output.append({**dict(specification), **estimate})
    return output
