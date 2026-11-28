#!/usr/bin/env python3
r"""
pilot25b - the census of pilot25 read at the level of the seed, not the call.

Why this script exists.  pilot25 pools 1,200-step runs into one hazard per cell by
treating every call as an exposed unit.  Calls inside a run share weights and momentum,
so they are not independent, and the call-level exact test it reports is an exposure
statistic.  The unit that *is* independent between repetitions is the run: 25 seeds per
cell.  This script computes, from the same runs and no new training:

  1. per cell, the run rate (seeds diverged / seeds) with an exact Clopper-Pearson
     interval, next to the call hazard with a Wilson interval for comparison;
  2. a cluster bootstrap that resamples *seeds* with replacement and recomputes the
     pooled call hazard (and the run rate), so that the within-run correlation enters
     the interval; the ratio of its standard error to the binomial standard error is
     the design effect of the clustering;
  3. seed-level tests between conditions of one task: Fisher's exact test on the
     2x2 table of (diverged, survived) x (two conditions), which is the run-level
     statement the paper can make;
  4. a seed-level comparison of *when* a run fails between the two tasks (permutation
     test on the difference of median steps to divergence), which is defined where the
     run counts are tied at 25/25;
  5. the same quantity the paper quotes as a call-level p, recomputed both ways, so
     the two readings can be compared.

Reads:  ../../data/pilot25_results.json   (or ../results/ in the packaged layout)
Writes: ../../data/pilot25b_seedlevel_results.json
Deterministic: fixed RNG seeds, no wall-clock field.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))


def find(name):
    for cand in (os.path.join(HERE, name),
                 os.path.join(HERE, "..", "results", name),
                 os.path.join(HERE, "results", name)):
        if os.path.exists(cand):
            return os.path.abspath(cand)
    sys.exit(f"cannot find {name}")


OUT = os.path.join(HERE, "pilot25b_seedlevel_results.json")
B = 4000           # bootstrap resamples
NPERM = 40000      # permutations
RNG = 20261128     # fixed: the artifact must be a function of the data alone

src = json.load(open(find("pilot25_results.json")))
runs = src["runs"]


def group(task, cond):
    out = []
    for key, r in runs.items():
        if r["task"] == task and r["cond"] == cond:
            out.append(r)
    out.sort(key=lambda r: r["seed"])
    return out


def cp_interval(k, n, alpha=0.05):
    """Exact Clopper-Pearson interval for k successes in n trials."""
    if n == 0:
        return [None, None]

    def cdf(x, p):
        return stats.binom.cdf(x, n, p)

    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return [float(lo), float(hi)]


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [float((c - h) / den), float((c + h) / den)]


def hazard_of(rows):
    calls = sum(r["calls"] for r in rows)
    events = sum(1 for r in rows if r["div_step"] is not None)
    return events, calls, (events / calls if calls else None)


def cluster_bootstrap_hazard(rows, rng):
    """Resample seeds (runs) with replacement; pool calls and events."""
    n = len(rows)
    idx = rng.integers(0, n, size=(B, n))
    vals, rates = [], []
    for b in range(B):
        calls = 0
        events = 0
        for i in idx[b]:
            r = rows[i]
            calls += r["calls"]
            events += 1 if r["div_step"] is not None else 0
        vals.append(events / calls if calls else np.nan)
        rates.append(events / n)
    return np.asarray(vals), np.asarray(rates)


def pct(v, lo=2.5, hi=97.5):
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [None, None]
    return [float(np.percentile(v, lo)), float(np.percentile(v, hi))]


def cell_stats(task, cond):
    rows = group(task, cond)
    events, calls, haz = hazard_of(rows)
    rng = np.random.default_rng(RNG)
    haz_boot, rate_boot = cluster_bootstrap_hazard(rows, rng)
    # design effect: cluster-bootstrap sd over the binomial sd the Wilson interval uses
    p = haz if haz is not None else 0.0
    binom_sd = np.sqrt(max(p * (1 - p), 1e-30) / calls) if calls else None
    boot_sd = float(np.nanstd(haz_boot, ddof=1)) if calls else None
    ded = (boot_sd / binom_sd) if (binom_sd and boot_sd) else None
    steps = [r["div_step"] for r in rows if r["div_step"] is not None]
    return {
        "task": task,
        "cond": cond,
        "runs": len(rows),
        "diverged": events,
        "run_rate": events / len(rows),
        "run_rate_cp95": cp_interval(events, len(rows)),
        "calls": calls,
        "hazard_per_call": haz,
        "hazard_wilson95": wilson(events, calls),
        "hazard_seed_boot95": pct(haz_boot),
        "hazard_seed_boot_sd": boot_sd,
        "binomial_sd": float(binom_sd) if binom_sd else None,
        "design_effect": float(ded) if ded else None,
        "run_rate_seed_boot95": pct(rate_boot),
        "steps_to_div_median": float(np.median(steps)) if steps else None,
        "steps_to_div_iqr": ([float(np.percentile(steps, 25)), float(np.percentile(steps, 75))]
                             if steps else None),
        "steps_to_div": sorted(steps),
    }


def fisher(r1, r2):
    a = [[r1["diverged"], r1["runs"] - r1["diverged"]],
         [r2["diverged"], r2["runs"] - r2["diverged"]]]
    _, p = stats.fisher_exact(a)
    return float(p)


def rate_diff_boot(r1, r2):
    rng = np.random.default_rng(RNG + 1)
    d1 = np.array([1 if r["div_step"] is not None else 0 for r in group(r1["task"], r1["cond"])])
    d2 = np.array([1 if r["div_step"] is not None else 0 for r in group(r2["task"], r2["cond"])])
    diffs = []
    for _ in range(B):
        a = d1[rng.integers(0, d1.size, d1.size)].mean()
        b = d2[rng.integers(0, d2.size, d2.size)].mean()
        diffs.append(a - b)
    return pct(np.asarray(diffs))


def hazard_ratio_boot(c1, c2):
    rng = np.random.default_rng(RNG + 2)
    out = []
    r1 = group(c1["task"], c1["cond"])
    r2 = group(c2["task"], c2["cond"])
    for _ in range(B):
        i1 = rng.integers(0, len(r1), len(r1))
        i2 = rng.integers(0, len(r2), len(r2))
        e1 = sum(1 for i in i1 if r1[i]["div_step"] is not None)
        c1_ = sum(r1[i]["calls"] for i in i1)
        e2 = sum(1 for i in i2 if r2[i]["div_step"] is not None)
        c2_ = sum(r2[i]["calls"] for i in i2)
        if c1_ and c2_ and e2:
            out.append((e1 / c1_) / (e2 / c2_))
        elif c1_ and c2_ and e1:
            out.append(np.inf)
    return pct(np.asarray([o for o in out if np.isfinite(o)]))


def permutation_median_shift(a, b, statistic):
    rng = np.random.default_rng(RNG + 3)
    pool = np.array(a + b, float)
    na = len(a)
    obs = statistic(np.array(a, float), np.array(b, float))
    hits = 0
    for _ in range(NPERM):
        perm = rng.permutation(pool)
        s = statistic(perm[:na], perm[na:])
        if abs(s) >= abs(obs) - 1e-12:
            hits += 1
    return float(obs), float((hits + 1) / (NPERM + 1))


def main():
    cells = {}
    for task, conds in (("digits", ("e5m2", "e5m2_s1035", "e5m2_clamp")),
                        ("diabetes", ("e5m2", "e5m2_s1035", "e5m2_clamp"))):
        for cond in conds:
            cells[f"{task}_{cond}"] = cell_stats(task, cond)

    print("=" * 104)
    print("pilot25b - the census read at the level of the seed")
    print("=" * 104)
    print(f"  {'cell':<26}{'div/runs':>10}{'run rate':>10}{'CP95':>20}"
          f"{'hazard':>11}{'seed-boot 95':>22}{'DEFF':>7}")
    for k, c in cells.items():
        cp = c["run_rate_cp95"]
        hb = c["hazard_seed_boot95"]
        print(f"  {k:<26}{c['diverged']:>4}/{c['runs']:<5}{c['run_rate']:>10.2f}"
              f"{f'[{cp[0]:.2f},{cp[1]:.2f}]':>20}"
              f"{(c['hazard_per_call'] or 0):>11.3e}"
              f"{(f'[{hb[0]:.2e},{hb[1]:.2e}]' if hb[0] is not None else '--'):>22}"
              f"{(c['design_effect'] or 0):>7.2f}")

    pairs = {}
    for task in ("digits", "diabetes"):
        for a, b in (("e5m2", "e5m2_s1035"), ("e5m2", "e5m2_clamp"),
                     ("e5m2_s1035", "e5m2_clamp")):
            ka, kb = f"{task}_{a}", f"{task}_{b}"
            row = {"task": task, "a": a, "b": b,
                   "rates": [cells[ka]["diverged"], cells[ka]["runs"],
                             cells[kb]["diverged"], cells[kb]["runs"]],
                   "fisher_p": fisher(cells[ka], cells[kb]),
                   "rate_diff_boot95": rate_diff_boot(cells[ka], cells[kb]),
                   "hazard_ratio_boot95": hazard_ratio_boot(cells[ka], cells[kb])}
            if cells[kb]["hazard_per_call"]:
                row["hazard_ratio"] = (cells[ka]["hazard_per_call"] /
                                       cells[kb]["hazard_per_call"])
            else:
                row["hazard_ratio"] = None
            pairs[f"{task}:{a}_vs_{b}"] = row
            print(f"  {task:<9}{a:>11} vs {b:<12} Fisher p {row['fisher_p']:.3e}"
                  f"   rate diff {row['rate_diff_boot95'][0]:+.2f}"
                  f"..{row['rate_diff_boot95'][1]:+.2f}")

    # cross-task: counts tied at 25/25, so compare the *time* to failure at the seed level
    dg = cells["digits_e5m2"]["steps_to_div"]
    db = cells["diabetes_e5m2"]["steps_to_div"]
    obs, p_perm = permutation_median_shift(dg, db, lambda x, y: float(np.median(y) - np.median(x)))
    u = stats.mannwhitneyu(dg, db, alternative="two-sided")
    cross = {
        "digits_e5m2": {"diverged": cells["digits_e5m2"]["diverged"],
                        "runs": cells["digits_e5m2"]["runs"],
                        "steps_median": cells["digits_e5m2"]["steps_to_div_median"]},
        "diabetes_e5m2": {"diverged": cells["diabetes_e5m2"]["diverged"],
                          "runs": cells["diabetes_e5m2"]["runs"],
                          "steps_median": cells["diabetes_e5m2"]["steps_to_div_median"]},
        "fisher_p_counts": fisher(cells["digits_e5m2"], cells["diabetes_e5m2"]),
        "median_shift_steps_diabetes_minus_digits": obs,
        "permutation_p_median_shift": p_perm,
        "mannwhitney_p": float(u.pvalue),
        "mannwhitney_u": float(u.statistic),
        "hazard_ratio_diabetes_over_digits": (
            cells["diabetes_e5m2"]["hazard_per_call"] /
            cells["digits_e5m2"]["hazard_per_call"]),
        "hazard_ratio_boot95": hazard_ratio_boot(cells["diabetes_e5m2"],
                                                 cells["digits_e5m2"]),
        "hazard_ratio_seed_boot95": pct(
            np.asarray([a / b for a, b in zip(
                cluster_bootstrap_hazard(group("diabetes", "e5m2"),
                                         np.random.default_rng(RNG + 4))[0],
                cluster_bootstrap_hazard(group("digits", "e5m2"),
                                         np.random.default_rng(RNG + 4))[0])])),
    }
    print(f"\n  digits vs diabetes at E5M2, 1.01: counts {cross['digits_e5m2']['diverged']}/25 vs"
          f" {cross['diabetes_e5m2']['diverged']}/25 (Fisher p {cross['fisher_p_counts']:.2f}),"
          f" median steps {cross['digits_e5m2']['steps_median']:.0f} vs"
          f" {cross['diabetes_e5m2']['steps_median']:.0f}"
          f" (permutation p {p_perm:.2e}; Mann-Whitney p {u.pvalue:.2e})")
    print(f"  call-level hazard ratio {cross['hazard_ratio_diabetes_over_digits']:.1f}x,"
          f" seed bootstrap 95% [{cross['hazard_ratio_boot95'][0]:.1f},"
          f" {cross['hazard_ratio_boot95'][1]:.1f}] of the ratio")

    out = {
        "source": "pilot25_results.json",
        "unit": "seed (one 1,200-step run); a call is one application of the composite "
                "to one hidden weight matrix",
        "bootstrap": {"resamples": B, "seed": RNG,
                      "resamples_unit": "run (cluster), pooled calls and events"},
        "permutation": {"draws": NPERM, "seed": RNG + 3,
                        "statistic": "median steps to divergence, diabetes minus digits"},
        "cells": cells,
        "pairs": pairs,
        "cross_task": cross,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=False)
        f.write("\n")
    print(f"\n  wrote {os.path.basename(OUT)}")


if __name__ == "__main__":
    main()
