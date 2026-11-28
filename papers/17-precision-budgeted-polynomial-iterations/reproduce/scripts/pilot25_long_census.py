#!/usr/bin/env python3
"""
Pilot 25 - the long divergence census: 1,200 steps x 25 seeds, digits and diabetes.

Pilot 23 censused 400 steps x 11 seed-runs per condition and reported a boundary as a
count of runs (E5M2 at the shipped safety s = 1.01: 6 of 11; digits 2/5, residual 1/3,
diabetes 3/3).  Eleven runs cannot separate two tasks whose failure probabilities differ
by a factor of three, and 400 steps cannot tell whether a run that survived is safe or
merely slow: the escape mechanism of pilot 23 is a single call landing past a window edge,
so the risk is per call, not per run, and the run-level count depends on how many calls the
run makes.

This census fixes both: three conditions, two tasks, 25 seeds each, 1,200 training steps
(4,800 Muon calls per digits run, 3,600 per diabetes run), and the estimator is

    hazard per call  =  (calls that diverged) / (calls executed),

with a 95% Wilson interval on that binomial proportion, censored at 1,200 steps.  A run
that has not diverged by then contributes all of its calls and no event (right censoring),
which is what makes the per-call rate estimable from unequal exposures.  The difference
between the two tasks is tested at run level (Fisher's exact test) and at call level
(exact conditional binomial test given the pooled events, since the exposures differ).

Conditions: e5m2 (s = 1.01, the shipped point), e5m2_s1035 (the known cure) and e5m2 with
the state clamped into its certified window (the mechanism counterfactual of pilot 23).
Kernel, data, model and the divergence definition are pilot 23's, unchanged, so a 1,200-step
run restricted to its first 400 steps reproduces pilot 23's census (checked in stage
`check`).

The script is resumable and parallel; no wall-clock field is written, so the JSON is a
function of the seeds alone and reruns byte for byte.

    python3 pilot25_long_census.py --task digits --cond e5m2 --seeds 0-11 --workers 2
    python3 pilot25_long_census.py --analyse
"""
import argparse
import json
import math
import os
import sys
from math import comb, lgamma

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pilot21_loss_phase_and_scheduler as p21  # noqa: E402
from pilot21_loss_phase_and_scheduler import (  # noqa: E402
    CUSH, ELL0, P_BITS, exact_polar, get_data, layers_of, loss_and_grads, norm_est,
    preset, quant, window_err,
)
from pilot7_pert_aware_design import pe_coeffs  # noqa: E402

RES = os.path.join(HERE, "pilot25_results.json")
STEPS = 1200
SEEDS = tuple(range(25))
TASKS = ("digits", "diabetes")
CONDS = {"e5m2": dict(), "e5m2_s1035": dict(), "e5m2_clamp": dict(clamp=True)}
DIV_THR = 1e3
Z95 = 1.959963984540054


def load():
    if os.path.exists(RES):
        with open(RES) as f:
            return json.load(f)
    return {"runs": {}}


def save(R):
    R["census"], R["blocks"], R["tests"] = analyse(R["runs"])
    tmp = RES + ".tmp"
    with open(tmp, "w") as f:
        json.dump(R, f, indent=1, sort_keys=True)
    os.replace(tmp, RES)


# ---------------------------------------------------------------- statistics
def wilson(k, n, z=Z95):
    """95% Wilson score interval for k successes in n trials."""
    if n <= 0:
        return (None, None)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fisher_p(a, b, c, d):
    """Two-sided Fisher exact p for [[a, b], [c, d]] (sum of tables no likelier)."""
    n = a + b + c + d
    r1, r2, c1 = a + b, c + d, a + c
    if n == 0 or r1 == 0 or r2 == 0 or c1 == 0 or c1 == n:
        return None
    def pr(x):
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)
    p0 = pr(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    s = 0.0
    for x in range(lo, hi + 1):
        p = pr(x)
        if p <= p0 * (1 + 1e-12):
            s += p
    return min(1.0, s)


def binom_pmf(k, n, p):
    if p <= 0.0 or p >= 1.0:
        return 1.0 if k == (n if p >= 1 else 0) else 0.0
    return math.exp(lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
                    + k * math.log(p) + (n - k) * math.log1p(-p))


def binom_exact_p(k1, n1, k2, n2):
    """Exact conditional binomial test: given k1+k2 events, k1 ~ Bin(k1+k2, n1/(n1+n2))."""
    k, n, p = k1 + k2, n1 + n2, n1 / (n1 + n2)
    if k == 0 or k == n:
        return None
    p0 = binom_pmf(k1, k, p)
    s = 0.0
    for x in range(0, k + 1):
        q = binom_pmf(x, k, p)
        if q <= p0 * (1 + 1e-12):
            s += q
    return min(1.0, s)


# ---------------------------------------------------------------- one run
def top_est(X, iters=4):
    """Largest singular value by power iteration (the estimate pilot 23 clamps with)."""
    m, n = X.shape
    if n <= m:
        v = np.ones(n) / np.sqrt(n)
        for _ in range(iters):
            w = X @ v
            w /= np.linalg.norm(w) + 1e-300
            v = X.T @ w
            v /= np.linalg.norm(v) + 1e-300
        return float(np.linalg.norm(X @ v))
    u = np.ones(m) / np.sqrt(m)
    for _ in range(iters):
        w = X.T @ u
        w /= np.linalg.norm(w) + 1e-300
        u = X @ w
        u /= np.linalg.norm(u) + 1e-300
    return float(np.linalg.norm(X.T @ u))


def run_census(task, cond, seed, steps=STEPS, K=6, lr=0.01):
    """Pilot 23's training loop, instrumented only with the divergence record."""
    kw = CONDS[cond]
    clamp = bool(kw.get("clamp"))
    Xtr, ytr, Xva, yva = get_data(task, seed)
    rs = np.random.RandomState(1000 + seed)
    Ws, W5, dims = layers_of(task, rs)
    b5 = np.zeros(W5.shape[0])
    ms = [np.zeros_like(W) for W in Ws]
    aW5 = np.zeros_like(W5); vW5 = np.zeros_like(W5)
    ab5 = np.zeros_like(b5); vb5 = np.zeros_like(b5)
    base = "e5m2" if cond == "e5m2_clamp" else cond
    co, cast_fmt, step_fmts, safety = preset(base, K)
    _, win = pe_coeffs(ELL0, K, CUSH, safety)
    div, reason, div_step, div_layer, calls = False, "completed", None, None, 0
    t = -1
    for t in range(steps):
        idx = rs.choice(len(ytr), min(256, len(ytr)), replace=False)
        loss, gs, dW5, db5 = loss_and_grads(task, Ws, W5, b5, Xtr[idx], ytr[idx])
        for l, (W, g) in enumerate(zip(Ws, gs)):
            ms[l] = 0.95 * ms[l] + g
            upd = g + 0.95 * ms[l]
            nev = norm_est(upd)
            X = quant(upd / nev, cast_fmt)
            for tt, ((a, b, c), fmt) in enumerate(zip(co, step_fmts)):
                A = X @ X.T
                B = quant(b * A + c * (A @ A), fmt)
                X = quant(a * X + B @ X, fmt)
                if clamp:
                    ub = win[min(tt + 1, len(win) - 1)][1]
                    tp = top_est(X)
                    if tp > ub:
                        X = X * (ub / tp)
            calls += 1
            step_lr = lr * 0.2 * np.sqrt(max(W.shape))
            W -= step_lr * X
            if not np.all(np.isfinite(W)) or np.abs(W).max() > DIV_THR:
                div, reason, div_step, div_layer = True, "weight_blowup", t, l
                break
        if div:
            break
        aW5 = 0.9 * aW5 + 0.1 * dW5; vW5 = 0.999 * vW5 + 0.001 * dW5 ** 2
        W5 -= 3e-3 * (aW5 / (1 - 0.9 ** (t + 1))) / (np.sqrt(vW5 / (1 - 0.999 ** (t + 1))) + 1e-8)
        ab5 = 0.9 * ab5 + 0.1 * db5; vb5 = 0.999 * vb5 + 0.001 * db5 ** 2
        b5 -= 3e-3 * (ab5 / (1 - 0.9 ** (t + 1))) / (np.sqrt(vb5 / (1 - 0.999 ** (t + 1))) + 1e-8)
        if not np.isfinite(loss):
            div, reason, div_step, div_layer = True, "nonfinite_loss", t, len(Ws) - 1
            break
    return {"task": task, "cond": cond, "seed": seed, "safety": float(safety),
            "layers": len(Ws), "steps_run": (div_step + 1 if div else t + 1),
            "finished": not div, "stop_reason": reason,
            "div_step": div_step, "div_layer": div_layer, "calls": calls,
            "final_loss": float(loss)}


def _work(args):
    return run_census(*args)


# ---------------------------------------------------------------- the census
def collect(tasks, conds, seeds, workers):
    R = load()
    todo = [(tk, cd, sd) for tk in tasks for cd in conds for sd in seeds
            if f"{tk}_{cd}_s{sd}" not in R["runs"]]
    print(f"  runs to do: {len(todo)} of {len(tasks) * len(conds) * len(seeds)}"
          f" (workers {workers})", flush=True)
    if not todo:
        return R
    if workers <= 1:
        for i, a in enumerate(todo):
            r = run_census(*a)
            R["runs"][f"{r['task']}_{r['cond']}_s{r['seed']}"] = r
            print(f"  [{i + 1}/{len(todo)}] {r['task']:>8} {r['cond']:>11} s{r['seed']:<2} "
                  f"{'OK  ' if r['finished'] else 'DIV '} step {r['steps_run']:5d}", flush=True)
            save(R)
    else:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_work, todo)):
                R["runs"][f"{r['task']}_{r['cond']}_s{r['seed']}"] = r
                print(f"  [{i + 1}/{len(todo)}] {r['task']:>8} {r['cond']:>11} s{r['seed']:<2} "
                      f"{'OK  ' if r['finished'] else 'DIV '} step {r['steps_run']:5d}", flush=True)
                save(R)
    return R


def analyse(runs):
    """Per (task, condition): run rate, calls at risk, hazard per call, both with intervals."""
    census, blocks = {}, {}
    for tk in TASKS:
        for cd in CONDS:
            keys = [k for k in runs if runs[k]["task"] == tk and runs[k]["cond"] == cd]
            if not keys:
                continue
            rows = [runs[k] for k in sorted(keys, key=lambda x: runs[x]["seed"])]
            n = len(rows)
            k = sum(not r["finished"] for r in rows)
            calls = sum(r["calls"] for r in rows)
            lo, hi = wilson(k, n)
            hlo, hhi = wilson(k, calls)
            dsteps = [r["div_step"] for r in rows if not r["finished"]]
            census[f"{tk}_{cd}"] = {
                "runs": n, "seeds": [r["seed"] for r in rows],
                "diverged": k, "run_rate": k / n, "run_ci95": [lo, hi],
                "calls": calls, "hazard_per_call": k / calls, "hazard_ci95": [hlo, hhi],
                "calls_per_divergence": (calls / k if k else None),
                "steps_to_div": dsteps,
                "steps_to_div_median": (float(np.median(dsteps)) if dsteps else None),
                "steps_mean_run": float(np.mean([r["steps_run"] for r in rows])),
                "div_layers": [r["div_layer"] for r in rows if not r["finished"]],
            }
            # hazard in blocks of 200 training steps (exposure = calls in the block)
            if cd != "e5m2_clamp":
                def cum(r, s):
                    """calls the run has made by the end of training step s."""
                    s = min(s, r["steps_run"])
                    if s <= 0:
                        return 0
                    if r["finished"]:
                        return r["layers"] * s
                    if s <= r["div_step"]:
                        return r["layers"] * s
                    return r["layers"] * r["div_step"] + (r["div_layer"] or 0) + 1
                bl = []
                for b0 in range(0, STEPS, 200):
                    b1 = b0 + 200
                    ev = sum(1 for r in rows if not r["finished"] and r["div_step"] is not None
                             and b0 <= r["div_step"] < b1)
                    ex = sum(cum(r, b1) - cum(r, b0) for r in rows)
                    bl.append({"steps": [b0, b1], "events": ev, "calls": ex,
                               "hazard_per_call": (ev / ex if ex else None),
                               "ci95": list(wilson(ev, ex))})
                blocks[f"{tk}_{cd}"] = bl
    tests = {}
    for cd in CONDS:
        a, b = census.get(f"digits_{cd}"), census.get(f"diabetes_{cd}")
        if not a or not b:
            continue
        tests[cd] = {
            "run_level_fisher_p": fisher_p(a["diverged"], a["runs"] - a["diverged"],
                                           b["diverged"], b["runs"] - b["diverged"]),
            "call_level_exact_p": binom_exact_p(a["diverged"], a["calls"],
                                               b["diverged"], b["calls"]),
            "hazard_ratio_diabetes_over_digits": (b["hazard_per_call"] / a["hazard_per_call"]
                                                 if a["hazard_per_call"] else None),
        }
    for tk in TASKS:
        a, b = census.get(f"{tk}_e5m2"), census.get(f"{tk}_e5m2_s1035")
        if not a or not b:
            continue
        tests[f"{tk}_safety"] = {
            "run_level_fisher_p": fisher_p(a["diverged"], a["runs"] - a["diverged"],
                                           b["diverged"], b["runs"] - b["diverged"]),
            "call_level_exact_p": binom_exact_p(a["diverged"], a["calls"],
                                               b["diverged"], b["calls"]),
        }
    return census, blocks, tests


def check(R):
    """Do the 1,200-step runs reproduce pilot 23's 400-step census on their prefix?"""
    import os.path as op
    cand = [op.join(HERE, "pilot23_results.json"),
            op.join(HERE, "..", "results", "pilot23_results.json")]
    src = next((c for c in cand if op.exists(c)), None)
    if src is None:
        print("  pilot23_results.json not found; no prefix check")
        return {}
    with open(src) as f:
        p23 = json.load(f)["a_census"]
    out = {}
    for rk, r in R["runs"].items():
        k23 = (f"{r['task']}_s{r['seed']}_e5m2_clamp" if r["cond"] == "e5m2_clamp"
               else f"{r['task']}_s{r['seed']}_{r['cond']}")
        if k23 not in p23:
            continue
        p = p23[k23]
        here_div = (not r["finished"]) and r["div_step"] is not None and r["div_step"] < 400
        out[f"{rk}_prefix400"] = {
            "pilot23_finished": p["finished"], "pilot25_diverged_under_400": here_div,
            "agree": bool(p["finished"]) == (not here_div),
            "pilot23_steps": p["steps"], "pilot25_div_step": r["div_step"]}
    agree = [v["agree"] for v in out.values()]
    print(f"  prefix check vs pilot23: {sum(agree)}/{len(agree)} agree")
    for k, v in out.items():
        if not v["agree"]:
            print(f"    MISMATCH {k}: pilot23 finished={v['pilot23_finished']} "
                  f"steps={v['pilot23_steps']}, pilot25 div_step={v['pilot25_div_step']}")
    return out


def report(R):
    print("\n" + "=" * 110)
    print("the long census: 1,200 steps, 25 seeds, hazard per Muon call with 95% Wilson CI")
    print("=" * 110)
    for k in sorted(R["census"]):
        c = R["census"][k]
        print(f"  {k:>22}: {c['diverged']:2d}/{c['runs']:2d} runs "
              f"(rate {100 * c['run_rate']:5.1f}%, CI {100 * c['run_ci95'][0]:.1f}-"
              f"{100 * c['run_ci95'][1]:.1f}%)  calls {c['calls']:6d}  "
              f"hazard {100 * c['hazard_per_call']:.4f}% "
              f"[{100 * c['hazard_ci95'][0]:.4f},{100 * c['hazard_ci95'][1]:.4f}]  "
              f"1 in {c['calls_per_divergence']:.0f} calls" if c["diverged"] else
              f"  {k:>22}: {c['diverged']:2d}/{c['runs']:2d} runs "
              f"(rate {100 * c['run_rate']:5.1f}%, CI {100 * c['run_ci95'][0]:.1f}-"
              f"{100 * c['run_ci95'][1]:.1f}%)  calls {c['calls']:6d}  hazard 0 "
              f"(95% upper {100 * c['hazard_ci95'][1]:.4f}%)")
    print("\n  the hazard in blocks of 200 steps:")
    for k in sorted(R["blocks"]):
        print(f"  {k:>22}: " + "  ".join(f"{b['events']:2d}/{b['calls']:5d}"
                                         f"({100 * (b['hazard_per_call'] or 0):.3f}%)"
                                         for b in R["blocks"][k]))
    print("\n  tests:")
    for k, v in R["tests"].items():
        fr = v.get("run_level_fisher_p")
        cl = v.get("call_level_exact_p")
        hr = v.get("hazard_ratio_diabetes_over_digits")
        print(f"  {k:>22}: Fisher(run) p={fr if fr is None else round(fr, 4)}  "
              f"exact(call) p={cl if cl is None else round(cl, 6)}"
              + (f"  hazard ratio {hr:.2f}" if hr else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="digits,diabetes")
    ap.add_argument("--cond", default="e5m2,e5m2_s1035,e5m2_clamp")
    ap.add_argument("--seeds", default="0-24")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.analyse or a.check:
        R = load()
        R["prefix_check"] = check(R)
        save(R)
        report(R)
        print("\nwrote", RES)
        return
    lo, hi = (int(x) for x in a.seeds.split("-"))
    R = collect([t for t in a.task.split(",") if t], [c for c in a.cond.split(",") if c],
                tuple(range(lo, hi + 1)), a.workers)
    R["design"] = {"steps": STEPS, "seeds": [0, 24], "K": 6, "lr": 0.01,
                   "divergence": "some weight above 1e3 in magnitude, or a weight, loss or "
                                 "update that stops being finite",
                   "call": "one application of the composite to one hidden weight matrix",
                   "censoring": "a run that has not diverged at 1,200 steps contributes all "
                                "of its calls and no event",
                   "python": sys.version.split()[0], "numpy": np.__version__,
                   "conditions": {"e5m2": "s = 1.01, shipped", "e5m2_s1035": "s = 1.035",
                                  "e5m2_clamp": "s = 1.01, state clamped into its window"}}
    save(R)
    report(R)


if __name__ == "__main__":
    main()
