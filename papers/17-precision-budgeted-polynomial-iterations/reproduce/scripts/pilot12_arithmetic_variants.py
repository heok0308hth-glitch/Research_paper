#!/usr/bin/env python3
"""
Pilot 12 - does the two-term law survive a kernel that is not the paper's?

The paper's measurements all use one kernel model: state and intermediate are
rounded to p significant bits and products accumulate in float64.  A reviewer is
entitled to ask whether the constants are constants of that model rather than of
the arithmetic.  This script re-runs the shipped design (cushion 0.02, safety
1.01) under four kernel variants and compares each measurement with the same
certificate  err ~ E_K + 0.142 * 2^-p * S_pert(K):

  kernel   the paper's model (reproduces Table 1 bit for bit)
  eager    every matrix product is rounded to p bits as well (A, A^2, B X, a X)
  fp32acc  storage at p bits, but every dot product accumulated in float32 with
           one rounding per addition (the tensor-core picture: fp32 accumulator)
  sr       the paper's model with stochastic rounding instead of round-to-nearest
           (3 seeds, mean and spread reported)

Also produced here:
  * self-calibration of c on 8 cheap cells, then applied to a disjoint held-out
    grid (a caller who does not trust 0.142 can fit it himself);
  * the cost of the budget plan (certificate on a 400-point grid) against the
    cost of one 128x32, K=8 iteration, for the planner paragraph.

Writes pilot12_results.json.
"""
import json
import os
import time

import numpy as np

from pilot7_pert_aware_design import make_M, mat_run, pe_coeffs, q, stats

HERE = os.path.dirname(os.path.abspath(__file__))
C = 0.142
GRID = np.logspace(-4, 0, 1500)          # replaced per cell by its own [L,1] grid


def q_sr(x, p, rng):
    """Stochastic rounding to p significant bits (unbiased in expectation)."""
    x = np.asarray(x, float)
    m, e = np.frexp(x)
    s = float(2 ** (p - 1))
    t = m * s
    lo = np.floor(t)
    frac = t - lo
    step = (rng.random(x.shape) < frac).astype(float)
    return np.ldexp((lo + step) / s, e)


def mm32(A, B):
    """Matrix product with float32 accumulation and one rounding per addition."""
    A32 = A.astype(np.float32)
    B32 = B.astype(np.float32)
    acc = np.zeros((A32.shape[0], B32.shape[1]), dtype=np.float32)
    for k in range(A32.shape[1]):
        acc = acc + np.outer(A32[:, k], B32[k, :]).astype(np.float32)
    return acc.astype(float)


def step_variant(X, coef, p, variant, rng):
    a, b, c = coef
    if variant == "kernel":
        A = X @ X.T
        B = q(b * A + c * (A @ A), p)
        return q(a * X + B @ X, p)
    if variant == "eager":
        A = q(X @ X.T, p)
        A2 = q(A @ A, p)
        B = q(b * A + c * A2, p)
        BX = q(B @ X, p)
        return q(q(a * X, p) + BX, p)
    if variant == "fp32acc":
        A = mm32(X, X.T)
        A2 = mm32(A, A)
        B = q(b * A + c * A2, p)
        BX = mm32(B, X)
        return q(a * X + BX, p)
    if variant == "sr":
        A = X @ X.T
        B = q_sr(b * A + c * (A @ A), p, rng)
        return q_sr(a * X + B @ X, p, rng)
    raise ValueError(variant)


def run_variant(co, M, p, variant, rng=None):
    X = M / np.linalg.norm(M, 2)
    if variant == "sr":
        X = q_sr(X, p, rng)
    else:
        X = q(X, p)
    for coef in co:
        X = step_variant(X, coef, p, variant, rng)
    return X


def split_errors(X, ref):
    U, _, Vt = np.linalg.svd(X, full_matrices=False)
    P = U @ Vt
    return (float(np.linalg.norm(X - ref, 2)),
            float(np.linalg.norm(P - ref, 2)),
            float(np.linalg.norm(X - P, 2)))


def cert(L, K, p):
    co, _ = pe_coeffs(L, K)
    grid = np.logspace(np.log10(L), 0, 1500)
    EK, S, _ = stats(co, grid, K)
    return co, EK, S, EK + C * 2.0 ** -p * S


def main():
    out = {"config": {"c": C, "design": "cushion 0.02, safety 1.01",
                      "K": 6}, "variants": {}, "reproduction_check": {}}

    # ---- reproduction of Table 1 (kernel variant must match the paper)
    for L, p in ((1e-3, 8), (1e-3, 11), (1e-3, 16)):
        M, ref = make_M(64, L)
        co, EK, S, pred = cert(L, 6, p)
        err, _, _ = split_errors(run_variant(co, M, p, "kernel"), ref)
        out["reproduction_check"][f"L{L}_p{p}"] = {
            "err": err, "pred": pred, "ratio": pred / err, "E_K": EK, "S_pert": S}
        print(f"reproduction L={L:g} p={p:2d}: err={err:.4e} pred={pred:.4e} "
              f"ratio={pred / err:.2f}  (E_K={EK:.4e}, S_pert={S:.1f})")

    # ---- variants over ell x d x p
    cells = [(1e-2, 64), (1e-3, 64), (1e-3, 128), (1e-4, 64)]
    for variant in ("kernel", "eager", "fp32acc", "sr"):
        rows = []
        for L, d in cells:
            M, ref = make_M(d, L)
            for p in (8, 11, 16):
                co, EK, S, pred = cert(L, 6, p)
                if variant == "sr":
                    es, subs = [], []
                    for seed in range(3):
                        rng = np.random.default_rng(7000 + seed)
                        e, s_, _ = split_errors(run_variant(co, M, p, "sr", rng), ref)
                        es.append(e)
                        subs.append(s_)
                    err = float(np.mean(es))
                    spread = float(max(es) - min(es))
                    sub = float(np.mean(subs))
                else:
                    try:
                        err, sub, _ = split_errors(run_variant(co, M, p, variant), ref)
                        spread = 0.0
                    except Exception:
                        err, sub, spread = float("inf"), float("inf"), 0.0
                rows.append({"ell": L, "d": d, "p": p, "err": err, "sub": sub,
                             "spread": spread, "pred": pred,
                             "ratio": pred / err if np.isfinite(err) and err > 0 else None})
                print(f"{variant:8s} L={L:g} d={d:3d} p={p:2d}: err={err:.3e} "
                      f"pred={pred:.3e} ratio={pred / err if err else float('nan'):.2f}")
        out["variants"][variant] = rows

    # ---- summary per variant
    summ = {}
    for v, rows in out["variants"].items():
        r = [x["ratio"] for x in rows if x["ratio"] is not None and np.isfinite(x["ratio"])]
        summ[v] = {"n": len(rows), "within_2x": int(sum(1 for x in r if 0.5 <= x <= 2.0)),
                   "ratio_min": float(min(r)), "ratio_max": float(max(r)),
                   "ratio_median": float(np.median(r))}
    out["summary"] = summ
    print("\nsummary:", json.dumps(summ, indent=1))

    # ---- self-calibration of c on 8 cells, applied to a held-out grid
    fit = []
    for L in (1e-2, 1e-3):
        M, ref = make_M(64, L)
        for p in (8, 11):
            for K in (4, 6):
                co, EK, S, _ = cert(L, K, p)
                err, _, _ = split_errors(run_variant(co, M, p, "kernel"), ref)
                fit.append(max(err - EK, 0.0) / (2.0 ** -p * S))
    c_hat = float(np.median(fit))
    out["calibration"] = {"c_hat": c_hat, "n": len(fit),
                          "fit": [float(x) for x in fit],
                          "c_true": C}
    print(f"\ncalibrated c_hat = {c_hat:.4f} on {len(fit)} cells (true {C})")

    held = []
    for L in (3e-3, 1e-3, 3e-4):
        M, ref = make_M(128, L)
        for p in (9, 13, 16):
            co, EK, S, _ = cert(L, 6, p)
            err, _, _ = split_errors(run_variant(co, M, p, "kernel"), ref)
            pr = EK + c_hat * 2.0 ** -p * S
            held.append({"ell": L, "p": p, "err": err, "pred": pr, "ratio": pr / err})
            print(f"held-out L={L:g} p={p:2d}: err={err:.3e} pred(c_hat)={pr:.3e} ratio={pr / err:.2f}")
    out["heldout_calibrated"] = held
    rr = [h["ratio"] for h in held]
    out["heldout_calibrated_summary"] = {"n": len(rr), "min": float(min(rr)),
                                         "max": float(max(rr)),
                                         "within_1_5x": int(sum(1 for x in rr if 1 / 1.5 <= x <= 1.5))}

    # ---- timing: plan vs one iteration
    M = np.random.RandomState(3).randn(128, 32)
    grid = np.logspace(np.log10(0.15), 0, 400)
    t0 = time.perf_counter()
    for _ in range(200):
        co, _ = pe_coeffs(0.15, 8)
        stats(co, grid, 8)
    t_plan = (time.perf_counter() - t0) / 200
    t0 = time.perf_counter()
    for _ in range(20):
        mat_run(co, M, 8)
    t_iter = (time.perf_counter() - t0) / 20
    out["timing"] = {"plan_s": t_plan, "iteration_128x32_K8_s": t_iter,
                     "plan_over_iteration": t_plan / t_iter}
    print(f"\nplan {t_plan * 1e6:.1f} us vs iteration {t_iter * 1e3:.2f} ms "
          f"(ratio {t_plan / t_iter:.2e})")

    path = os.path.join(HERE, "pilot12_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
