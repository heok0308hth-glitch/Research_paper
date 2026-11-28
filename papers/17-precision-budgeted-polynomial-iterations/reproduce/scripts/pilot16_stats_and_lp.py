#!/usr/bin/env python3
"""
Pilot 16 - uncertainty on the headline ratios, and independent checks of the two
places where a discretisation could be doing the work.

 [1] Bootstrap 95% confidence intervals (percentile, 10,000 resamples, fixed seed) on
     every ratio the abstract and the introduction quote:
       certificate ratio over the 192 design-family conditions (re-read from pilot6),
       the stopping-rule ratio over the 90-cell sweep (pilot7c),
       the two-term polar formula over the 500 real momentum matrices (pilot11),
       the self-calibrated prediction on the 9 held-out cells (pilot12),
       the graded-schedule gain over the 48 equal-total-bit cells (pilot10b).
     Both the median and the "within a factor" fractions get intervals, because the
     text quotes both.

 [2] Grid convergence of the linear programs.  The single-step frontier E_d(ell) and
     the last-step derivative family p'(1) = lambda are solved on an n-point grid; the
     paper quotes lambda* = -0.0353 and the eps_d table.  Re-solve on
     n in {50, 100, 200, 400, 800, 1600, 3200} and report how the answers move.

 [3] The integer allocation.  The schedule uses largest-remainder rounding; compare it
     with the exact integer optimum from a dynamic program over the K+1 slots at the
     same total bits, on the 48 cells the paper quotes, so the 3.06x is the schedule
     and not the rounding.

 [4] Held-out shapes and spectra.  The certificate was fitted on logspace spectra at
     d in {64,128}; test it on d in {48,96,256} and on three spectra that are not
     logspace (two-plateau, spectral gap, power law), measuring the certificate ratio
     and the first-order polar-response ratio.

Kernel model as in pilots 1-15.
"""
import json
import os
import numpy as np
from scipy.optimize import linprog

from pilot7_pert_aware_design import q, pe_coeffs, stats

HERE = os.path.dirname(os.path.abspath(__file__))
R = {}
C_CERT = 0.142
CUSH, SAFE = 0.02, 1.01
NB = 10000
RNG = np.random.default_rng(20261127)


# ---------------------------------------------------------------- bootstrap
def boot_ci(x, stat=np.median, nb=NB):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return None
    idx = RNG.integers(0, x.size, size=(nb, x.size))
    s = np.array([stat(x[i]) for i in idx])
    return {"n": int(x.size), "point": float(stat(x)),
            "ci": [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]}


def boot_frac(x, pred, nb=NB):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    idx = RNG.integers(0, x.size, size=(nb, x.size))
    s = np.array([float(pred(x[i])) for i in idx])
    return {"n": int(x.size), "point": float(pred(x)),
            "ci": [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]}


def maxdev(r):
    r = np.asarray(r, float)
    r = r[np.isfinite(r) & (r > 0)]
    return float(np.max(np.maximum(r, 1.0 / r)))


def section1():
    print("=" * 100)
    print("[1] bootstrap 95% intervals on the quoted ratios")
    print("=" * 100)
    out = {}

    # --- 192 design-family conditions
    d6 = json.load(open(os.path.join(HERE, "pilot6_results.json")))
    rows = [r for r in d6["data"]]
    rr = np.array([r["err"] / (r["EK"] + C_CERT * r["u"] * r["S_prefix_run"]) for r in rows], float)
    ok = np.isfinite(rr) & (rr > 0)
    out["cert192"] = {"median": boot_ci(rr[ok]),
                      "max_deviation": boot_ci(rr[ok], stat=maxdev),
                      "frac_within_1_6": boot_frac(rr[ok], lambda v: float(np.mean(np.maximum(v, 1 / v) <= 1.6))),
                      "raw_median": float(np.median(rr[ok])),
                      "raw_range": [float(np.min(rr[ok])), float(np.max(rr[ok]))]}
    c = out["cert192"]
    print(f"  192 conditions: err/pred median {c['median']['point']:.3f} "
          f"CI [{c['median']['ci'][0]:.3f}, {c['median']['ci'][1]:.3f}]; "
          f"max deviation {c['max_deviation']['point']:.2f} "
          f"CI [{c['max_deviation']['ci'][0]:.2f}, {c['max_deviation']['ci'][1]:.2f}]; "
          f"within 1.6x {c['frac_within_1_6']['point'] * 100:.0f}% "
          f"CI [{c['frac_within_1_6']['ci'][0] * 100:.0f}, {c['frac_within_1_6']['ci'][1] * 100:.0f}]%")

    # --- 90-cell stopping sweep
    d7 = json.load(open(os.path.join(HERE, "pilot7c_results.json")))
    for dk in ("ratio_d64", "ratio_d128"):
        v = np.array([s[dk] for s in d7["stop"].values() if s.get(dk) is not None and np.isfinite(s[dk])], float)
        out[f"stop_{dk[-3:]}"] = {"median": boot_ci(v), "min": float(v.min()), "max": float(v.max()),
                                  "frac_within_1_5": boot_frac(v, lambda x: float(np.mean((x > 1 / 1.5) & (x < 1.5))))}
        e = out[f"stop_{dk[-3:]}"]
        print(f"  90-cell stopping, {dk[-3:]}: median {e['median']['point']:.3f} "
              f"CI [{e['median']['ci'][0]:.3f}, {e['median']['ci'][1]:.3f}], range "
              f"[{e['min']:.2f}, {e['max']:.2f}], within 1.5x {e['frac_within_1_5']['point'] * 100:.0f}% "
              f"CI [{e['frac_within_1_5']['ci'][0] * 100:.0f}, {e['frac_within_1_5']['ci'][1] * 100:.0f}]%")

    # --- 500 real momentum matrices
    d11 = json.load(open(os.path.join(HERE, "pilot11_results.json")))
    for p in ("8", "11", "16"):
        v = np.array([r["ratio"] for r in d11["real_spectra"][p]["rows"]], float)
        v = v[np.isfinite(v)]
        out[f"real500_p{p}"] = {"median": boot_ci(v),
                                "frac_10pct": boot_frac(v, lambda x: float(np.mean(np.abs(x - 1) <= 0.10))),
                                "min": float(v.min()), "max": float(v.max())}
        e = out[f"real500_p{p}"]
        print(f"  500 momentum matrices, p={p}: two-term pred/meas median {e['median']['point']:.4f} "
              f"CI [{e['median']['ci'][0]:.4f}, {e['median']['ci'][1]:.4f}], within 10% "
              f"{e['frac_10pct']['point'] * 100:.0f}% CI [{e['frac_10pct']['ci'][0] * 100:.0f}, "
              f"{e['frac_10pct']['ci'][1] * 100:.0f}]%")

    # --- 9 held-out calibration cells
    d12 = json.load(open(os.path.join(HERE, "pilot12_results.json")))
    v = np.array([r["ratio"] for r in d12["heldout_calibrated"]], float)
    out["heldout9"] = {"median": boot_ci(v), "min": float(v.min()), "max": float(v.max()),
                       "frac_within_1_5": boot_frac(v, lambda x: float(np.mean((x > 1 / 1.5) & (x < 1.5))))}
    e = out["heldout9"]
    print(f"  9 held-out cells: pred/meas median {e['median']['point']:.3f} "
          f"CI [{e['median']['ci'][0]:.3f}, {e['median']['ci'][1]:.3f}], range [{e['min']:.2f}, {e['max']:.2f}], "
          f"within 1.5x {e['frac_within_1_5']['point'] * 100:.0f}%")

    # --- 48 allocation cells
    d10 = json.load(open(os.path.join(HERE, "pilot10b_results.json")))
    gains, cells = [], []
    for ck, cell in d10["alloc"].items():
        for bk, b in cell.items():
            if isinstance(b, dict) and "gain" in b and b["gain"] is not None:
                gains.append(b["gain"])
                cells.append((ck, bk, b))
    g = np.array(gains, float)
    out["alloc48"] = {"median": boot_ci(g), "p10": float(np.percentile(g, 10)),
                      "p90": float(np.percentile(g, 90)),
                      "frac_gt_2": boot_frac(g, lambda x: float(np.mean(x > 2.0)))}
    e = out["alloc48"]
    print(f"  48 allocation cells: median gain {e['median']['point']:.2f} "
          f"CI [{e['median']['ci'][0]:.2f}, {e['median']['ci'][1]:.2f}], p10-p90 "
          f"{e['p10']:.2f}-{e['p90']:.2f}, share >2x {e['frac_gt_2']['point'] * 100:.0f}%")
    R["bootstrap"] = out
    return cells


# ---------------------------------------------------------------- grid convergence
def lp_minimax(deg, L, n):
    """min over odd polynomials of degree deg of max_{x in [L,1]} |P(x)-1|, on an n-point grid."""
    x = np.arange(n) * (1.0 - L) / (n - 1) + L
    powers = list(range(1, deg + 1, 2))
    A = np.stack([x ** p for p in powers], axis=1)
    Aub = np.vstack([np.hstack([A, -np.ones((n, 1))]), np.hstack([-A, -np.ones((n, 1))])])
    bub = np.concatenate([np.ones(n), -np.ones(n)])
    c = np.zeros(len(powers) + 1)
    c[-1] = 1.0
    res = linprog(c, A_ub=Aub, b_ub=bub, bounds=[(None, None)] * len(powers) + [(0.0, None)],
                  method="highs")
    if not res.success:
        return None
    # verify on a much finer grid (the continuous maximum, approximated)
    xf = np.linspace(L, 1.0, 200001)
    P = sum(res.x[i] * xf ** p for i, p in enumerate(powers))
    return {"eps_grid": float(res.x[-1]), "eps_fine": float(np.max(np.abs(P - 1))),
            "coef": [float(v) for v in res.x[:-1]]}


def lp_lambda(L, lam, n):
    """odd quintic with p'(1) = lam minimising the max error on [L,1], n-point grid."""
    x = np.linspace(L, 1.0, n)
    A = np.stack([x, x ** 3, x ** 5], axis=1)
    Aub = np.vstack([np.hstack([A, -np.ones((n, 1))]), np.hstack([-A, -np.ones((n, 1))])])
    bub = np.concatenate([np.ones(n), -np.ones(n)])
    Aeq = np.array([[1.0, 3.0, 5.0, 0.0]])
    beq = np.array([lam])
    c = np.array([0.0, 0.0, 0.0, 1.0])
    res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq,
                  bounds=[(None, None)] * 3 + [(0.0, None)], method="highs")
    if not res.success:
        return None
    a, b, cc, eps = res.x
    return {"a": float(a), "b": float(b), "c": float(cc), "eps": float(eps)}


def section2():
    print("\n" + "=" * 100)
    print("[2] grid convergence of the linear programs")
    print("=" * 100)
    out = {}
    for deg in (5, 9, 17):
        rows = {}
        print(f"  minimax odd degree {deg} at ell=1e-3 (eps on the LP grid vs on a 200,001-point grid):")
        for n in (50, 100, 200, 400, 800, 1600, 3200):
            r = lp_minimax(deg, 1e-3, n)
            if r:
                rows[n] = r
                print(f"    n={n:5d}: eps_grid {r['eps_grid']:.10f}  eps_fine {r['eps_fine']:.10f}  "
                      f"rel {abs(r['eps_fine'] / r['eps_grid'] - 1):.2e}")
        fine = [r["eps_fine"] for r in rows.values()]
        out[f"deg{deg}"] = {"rows": rows,
                            "eps_fine_range": [min(fine), max(fine)],
                            "coef_drift_50_vs_3200": [abs(a - b) / max(abs(b), 1e-30) for a, b in
                                                     zip(rows[50]["coef"], rows[3200]["coef"])]}
        print(f"    eps_fine over n: {min(fine):.10f} - {max(fine):.10f} "
              f"(spread {max(fine) / min(fine) - 1:.1e}); coefficient drift n=50 vs 3200: "
              + ", ".join(f"{x:.1e}" for x in out[f"deg{deg}"]["coef_drift_50_vs_3200"]))

    # lambda family: coefficients and eps at the paper's lambda*, then the argmin of J over lambda
    print("\n  last-step family p'(1) = lam at ell=1e-3:")
    lam0 = -0.0353
    rows = {}
    for n in (100, 200, 400, 800, 1600):
        r = lp_lambda(1e-3, lam0, n)
        rows[n] = r
        print(f"    n={n:5d}: (a,b,c) = ({r['a']:.6f}, {r['b']:.6e}, {r['c']:.6e})  eps {r['eps']:.8f}")
    a_drift = max(abs(rows[n]["a"] - rows[1600]["a"]) / abs(rows[1600]["a"]) for n in rows)
    c_drift = max(abs(rows[n]["c"] - rows[1600]["c"]) / abs(rows[1600]["c"]) for n in rows)
    out["lambda"] = {"rows": rows, "max_rel_drift_a": a_drift, "max_rel_drift_c": c_drift,
                     "lam": lam0}
    print(f"    max relative drift vs n=1600: a {a_drift:.1e}, c {c_drift:.1e}")

    # argmin of J(lambda) on each grid: does lambda* move?
    print("\n  argmin of J over lam (K=8, p=8, ell=1e-3, grid 400):")
    grid = np.logspace(-3, 0, 400)
    lam_grid = np.round(np.arange(-0.5, 0.001, 0.01), 4)
    best = {}
    for n in (100, 400, 1600):
        Js = []
        for lam in lam_grid:
            r = lp_lambda(1e-3, lam, n)
            if r is None:
                Js.append(np.inf)
                continue
            base = pe_coeffs(1e-3, 8, CUSH, SAFE)[0]
            co = base[:7] + [(r["a"], r["b"], r["c"])]
            EK, S, _ = stats(co, grid, 8)
            Js.append(EK + C_CERT * 2.0 ** -8 * S)
        Js = np.array(Js)
        i = int(np.argmin(Js))
        best[n] = {"lam_star": float(lam_grid[i]), "J": float(Js[i]),
                   "J_at_paper_lam": float(Js[int(np.argmin(np.abs(lam_grid - lam0)))])}
        print(f"    n={n:5d}: lam* = {best[n]['lam_star']:.2f}  J {best[n]['J']:.6e}  "
              f"(J at lam=-0.0353: {best[n]['J_at_paper_lam']:.6e})")
    out["lambda_argmin"] = best
    R["lp_convergence"] = out


# ---------------------------------------------------------------- integer allocation
def dp_integer(kappa, B, lo=4, hi=30):
    """exact integer minimisation of max_t kappa_t 2^-p_t with sum p_t = B."""
    n = len(kappa)
    # cost(p) = log2(kappa_t) - p ; minimise the max over t
    best = {}

    def rec(t, left):
        key = (t, left)
        if key in best:
            return best[key]
        if t == n - 1:
            if lo <= left <= hi:
                best[key] = (float(np.log2(kappa[t]) - left), [left])
                return best[key]
            best[key] = (np.inf, None)
            return best[key]
        bc, bs = np.inf, None
        for p in range(lo, min(hi, left - lo * (n - 1 - t)) + 1):
            c, s = rec(t + 1, left - p)
            if s is None:
                continue
            c = max(c, float(np.log2(kappa[t]) - p))
            if c < bc:
                bc, bs = c, [p] + s
        best[key] = (bc, bs)
        return best[key]

    c, s = rec(0, B)
    return s, c


def waterfill_int(kappa, B, lo=4, hi=30):
    """closed-form schedule rounded to integers by largest remainder (what the paper uses)."""
    n = len(kappa)
    v = np.log2(np.asarray(kappa, float)) + (B - np.sum(np.log2(kappa))) / n
    q_ = np.floor(np.clip(v, lo, hi)).astype(int)
    frac = v - q_
    free = np.ones(n, bool)
    while q_.sum() < B:
        cand = [i for i in range(n) if free[i] and q_[i] < hi]
        if not cand:
            return None
        i = max(cand, key=lambda j: frac[j])
        q_[i] += 1
        free[i] = False
    while q_.sum() > B:
        cand = [i for i in range(n) if free[i] and q_[i] > lo]
        if not cand:
            return None
        i = min(cand, key=lambda j: frac[j])
        q_[i] -= 1
        free[i] = False
    return [int(x) for x in q_]


def section3():
    print("\n" + "=" * 100)
    print("[3] integer allocation: largest-remainder schedule vs the exact DP optimum")
    print("=" * 100)
    d10 = json.load(open(os.path.join(HERE, "pilot10b_results.json")))
    n_ok = 0
    ratio = []
    for ck, cell in d10["alloc"].items():
        kap = np.array(cell["profile_p10"], float)
        for bk, b in cell.items():
            if not isinstance(b, dict) or "schedule" not in b:
                continue
            B = b["bits"]
            sched = b["schedule"]
            ws = waterfill_int(kap, B)
            dp, _ = dp_integer(kap, B)
            if dp is None or ws is None:
                continue
            f = lambda s: max(k * 2.0 ** -p for k, p in zip(kap, s))
            ratio.append(f(sched) / f(dp))
            n_ok += 1
    ratio = np.array(ratio, float)
    R["integer_dp"] = {"n_cells": int(n_ok), "ratio_schedule_over_dp": {
        "median": float(np.median(ratio)), "max": float(np.max(ratio)),
        "min": float(np.min(ratio)),
        "n_exact": int(np.sum(np.abs(ratio - 1) < 1e-12))}}
    e = R["integer_dp"]["ratio_schedule_over_dp"]
    print(f"  {n_ok} cells: schedule/DP optimality ratio median {e['median']:.6f}, "
          f"max {e['max']:.6f}, min {e['min']:.6f}, exactly optimal {e['n_exact']}/{n_ok}")


# ---------------------------------------------------------------- held-out shapes
def spectra(kind, d, L, seed=0):
    rs = np.random.RandomState(seed)
    U, _ = np.linalg.qr(rs.randn(d, d))
    V, _ = np.linalg.qr(rs.randn(d, d))
    if kind == "logspace":
        sv = np.logspace(0, np.log10(L), d)
    elif kind == "plateau":
        sv = np.ones(d)
        sv[d // 2:] = L
    elif kind == "gap":
        h = d // 2
        a = np.logspace(0, 0.5 * np.log10(L), h)
        b = np.logspace(0.5 * np.log10(L) - 2, np.log10(L), d - h)
        sv = np.concatenate([a, np.clip(b, L, 1.0)])
        sv = sv / sv[0]
    elif kind == "power":
        sv = 1.0 / (1.0 + np.arange(d))
        sv = sv / sv[0]
        sv = np.maximum(sv, L)
    else:
        raise ValueError(kind)
    M = (U * sv) @ V.T
    return M, U @ V.T, sv


def n2(X):
    return float(np.linalg.norm(X, 2))


def kernel_run(co, X0, p):
    X = q(X0, p)
    for a, b, c in co:
        A = X @ X.T
        B = q(b * A + c * (A @ A), p)
        X = q(a * X + B @ X, p)
    return X


def first_order(X0, Mhat, p):
    """the two-term polar-response prediction of the input cast (Theorem 3)."""
    U, s, Vt = np.linalg.svd(Mhat, full_matrices=False)
    m, n = Mhat.shape
    E = q(X0, p) - Mhat
    F = U.T @ E @ Vt.T
    K = (F - F.T) / (s[:, None] + s[None, :])
    t1 = U @ K @ Vt
    V = Vt.T
    invsqrt = (V * (1.0 / s)) @ V.T
    t2 = (np.eye(m) - U @ U.T) @ E @ invsqrt
    return n2(t1), n2(t2), n2(t1 + t2)


def section4():
    print("\n" + "=" * 100)
    print("[4] held-out shapes and spectra (not used to fit c)")
    print("=" * 100)
    R["heldout_shapes"] = {}
    print(f"  {'shape':>9} {'spectrum':>9} {'ell':>7} {'p':>3} {'measured':>10} {'cert':>9} "
          f"{'ratio':>6} {'1st-order':>10} {'ratio1':>7}")
    for d in (48, 96, 256):
        for kind in ("logspace", "plateau", "gap", "power"):
            for L in (0.1, 1e-2):
                for p in (8, 12):
                    M, ref, sv = spectra(kind, d, L, seed=7)
                    co, _ = pe_coeffs(L, 6, CUSH, SAFE)
                    X0 = M / n2(M)
                    X = kernel_run(co, X0, p)
                    W, _, Vt2 = np.linalg.svd(X, full_matrices=False)
                    e = n2(W @ Vt2 - ref)
                    grid = np.logspace(np.log10(max(sv[-1] / sv[0], 1e-12)), 0, 400)
                    EK, S, _ = stats(co, grid, 6)
                    pred = EK + C_CERT * 2.0 ** -p * S
                    p1, p2, pt = first_order(X0, M, p)
                    rec = {"d": d, "spectrum": kind, "ell": L, "p": p, "err": e,
                           "pred": pred, "ratio": pred / e if e else None,
                           "first_in": p1, "first_proj": p2, "first_total": pt,
                           "ratio_first": pt / e if e else None}
                    R["heldout_shapes"][f"{kind}_d{d}_L{L:g}_p{p}"] = rec
                    print(f"  {d:9d} {kind:>9} {L:7g} {p:3d} {e:10.3e} {pred:9.3e} "
                          f"{(pred / e) if e else float('nan'):6.2f} {pt:10.3e} "
                          f"{(pt / e) if e else float('nan'):7.2f}")
    rows = [r for r in R["heldout_shapes"].values() if r["ratio"]]
    rr = np.array([r["ratio"] for r in rows], float)
    r1 = np.array([r["ratio_first"] for r in rows], float)
    R["heldout_summary"] = {
        "n": len(rows),
        "cert_ratio": boot_ci(rr), "cert_min": float(rr.min()), "cert_max": float(rr.max()),
        "first_ratio": boot_ci(r1), "first_min": float(np.nanmin(r1)), "first_max": float(np.nanmax(r1)),
        "cert_within_1_5": boot_frac(rr, lambda x: float(np.mean((x > 1 / 1.5) & (x < 1.5)))),
        "first_within_10pct": boot_frac(r1[np.isfinite(r1)], lambda x: float(np.mean(np.abs(x - 1) <= 0.10)))}
    h = R["heldout_summary"]
    print(f"\n  {len(rows)} held-out cells: certificate ratio median {h['cert_ratio']['point']:.2f} "
          f"CI [{h['cert_ratio']['ci'][0]:.2f}, {h['cert_ratio']['ci'][1]:.2f}], range "
          f"[{h['cert_min']:.2f}, {h['cert_max']:.2f}], within 1.5x "
          f"{h['cert_within_1_5']['point'] * 100:.0f}%")
    print(f"  first-order prediction: median {h['first_ratio']['point']:.2f} "
          f"CI [{h['first_ratio']['ci'][0]:.2f}, {h['first_ratio']['ci'][1]:.2f}], range "
          f"[{h['first_min']:.2f}, {h['first_max']:.2f}], within 10% "
          f"{h['first_within_10pct']['point'] * 100:.0f}% "
          f"CI [{h['first_within_10pct']['ci'][0] * 100:.0f}, {h['first_within_10pct']['ci'][1] * 100:.0f}]%")


def main():
    R["config"] = {"bootstrap": NB, "seed": 20261127, "c": C_CERT}
    section1()
    section2()
    section3()
    section4()
    out = os.path.join(HERE, "pilot16_results.json")
    with open(out, "w") as f:
        json.dump(R, f, indent=1, default=float)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
