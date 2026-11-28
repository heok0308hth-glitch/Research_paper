#!/usr/bin/env python3
"""
Pilot 10b - the epsilon-target stopping rule and the graded precision schedule.

[1] Stopping.  The certificate err ~ E_K + c u S_pert(K) turns a target accuracy
    into a step count computable from the coefficients alone (no matrix):
        K(eps) = min{ K : E_K + c u S_pert(K) <= eps }.
    Measured against the realised p-bit error: the reachable target range (the
    floor min_K J(K) bounds it below), the overshoot meas/eps, monotonicity in
    eps, and the response to c being wrong by up to 7x.

[2] Allocation.  Pilot 8b reported "the floor is 84% first-step and bits belong
    to the first step".  [A] separates the three things that were mixed there:
    the input cast, step 0's arithmetic, and the remaining steps.  The premium is
    the first quantisation of an ILL-CONDITIONED state (the input cast, and to a
    smaller degree step 0); once the state's spectrum is inside [1-eps,1+eps] a
    rounding costs ~u.  [A2] measures the per-slot gain profile

        beta_t = subspace error caused by rounding slot t alone, in units of u,

    which decays roughly geometrically along the schedule.  With a max structure
    (measured: err / max_t = 1.0-1.2) the allocation problem is

        minimise max_t kappa_t 2^-p_t    s.t.  sum_t p_t <= B,

    whose solution is the graded (inverse-waterfilling) schedule
        p_t* = log2(kappa_t) + (B - sum_s log2 kappa_s)/n.
    [B] measures it against uniform allocation at the same total bits.

Kernel model as in pilots 1-10; slot 0 is the input cast q(M/||M||, p_0), slots
1..K are the iteration steps.
"""
import json
import os
import numpy as np

from pilot7_pert_aware_design import q, pe_coeffs, make_M
from pilot10_subspace_theory import polar_safe, n2, orbit, start

HERE = os.path.dirname(os.path.abspath(__file__))
R = {}
CUSH, SAFE = 0.02, 1.01
C_CERT = 0.142
EXACT = 60
KMAX = 12


def coeff_stats(L, K):
    co, _ = pe_coeffs(L, K, CUSH, SAFE)
    g = np.logspace(np.log10(L), 0, 400)
    x = g.copy()
    pref, cur = [], None
    for a, b, c in co:
        d = np.abs(a + 3 * b * x ** 2 + 5 * c * x ** 4)
        cur = d[None, :] if cur is None else np.vstack([cur * d[None, :], d[None, :]])
        pref.append(float(cur.max()))
        x = a * x + b * x ** 3 + c * x ** 5
    return co, float(np.max(np.abs(x - 1))), max(pref), pref


def run(co, Mn, p0, ps):
    with np.errstate(over="ignore", invalid="ignore"):
        X, _ = orbit(co, q(Mn, p0), ps)
    return X if np.all(np.isfinite(X)) else None


# ------------------------------------------------------------------ [A]
def part_A():
    print("=" * 108)
    print("[A] where the subspace floor comes from: the input cast, step 0's arithmetic, or the tail")
    print("=" * 108)
    print(f"  {'ell':>7s} {'K':>2s} {'p':>3s} {'all at p':>10s} {'input exact':>11s} "
          f"{'step0+ exact':>12s} {'input+step0 exact':>17s} {'steps exact':>11s} {'all exact':>10s}")
    R["decomposition"] = {}
    for L in (1e-2, 1e-3, 1e-4):
        for K in (6, 8):
            for p in (8, 11):
                M, ref = make_M(64, L, 1)
                Mn, _ = start(M, p)
                co, _, _, _ = coeff_stats(L, K)
                variants = {
                    "all_p": (p, [p] * K),
                    "input_exact": (EXACT, [p] * K),
                    "step0_exact": (p, [EXACT] + [p] * (K - 1)),
                    "input_step0_exact": (EXACT, [EXACT] + [p] * (K - 1)),
                    "steps_exact": (p, [EXACT] * K),
                    "all_exact": (EXACT, [EXACT] * K),
                }
                row = {}
                for nm, (p0, ps) in variants.items():
                    X = run(co, Mn, p0, ps)
                    pl = polar_safe(X) if X is not None else None
                    row[nm] = n2(pl - ref) if pl is not None else float("nan")
                R["decomposition"][f"L{L:g}_K{K}_p{p}"] = row
                print(f"  {L:7g} {K:2d} {p:3d} {row['all_p']:10.3e} {row['input_exact']:11.3e} "
                      f"{row['step0_exact']:12.3e} {row['input_step0_exact']:17.3e} "
                      f"{row['steps_exact']:11.3e} {row['all_exact']:10.3e}")
    v = list(R["decomposition"].values())
    gi = [x["all_p"] / x["input_exact"] for x in v]
    g0 = [x["all_p"] / x["step0_exact"] for x in v]
    gs = [x["all_p"] / x["input_step0_exact"] for x in v]
    print(f"\n  median gain: input cast exact {np.median(gi):.2f}x | step 0 arithmetic exact "
          f"{np.median(g0):.2f}x (range {min(g0):.2f}-{max(g0):.2f}x) | both {np.median(gs):.2f}x "
          f"(range {min(gs):.2f}-{max(gs):.2f}x)")
    R["gain_summary"] = {"input_exact_median": float(np.median(gi)),
                         "step0_exact_median": float(np.median(g0)),
                         "step0_exact_min": float(min(g0)),
                         "step0_exact_max": float(max(g0)),
                         "input_step0_median": float(np.median(gs)),
                         "min": float(min(gs)), "max": float(max(gs))}


# ------------------------------------------------------------------ [A2]
def part_A2():
    print("\n" + "=" * 108)
    print("[A2] per-slot gain profile  beta_t = (subspace error from rounding slot t alone) / u")
    print("=" * 108)
    R["profile"] = {}
    print(f"  {'ell':>7s} {'p':>3s} {'input':>7s} " + " ".join(f"{'step'+str(t):>7s}" for t in range(6))
          + f" {'pred/inf':>8s}")
    for L in (1e-2, 1e-3, 1e-4):
        for p in (8, 11):
            u = 2.0 ** -p
            M, ref = make_M(64, L, 1)
            Mn, _ = start(M, p)
            K = 6
            co, _, _, _ = coeff_stats(L, K)
            betas = []
            for t in range(K + 1):
                p0, ps = EXACT, [EXACT] * K
                if t == 0:
                    p0 = p
                else:
                    ps[t - 1] = p
                X = run(co, Mn, p0, ps)
                pl = polar_safe(X) if X is not None else None
                betas.append(n2(pl - ref) / u if pl is not None else float("nan"))
            R["profile"][f"L{L:g}_p{p}"] = betas
            print(f"  {L:7g} {p:3d} {betas[0]:7.1f} " + " ".join(f"{b:7.1f}" for b in betas[1:])
                  + f" {betas[1]/max(betas[-2:]):8.1f}")

    print("\n  the same profile over step count K (input slot keeps its gain, tail slot ~ 1u):")
    R["profile_K"] = {}
    for L in (1e-2, 1e-3):
        for K in (4, 6, 8, 10):
            M, ref = make_M(64, L, 1)
            Mn, _ = start(M, 8)
            co, _, _, _ = coeff_stats(L, K)
            X = run(co, Mn, 8, [EXACT] * K)
            pl = polar_safe(X) if X is not None else None
            b_in = n2(pl - ref) / 2.0 ** -8 if pl is not None else float("nan")
            X = run(co, Mn, EXACT, [EXACT] * (K - 1) + [8])
            pl = polar_safe(X) if X is not None else None
            b_tail = n2(pl - ref) / 2.0 ** -8 if pl is not None else float("nan")
            R["profile_K"][f"L{L:g}_K{K}"] = {"beta_input": b_in, "beta_last": b_tail}
            print(f"    ell={L:<6g} K={K:2d}: beta_input={b_in:7.1f}  beta_last={b_tail:5.1f}  "
                  f"premium={np.log2(b_in/b_tail):5.2f} bits")


# ------------------------------------------------------------------ [1]
def part_stop():
    print("\n" + "=" * 108)
    print("[1] epsilon-target stopping rule  K(eps) = min{K : E_K + c u S_pert(K) <= eps}")
    print("=" * 108)
    EPS = (3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
    R["stopping"] = {}
    ratios, mono_ok, mono_tot, n_def = [], 0, 0, 0
    print(f"  {'d':>3s} {'ell':>7s} {'p':>3s} {'floor':>9s} {'K_fl':>4s} {'eps':>7s} {'K(eps)':>6s} "
          f"{'pred':>9s} {'meas':>9s} {'meas/eps':>8s} {'K_or':>4s}")
    for d in (64, 128):
        for L in (1e-2, 1e-3, 1e-4):
            tab = {K: coeff_stats(L, K) for K in range(1, KMAX + 1)}
            for p in (8, 10, 12, 16):
                u = 2.0 ** -p
                M, ref = make_M(d, L, 1)
                Mn, _ = start(M, p)
                J = {K: tab[K][1] + C_CERT * u * tab[K][2] for K in tab}
                Kfl = min(J, key=J.get)
                key = f"d{d}_L{L:g}_p{p}"
                R["stopping"][key] = {"floor": J[Kfl], "K_floor": Kfl, "cells": []}
                for eps in EPS:
                    Ks = [K for K in tab if J[K] <= eps]
                    if not Ks:
                        R["stopping"][key]["cells"].append({"eps": eps, "K": None})
                        continue
                    K = min(Ks)
                    co = tab[K][0]
                    X = run(co, Mn, p, [p] * K)
                    pl = polar_safe(X) if X is not None else None
                    meas = n2(X - ref) if pl is not None and X is not None else float("nan")
                    # oracle: which K actually minimises the measured error
                    orc = None
                    for K2 in tab:
                        X2 = run(tab[K2][0], Mn, p, [p] * K2)
                        if X2 is None:
                            continue
                        e2 = n2(X2 - ref)
                        if orc is None or e2 < orc[1]:
                            orc = (K2, e2)
                    R["stopping"][key]["cells"].append(
                        {"eps": eps, "K": K, "pred": J[K], "meas": meas,
                         "oracle_K": orc[0] if orc else None,
                         "oracle_err": orc[1] if orc else None})
                    if np.isfinite(meas):
                        ratios.append(meas / eps)
                        n_def += 1
                    print(f"  {d:3d} {L:7g} {p:3d} {J[Kfl]:9.2e} {Kfl:4d} {eps:7.0e} {K:6d} "
                          f"{J[K]:9.2e} {meas:9.2e} {meas/eps:8.2f} {orc[0] if orc else 0:4d}")
                prev = None
                for c in R["stopping"][key]["cells"]:
                    if c["K"] is not None:
                        if prev is not None:
                            mono_tot += 1
                            mono_ok += int(c["K"] <= prev)
                        prev = c["K"]
    ratios = np.array(ratios)
    print(f"\n  defined cells {n_def}: meas/eps median {np.median(ratios):.2f}, "
          f"max {ratios.max():.2f}, min {ratios.min():.2f}, "
          f"<=1.5 in {int((ratios <= 1.5).sum())}/{len(ratios)}, <=1.0 in {int((ratios <= 1.0).sum())}")
    print(f"  monotone in eps: {mono_ok}/{mono_tot}")
    R["stopping_summary"] = {"n": int(len(ratios)), "median": float(np.median(ratios)),
                            "max": float(ratios.max()), "min": float(ratios.min()),
                            "le_1p5": int((ratios <= 1.5).sum()), "le_1p0": int((ratios <= 1.0).sum()),
                            "monotone": f"{mono_ok}/{mono_tot}"}

    print("\n  [1b] sensitivity to the constant c: (K, measured error)")
    print(f"  {'cell':>24s} " + " ".join(f"{'c='+str(c):>16s}" for c in (0.02, 0.142, 0.5, 1.0)))
    R["c_sensitivity"] = {}
    for L in (1e-2, 1e-3):
        tab = {K: coeff_stats(L, K) for K in range(1, KMAX + 1)}
        d, p = 64, 8
        u = 2.0 ** -p
        M, ref = make_M(d, L, 1)
        Mn, _ = start(M, p)
        for eps in (3e-1, 1e-1, 3e-2):
            out = {}
            for c in (0.02, 0.142, 0.5, 1.0):
                Ks = [K for K in tab if tab[K][1] + c * u * tab[K][2] <= eps]
                if not Ks:
                    out[str(c)] = None
                    continue
                K = min(Ks)
                X = run(tab[K][0], Mn, p, [p] * K)
                out[str(c)] = {"K": K, "meas": n2(X - ref) if X is not None else float("nan")}
            R["c_sensitivity"][f"L{L:g}_eps{eps:g}"] = out
            print(f"  {f'ell={L:g} eps={eps:g}':>24s} " +
                  " ".join(f"{(str(v['K'])+' / '+format(v['meas'],'.2e')) if v else '--':>16s}"
                           for v in out.values()))


# ------------------------------------------------------------------ [2]
def part_alloc():
    print("\n" + "=" * 108)
    print("[2] graded schedule vs uniform at equal total bits")
    print("      rule: p_t* = log2(kappa_t) + (B - sum log2 kappa_t)/n,  kappa from the profile")
    print("=" * 108)
    R["alloc"] = {}
    print(f"  {'ell':>7s} {'K':>2s} {'B':>4s} {'uniform':>8s} {'err_uni':>9s} {'schedule':>26s} "
          f"{'bits':>5s} {'err_grad':>9s} {'gain':>6s} {'pred_ratio':>10s}")
    for d in (64, 128):
        for L in (1e-2, 1e-3, 1e-4):
            for K in (6, 8):
                M, ref = make_M(d, L, 1)
                Mn, _ = start(M, 1)
                co, EK, Sp, _ = coeff_stats(L, K)
                # measured profile at p=10 (mid range) for this (L,K,d)
                prof = []
                for t in range(K + 1):
                    p0, ps = EXACT, [EXACT] * K
                    if t == 0:
                        p0 = 10
                    else:
                        ps[t - 1] = 10
                    X = run(co, Mn, p0, ps)
                    pl = polar_safe(X) if X is not None else None
                    prof.append(max(n2(pl - ref) / 2.0 ** -10, 1.0) if pl is not None else 1.0)
                R["alloc"].setdefault(f"d{d}_L{L:g}_K{K}", {})["profile_p10"] = prof
                n = K + 1
                for per in (8, 10, 12, 14):
                    B = n * per
                    s = float(np.sum(np.log2(prof)))
                    base = (B - s) / n
                    ps = [int(max(6, min(30, round(np.log2(x) + base)))) for x in prof]
                    Xu = run(co, Mn, per, [per] * K)
                    Xg = run(co, Mn, ps[0], ps[1:])
                    eu = n2(Xu - ref) if Xu is not None else float("inf")
                    eg = n2(Xg - ref) if Xg is not None else float("inf")
                    pred = max(p * 2.0 ** -pp for p, pp in zip(prof, ps))
                    R["alloc"][f"d{d}_L{L:g}_K{K}"][f"B{B}"] = {
                        "uniform": per, "err_uniform": eu, "schedule": ps,
                        "bits": sum(ps), "err_graded": eg,
                        "gain": eu / eg if eg else None, "pred": pred,
                        "pred_ratio": pred / eg if eg else None}
                    print(f"  {L:7g} {K:2d} {B:4d} {per:8d} {eu:9.2e} {str(ps):>26s} "
                          f"{sum(ps):5d} {eg:9.2e} {eu/eg:6.2f} {pred/eg:10.2f}")
    g = [v["gain"] for k, dct in R["alloc"].items() for kk, v in dct.items()
         if isinstance(v, dict) and v.get("gain")]
    g = np.array(g)
    print(f"\n  graded vs uniform at equal total bits: median {np.median(g):.2f}x, "
          f"range {g.min():.2f}-{g.max():.2f}x (n={len(g)})")
    R["alloc_summary"] = {"median": float(np.median(g)), "min": float(g.min()),
                          "max": float(g.max()), "n": int(len(g))}


def main():
    part_A()
    part_A2()
    part_stop()
    part_alloc()
    out = os.path.join(HERE, "pilot10b_results.json")
    with open(out, "w") as f:
        json.dump(R, f, indent=1, default=float)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
