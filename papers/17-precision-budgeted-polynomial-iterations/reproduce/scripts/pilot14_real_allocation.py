#!/usr/bin/env python3
"""
Pilot 14 - the graded bit schedule on the real momentum spectra.

Pilot 10b established the allocation rule on synthetic logspace spectra (48 cells,
median 3.06x over uniform precision at equal total bits) and pilot 11 measured the
per-slot gains of the subspace channel on 500 Muon momentum matrices (128x32).  This
script applies the schedule to those matrices and measures what a caller gets:

  [A] per-slot gains kappa_t = err(only slot t at p_ref)/u on a sample of the real
      momentum matrices, for K in {4,6,8}.  On the real range (median ell = 0.15)
      the first-slot premium is small, so this is where the rule must show whether
      it survives a spectrum it was not fitted to.
  [B] equal-total-bit cells: K in {4,6,8} x per-slot precision {8,10,12,14}, with
        uniform             p_t = per
        graded (measured)   p_t* = log2 kappa_t + (B - sum_s log2 kappa_s)/n
        graded (closed form) kappa_t = max{(kappa/ell) rho^t, 1.1}, kappa=0.06, rho=0.45
      Integer bits are assigned by largest remainder so that every schedule spends
      exactly B = n*per bits.  Two errors are reported: ||X_K - polar(M)||_2 (what a
      caller sees) and ||polar(X_K) - polar(M)||_2 (the subspace channel alone,
      which is what allocation can repair; the exact error E_K is not a rounding).
  [C] the max model on the real spectra: err against max_t kappa_t 2^-p_t, and the
      predicted/measured ratio of the schedule.
  [D] the gain as a function of the dynamic range alone: the same 128x32 shape with a
      logspace spectrum at ell = 1e-1, 1e-2, 1e-3 (1e-4), so that the allocation
      result of pilot 10b and the flat profile of the real momenta are one law --
      the schedule pays when the first-slot premium in bits exceeds the granularity
      of a whole bit.

Kernel model as in pilots 1-13; the matrices are the frozen exact-arithmetic Muon
momenta of pilot 11.
"""
import json
import os
import time
import numpy as np

from pilot7_pert_aware_design import q, pe_coeffs
from pilot10_subspace_theory import polar_safe

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "pilot11_momentum.npz")
R = {}
CUSH, SAFE = 0.02, 1.01
EXACT = 60
KAPPA_RHO = (0.06, 0.45)          # closed-form profile fitted in pilot 10b
PLATEAU = 1.1


def n2(X):
    return float(np.linalg.norm(X, 2))


def orbit(co, X0, p0, ps):
    X = q(X0, p0)
    for (a, b, c), p in zip(co, ps):
        A = X @ X.T
        B = q(b * A + c * (A @ A), p)
        X = q(a * X + B @ X, p)
    return X


def err_of(co, X0, p0, ps, ref):
    """(||X_K - ref||, ||polar(X_K) - ref||); (None, None) if the orbit left the finite range."""
    with np.errstate(over="ignore", invalid="ignore"):
        X = orbit(co, X0, p0, ps)
        if not np.all(np.isfinite(X)):
            return None, None
        pol = polar_safe(X)
        if pol is None:
            return None, None
        return n2(X - ref), n2(pol - ref)


def round_bits(vals, B, lo=4, hi=30):
    """integer p_t with sum exactly B: largest remainder, each slot adjusted at most once."""
    v = np.floor(np.clip(vals, lo, hi)).astype(int)
    frac = np.asarray(vals, float) - v
    free = np.ones(len(v), bool)
    while v.sum() < B:
        cand = [i for i in range(len(v)) if free[i] and v[i] < hi]
        if not cand:
            raise RuntimeError(f"cannot reach B={B}")
        i = max(cand, key=lambda j: frac[j])
        v[i] += 1
        free[i] = False
    while v.sum() > B:
        cand = [i for i in range(len(v)) if free[i] and v[i] > lo]
        if not cand:
            raise RuntimeError(f"cannot reach B={B}")
        i = min(cand, key=lambda j: frac[j])
        v[i] -= 1
        free[i] = False
    return [int(x) for x in v]


def profile(co, X0, ref, p_ref, K):
    """kappa_t = error from rounding slot t alone at p_ref, in units of u_ref."""
    u = 2.0 ** -p_ref
    out = []
    for t in range(K + 1):
        p0, ps = EXACT, [EXACT] * K
        if t == 0:
            p0 = p_ref
        else:
            ps[t - 1] = p_ref
        ei, ep = err_of(co, X0, p0, ps, ref)
        out.append(ep / u if ep is not None else float("nan"))
    return [max(v, PLATEAU) if np.isfinite(v) else float("nan") for v in out]


def main():
    z = np.load(NPZ)
    upds = z["upds"].astype(float)
    n = len(upds)
    step = max(1, n // 60)
    idx = list(range(0, n, step))[:60]
    print(f"{n} momentum matrices available, {len(idx)} sampled (every {step}th)")

    refs, ells = {}, {}
    for i in idx:
        M = upds[i]
        Mm = n2(M)
        X0 = M / Mm
        pol = polar_safe(X0)
        refs[i] = {"X0": X0, "ref": pol, "ell": float(np.linalg.svd(X0, compute_uv=False)[-1])}
        ells[i] = refs[i]["ell"]
    el = np.array(list(ells.values()))
    R["sample"] = {"n": len(idx), "idx_step": step, "ell_median": float(np.median(el)),
                   "ell_p10": float(np.percentile(el, 10)), "ell_p90": float(np.percentile(el, 90)),
                   "ell_min": float(el.min()), "ell_max": float(el.max())}
    print(f"  sample dynamic range: median {np.median(el):.4f}, "
          f"p10 {np.percentile(el, 10):.4f}, p90 {np.percentile(el, 90):.4f}")

    # ---------------------------------------------------------------- [A]
    print("\n" + "=" * 108)
    print("[A] per-slot gains kappa_t on the real spectra (slot 0 = input cast)")
    print("=" * 108)
    R["profile"] = {}
    for K in (4, 6, 8):
        profs = {}
        for i in idx:
            X0, ref = refs[i]["X0"], refs[i]["ref"]
            co, _ = pe_coeffs(refs[i]["ell"], K, CUSH, SAFE)
            profs[i] = profile(co, X0, ref, 10, K)
        tab = np.array([profs[i] for i in idx], float)
        med = np.array([np.nanmedian(tab[:, t]) for t in range(K + 1)])
        prem = float(np.log2(med[0] / max(med[-1], PLATEAU)))
        R["profile"][f"K{K}"] = {"median": [float(x) for x in med],
                                 "premium_bits": prem,
                                 "p90": [float(np.nanpercentile(tab[:, t], 90)) for t in range(K + 1)]}
        print(f"  K={K}: " + " ".join(f"{x:6.2f}" for x in med)
              + f"   first-slot premium {prem:.2f} bits")

    # ---------------------------------------------------------------- [B]
    print("\n" + "=" * 108)
    print("[B] equal total bits: uniform vs graded (measured profile) vs graded (closed form)")
    print("=" * 108)
    R["alloc"] = {}
    t0 = time.time()
    print(f"  {'K':>2s} {'per':>4s} {'B':>4s} | {'err_uni':>9s} {'err_grad':>9s} {'gain':>5s} "
          f"{'gain_pol':>8s} {'err_closed':>10s} {'gain_cl':>7s} | schedule (median matrix)")
    for K in (4, 6, 8):
        per_prof = {}
        for i in idx:
            X0, ref = refs[i]["X0"], refs[i]["ref"]
            co, _ = pe_coeffs(refs[i]["ell"], K, CUSH, SAFE)
            per_prof[i] = profile(co, X0, ref, 10, K)
        nslot = K + 1
        for per in (8, 10, 12, 14):
            B = nslot * per
            rows = []
            for i in idx:
                X0, ref = refs[i]["X0"], refs[i]["ref"]
                ell = refs[i]["ell"]
                co, _ = pe_coeffs(ell, K, CUSH, SAFE)
                kap = per_prof[i]
                eu_i, eu_p = err_of(co, X0, per, [per] * K, ref)
                if eu_i is None:
                    continue
                s = float(np.sum(np.log2(kap)))
                p_g = round_bits([np.log2(k) + (B - s) / nslot for k in kap], B)
                eg_i, eg_p = err_of(co, X0, p_g[0], p_g[1:], ref)
                kc = KAPPA_RHO[0] / ell
                cf = [max(kc * KAPPA_RHO[1] ** t, PLATEAU) for t in range(nslot)]
                sc = float(np.sum(np.log2(cf)))
                p_c = round_bits([np.log2(k) + (B - sc) / nslot for k in cf], B)
                ec_i, ec_p = err_of(co, X0, p_c[0], p_c[1:], ref)
                pred = max(k * 2.0 ** -pp for k, pp in zip(kap, p_g))
                pmi = float(np.log2(max(kap[0], PLATEAU) / max(kap[-1], PLATEAU)))
                rows.append({"ell": ell, "premium_bits": pmi, "eu": eu_i, "eu_p": eu_p,
                             "eg": eg_i, "eg_p": eg_p, "ec": ec_i, "ec_p": ec_p,
                             "pg": p_g, "pc": p_c, "pred": pred,
                             "sum_g": sum(p_g), "sum_c": sum(p_c)})
            if not rows:
                continue
            g = np.array([r["eu"] / r["eg"] for r in rows if r["eg"]])
            gp = np.array([r["eu_p"] / r["eg_p"] for r in rows if r["eg_p"]])
            gc = np.array([r["eu"] / r["ec"] for r in rows if r["ec"]])
            med_pg = np.median(np.array([r["pg"] for r in rows]), axis=0).astype(int).tolist()
            key = f"K{K}_B{B}"
            R["alloc"][key] = {
                "K": K, "per": per, "B": B, "n": len(rows),
                "gain_median": float(np.median(g)), "gain_min": float(g.min()), "gain_max": float(g.max()),
                "gain_pol_median": float(np.median(gp)),
                "gain_closed_median": float(np.median(gc)),
                "gain_closed_min": float(gc.min()), "gain_closed_max": float(gc.max()),
                "sum_ok": all(r["sum_g"] == B and r["sum_c"] == B for r in rows),
                "median_schedule": med_pg, "rows": rows}
            print(f"  {K:2d} {per:4d} {B:4d} | {np.median([r['eu'] for r in rows]):9.2e} "
                  f"{np.median([r['eg'] for r in rows]):9.2e} {np.median(g):5.2f} "
                  f"{np.median(gp):8.2f} {np.median([r['ec'] for r in rows]):10.2e} "
                  f"{np.median(gc):7.2f} | {med_pg}")

    # ---------------------------------------------------------------- [C]
    print("\n" + "=" * 108)
    print("[C] the max model and the certificate's allocation term on the real spectra")
    print("=" * 108)
    R["maxform"] = {}
    for key, cell in R["alloc"].items():
        rows = cell["rows"]
        ratio = np.array([r["eu"] / r["pred"] for r in rows if r["pred"] > 0])
        psum = np.array([r["eu"] / (r["pred"] * cell["K"] + 1) for r in rows if r["pred"] > 0])
        R["maxform"][key] = {"err_over_max_median": float(np.median(ratio)),
                             "err_over_max_min": float(ratio.min()),
                             "err_over_max_max": float(ratio.max()),
                             "err_over_sum_scaled_median": float(np.median(psum))}
        print(f"  {key:10s}: err/max_t(channel) median {np.median(ratio):.2f} "
              f"[{ratio.min():.2f}, {ratio.max():.2f}]  (sum would give ~1/{cell['K'] + 1} = "
              f"{1 / (cell['K'] + 1):.2f} of the sum)")

    # aggregate over every matrix-cell, and does the gain track the premium?
    allg, prem, ln = [], [], []
    for cell in R["alloc"].values():
        for r in cell["rows"]:
            if r["eg"] and r["ec"]:
                allg.append(r["eu"] / r["eg"])
                prem.append(r["premium_bits"])
                ln.append(np.log(r["eu"] / r["eg"]))
    allg = np.array(allg)
    prem = np.array(prem)
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    rho = float(np.corrcoef(rank(prem), rank(np.array(ln)))[0, 1]) if len(allg) > 2 else None
    R["real_summary"] = {
        "n": int(len(allg)), "gain_median": float(np.median(allg)),
        "gain_mean": float(allg.mean()), "gain_p10": float(np.percentile(allg, 10)),
        "gain_p90": float(np.percentile(allg, 90)), "gain_min": float(allg.min()),
        "gain_max": float(allg.max()),
        "within_5pct": int(np.sum(np.abs(np.log(allg)) <= np.log(1.05))),
        "worse_than_5pct": int(np.sum(allg < 1 / 1.05)), "better_than_5pct": int(np.sum(allg > 1.05)),
        "premium_median": float(np.median(prem)),
        "spearman_premium_gain": rho}
    rs = R["real_summary"]
    print(f"\n  [C2] every real matrix-cell (n={rs['n']}): gain median {rs['gain_median']:.3f}, "
          f"mean {rs['gain_mean']:.3f}, p10 {rs['gain_p10']:.3f}, p90 {rs['gain_p90']:.3f}, "
          f"range {rs['gain_min']:.2f}-{rs['gain_max']:.2f}")
    print(f"       within 5% of uniform {rs['within_5pct']}/{rs['n']}, worse by >5% "
          f"{rs['worse_than_5pct']}, better by >5% {rs['better_than_5pct']}; "
          f"premium median {rs['premium_median']:.2f} bits; "
          f"Spearman(premium, log gain) {rho:.2f}")
    g = [c["gain_median"] for c in R["alloc"].values()]
    gc = [c["gain_closed_median"] for c in R["alloc"].values()]
    R["summary"] = {"gain_median": float(np.median(g)), "gain_min": float(min(g)),
                    "gain_max": float(max(g)), "closed_median": float(np.median(gc)),
                    "closed_min": float(min(gc)), "closed_max": float(max(gc)),
                    "n_cells": len(g), "seconds": time.time() - t0}
    print(f"\n  graded (measured profile) vs uniform, equal total bits: median "
          f"{np.median(g):.2f}x over {len(g)} cells (range {min(g):.2f}-{max(g):.2f}x)")
    print(f"  graded (closed form, ell only)                         : median "
          f"{np.median(gc):.2f}x (range {min(gc):.2f}-{max(gc):.2f}x)")
    print(f"  {time.time() - t0:.1f}s")

    # ---------------------------------------------------------------- [D]
    print("\n" + "=" * 108)
    print("[D] same 128x32 shape, logspace spectrum: premium and gain against ell")
    print("=" * 108)
    R["ell_sweep"] = {}
    print(f"  {'ell':>7s} {'seed':>4s} {'K':>2s} {'premium':>8s} " +
          " ".join(f"{'gain B=' + str((Kk + 1) * pp):>10s}" for Kk in (6,) for pp in (8, 12, 16)))
    for L in (1e-1, 1e-2, 1e-3, 1e-4):
        for seed in (1, 2):
            rs = np.random.RandomState(seed)
            m, nn = 128, 32
            U, _ = np.linalg.qr(rs.randn(m, nn))
            V, _ = np.linalg.qr(rs.randn(nn, nn))
            sv = np.logspace(0, np.log10(L), nn)
            M = (U * sv) @ V.T
            ref = U @ V.T
            X0 = M / n2(M)
            for K in (6, 8):
                co, _ = pe_coeffs(L, K, CUSH, SAFE)
                kap = profile(co, X0, ref, 10, K)
                prem = float(np.log2(max(kap[0], PLATEAU) / max(kap[-1], PLATEAU)))
                row = {"ell": L, "seed": seed, "K": K, "premium_bits": prem, "cells": {}}
                for per in (8, 12, 16):
                    B = (K + 1) * per
                    eu, _ = err_of(co, X0, per, [per] * K, ref)
                    sc = float(np.sum(np.log2(kap)))
                    p_g = round_bits([np.log2(k) + (B - sc) / (K + 1) for k in kap], B)
                    eg, _ = err_of(co, X0, p_g[0], p_g[1:], ref)
                    row["cells"][f"B{B}"] = {"schedule": p_g, "err_uniform": eu, "err_graded": eg,
                                             "gain": (eu / eg) if eu and eg else None}
                R["ell_sweep"][f"L{L:g}_s{seed}_K{K}"] = row
                gs = " ".join(f"{row['cells'][f'B{(K + 1) * pp}']['gain']:10.2f}"
                              if row["cells"][f"B{(K + 1) * pp}"]["gain"] else "         -"
                              for pp in (8, 12, 16))
                print(f"  {L:7g} {seed:4d} {K:2d} {prem:8.2f} {gs}")
    gg = [c["gain"] for r in R["ell_sweep"].values() for c in r["cells"].values() if c["gain"]]
    if gg:
        print(f"\n  all {len(gg)} sweep cells: gain median {np.median(gg):.2f}x, "
              f"range {min(gg):.2f}-{max(gg):.2f}x")

    out = os.path.join(HERE, "pilot14_results.json")
    with open(out, "w") as f:
        json.dump(R, f, indent=1, default=float)
    print("wrote", out)


if __name__ == "__main__":
    main()
