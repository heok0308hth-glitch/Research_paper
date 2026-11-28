#!/usr/bin/env python3
"""
Pilot 7b - What is the direction factor c in  err ~ E_K + c * u * S_pert ?

pilot6 fixed c = 0.142 by calibrating on d = 64 and reported a residual band
err/(u*S_pert) = 0.150..0.227 at L = 1e-2 and 0.097..0.158 at L = 1e-3, i.e. a 2.3x
variation with no derivation.  This script tests whether that variation is a
spectral-weighted effect, in three steps:

  [A] the measured factor over the calibration grid (it must be reproduced first);
  [B] model 1 - linear superposition with the *full* product of the orbit gains.
      Each injected rounding has a per-direction mass ~ u/(2 sqrt d), so the error
      should be (u/2) * RMS over direction pairs of the gain product.  Measured
      against the kernel;
  [C] model 2 - the same superposition but with the prefix ratchet per direction,
      which is what the certificate itself uses.  Model 1 has a fatal structural
      defect (it predicts a floor that keeps falling with K while the measured one
      is flat), model 2 does not;
  [D] whether either model ranks designs (the question pilot7 raises);
  [E] conclusion: what c is and what it is not.

Kernel model as in pilot1c/pilot5b/pilot6/pilot7.
"""
import json
import os
import numpy as np

from pilot7_pert_aware_design import (pe_coeffs, orbit, stats, mat_run, make_M, q,
                                      design_grid, damp_first, spearman)

HERE = os.path.dirname(os.path.abspath(__file__))
R = {"config": {}}


# ---------------------------------------------------------------- gain profiles
def pair_gain(co, l, x):
    """|p'| on the diagonal, |gain| off the diagonal, for every pair of the grid"""
    a, b, c = co[l]
    D = np.abs(a + 3 * b * x ** 2 + 5 * c * x ** 4)
    H = np.abs(a + b * (x[:, None] ** 2 + x[None, :] ** 2)
               + c * (x[:, None] ** 4 + x[None, :] ** 2 * (x[:, None] ** 2 + x[None, :] ** 2)))
    np.fill_diagonal(H, D)
    return H


def gain_profiles(co, grid, start=0):
    """full[t] = product of gains from step start..t ; ratchet = elementwise max over t"""
    tr = orbit(co, grid)
    cur = np.ones((grid.size, grid.size))
    rat = np.zeros_like(cur)
    full = []
    for t in range(start, len(co)):
        cur = cur * pair_gain(co, t, tr[t])
        rat = np.maximum(rat, cur)
        full.append(cur)
    return rat, (full[-1] if full else cur)


def per_dir_mass(U, Vt, dX, u, d):
    """RMS of the coefficients of dX in the singular basis, in units of u/(2 sqrt d)"""
    C = U.T @ dX @ Vt
    return float(np.sqrt(np.mean(C ** 2))) / (u / (2 * np.sqrt(d)))


def main():
    R["config"] = {"kernel": "X,B stored in p significant bits, A not stored, products float64",
                   "note": "per-direction mass of an entrywise relative rounding is u/(2 sqrt d) for a "
                           "flat matrix; the measured factor kappa is reported below"}
    Ls = [1e-2, 1e-3, 1e-4]
    ds = [64, 128]
    ps = [8, 11, 14]
    Ks = [5, 6, 7, 8, 9]

    # ---------------- A: the measured direction factor
    print("=" * 112)
    print("[A] the measured direction factor  c = err / (u * S_pert)  (shipped design, cushion 0.02, safety 1.01)")
    print("=" * 112)
    print(f"  {'L':>7s} {'K':>2s} {'d':>4s} {'p':>3s} {'err':>10s} {'E_K':>10s} {'S_pert':>9s} {'err/u':>8s} {'c':>7s}")
    cells = []
    for L in Ls:
        grid = np.logspace(np.log10(L), 0, 1500)
        for K in Ks:
            co, _ = pe_coeffs(L, K, 0.02, 1.01)
            EK, S, _ = stats(co, grid, K)
            for d in ds:
                M, ref = make_M(d, L)
                for p in ps:
                    u = 2.0 ** -p
                    try:
                        with np.errstate(over="ignore", invalid="ignore"):
                            err = float(np.linalg.norm(mat_run(co, M, p) - ref, 2))
                    except Exception:
                        err = float("inf")
                    if not np.isfinite(err):
                        continue
                    c = err / (u * S)
                    cells.append({"L": L, "K": K, "d": d, "p": p, "u": u, "err": err, "EK": EK,
                                  "S_pert": S, "c": c})
                    print(f"  {L:7g} {K:2d} {d:4d} {p:3d} {err:10.3e} {EK:10.3e} {S:9.1f} {err/u:8.2f} {c:7.4f}")
    R["A_measured_c"] = cells
    print("\n  band by dynamic range (only cells where the floor dominates, E_K <= 0.2 err):")
    for L in Ls:
        v = [r["c"] for r in cells if r["L"] == L and r["EK"] <= 0.2 * r["err"]]
        if v:
            print(f"   L={L:<7g} n={len(v):3d}  c = {min(v):.4f} .. {max(v):.4f}   (max/min {max(v)/min(v):.2f}x)")
            R.setdefault("A_band", {})[str(L)] = [float(min(v)), float(max(v)), float(max(v) / min(v))]

    # ---------------- B/C: two superposition models
    print("\n" + "=" * 112)
    print("[B/C] two spectral superposition models, and the mass factor kappa actually injected")
    print("=" * 112)
    print(f"  {'L':>7s} {'K':>2s} {'d':>4s} {'p':>3s} {'S_pert':>8s} {'maxG_full':>10s} {'RMS_full':>9s} "
          f"{'maxG_rat':>9s} {'RMS_rat':>8s} {'c_model1':>9s} {'c_model2':>9s} {'c_meas':>7s}")
    model = []
    for L in Ls:
        grid = np.logspace(np.log10(L), 0, 1500)
        for K in Ks:
            co, _ = pe_coeffs(L, K, 0.02, 1.01)
            EK, S, _ = stats(co, grid, K)
            for d in ds:
                g = np.logspace(np.log10(L), 0, d)
                rat, full = gain_profiles(co, g)
                rms_f = float(np.sqrt(np.mean(full ** 2)))
                rms_r = float(np.sqrt(np.mean(rat ** 2)))
                M, ref = make_M(d, L)
                U, _, Vt = np.linalg.svd(M)
                X0 = M / np.linalg.norm(M, 2)
                for p in ps:
                    u = 2.0 ** -p
                    try:
                        with np.errstate(over="ignore", invalid="ignore"):
                            err = float(np.linalg.norm(mat_run(co, M, p) - ref, 2))
                    except Exception:
                        err = float("inf")
                    if not np.isfinite(err):
                        continue
                    kap = per_dir_mass(U, Vt, q(X0, p) - X0, u, d)
                    model.append({"L": L, "K": K, "d": d, "p": p, "u": u, "err": err, "EK": EK, "S_pert": S,
                                  "rms_full": rms_f, "rms_rat": rms_r, "max_full": float(full.max()),
                                  "max_rat": float(rat.max()), "kappa": kap,
                                  "c": err / (u * S), "c_model1": rms_f / (2 * S), "c_model2": kap * rms_r / (2 * S),
                                  "pred1": kap * (u / 2) * rms_f, "pred2": kap * (u / 2) * rms_r + EK})
                    print(f"  {L:7g} {K:2d} {d:4d} {p:3d} {S:8.1f} {full.max():10.1f} {rms_f:9.2f} "
                          f"{rat.max():9.1f} {rms_r:8.2f} {rms_f/(2*S):9.5f} {kap*rms_r/(2*S):9.4f} {err/(u*S):7.4f}")
    R["B_model"] = model
    ks = [r["kappa"] for r in model]
    print(f"\n  kappa (per-direction injected mass / (u/2sqrt(d))): {min(ks):.2f} .. {max(ks):.2f}")

    print("\n  the structural test: does each model's floor fall with K the way the measured one does?")
    for L in Ls:
        for p in ps:
            line = f"  L={L:<7g} p={p:<3d} "
            for K in Ks:
                r = [s for s in model if s["L"] == L and s["d"] == 64 and s["p"] == p and s["K"] == K]
                if r:
                    line += f" K{K}: meas {r[0]['err']:.1e} m1 {r[0]['pred1']:.1e} m2 {r[0]['pred2']:.1e}"
            print(line)

    # ---------------- D: ranking across designs
    print("\n" + "=" * 112)
    print("[D] do the models rank designs?  (L=1e-3, K=8, d=64, p=8, 39 designs of pilot7's grid)")
    print("=" * 112)
    L, K, d, p = 1e-3, 8, 64, 8
    grid = np.logspace(np.log10(L), 0, 1500)
    g = np.logspace(np.log10(L), 0, d)
    M, ref = make_M(d, L)
    U, _, Vt = np.linalg.svd(M)
    u = 2.0 ** -p
    kap = per_dir_mass(U, Vt, q(M / np.linalg.norm(M, 2), p) - M / np.linalg.norm(M, 2), u, d)
    rows = []
    for nm, co in design_grid(L, K).items():
        EK, S, _ = stats(co, grid, K)
        rat, full = gain_profiles(co, g)
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                err = float(np.linalg.norm(mat_run(co, M, p) - ref, 2))
        except Exception:
            err = float("inf")
        if not np.isfinite(err) or err > 1.0:
            continue
        rows.append({"design": nm, "err": err, "S_pert": S, "EK": EK,
                     "c_meas": err / (u * S), "pred_cert": EK + 0.142 * u * S,
                     "pred_m2": EK + kap * (u / 2) * float(np.sqrt(np.mean(rat ** 2))),
                     "rms_rat": float(np.sqrt(np.mean(rat ** 2)))})
    rows.sort(key=lambda r: r["err"])
    print(f"  {'design':20s} {'err':>10s} {'S_pert':>9s} {'c_meas':>8s} {'cert (c=.142)':>14s} {'model2':>10s} "
          f"{'cert/err':>9s} {'model2/err':>10s}")
    for r in rows:
        print(f"  {r['design']:20s} {r['err']:10.3e} {r['S_pert']:9.0f} {r['c_meas']:8.4f} "
              f"{r['pred_cert']:14.3e} {r['pred_m2']:10.3e} {r['pred_cert']/r['err']:9.2f} {r['pred_m2']/r['err']:10.2f}")
    sp_c = spearman([r["pred_cert"] for r in rows], [r["err"] for r in rows])
    sp_m = spearman([r["pred_m2"] for r in rows], [r["err"] for r in rows])
    print(f"\n  Spearman(prediction, measured error) over {len(rows)} designs: "
          f"certificate {sp_c:+.2f}  |  model2 {sp_m:+.2f}   (1.0 = perfect ranking)")
    print(f"  measured error spread of these designs: {rows[-1]['err']/rows[0]['err']:.2f}x;  "
          f"their c_meas spread: {min(r['c_meas'] for r in rows):.4f} .. {max(r['c_meas'] for r in rows):.3f}")
    R["D_ranking"] = {"rows": rows, "spearman_cert": sp_c, "spearman_model2": sp_m, "kappa": kap}

    out = os.path.join(HERE, "pilot7b_results.json")
    with open(out, "w") as f:
        json.dump(R, f, indent=1, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
