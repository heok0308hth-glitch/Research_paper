#!/usr/bin/env python3
"""
Pilot 7c - Table 1 of the outline, measured in one script.

The outline's precision table was assembled from several earlier runs and, for
several columns, still quoted the terminal sensitivity S(K) instead of the
ratchet S_pert.  This script re-measures the shipped design (cushion 0.02,
safety 1.01) at every reported bit width and both matrix sizes, adds the bit
widths that were missing (9, 13, 15, 20), and prints measured / predicted / ratio
for the two-term law  err ~ E_K + 0.142 * 2^-p * S_pert.

Kernel model as before: X and B stored in p significant bits, A = XX^T not
stored, products accumulated in float64.
"""
import json
import os
import numpy as np

from pilot7_pert_aware_design import pe_coeffs, stats, mat_run, make_M, q

HERE = os.path.dirname(os.path.abspath(__file__))
C = 0.142
R = {"config": {"c": C, "design": "cushion 0.02, safety 1.01"}, "rows": {}}
PS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 24]


def run(L, K, d, p, M=None, ref=None):
    if M is None:
        M, ref = make_M(d, L)
    co, _ = pe_coeffs(L, K, 0.02, 1.01)
    grid = np.logspace(np.log10(L), 0, 1500)
    EK, S, _ = stats(co, grid, K)
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            X = mat_run(co, M, p)
        err = float(np.linalg.norm(X - ref, 2))
        if not np.isfinite(err):
            err = float("inf")
    except Exception:
        err = float("inf")
    u = 2.0 ** -p
    pred = EK + C * u * S
    return {"L": L, "K": K, "d": d, "p": p, "u": u, "EK": EK, "S_pert": S, "err": err,
            "pred": pred, "ratio": pred / err if np.isfinite(err) and err > 0 else float("nan")}


def main():
    print("=" * 110)
    print("Table 1 re-measured: shipped design, err ~ E_K + 0.142 * 2^-p * S_pert")
    print("=" * 110)
    for L, K in ((1e-2, 6), (1e-3, 6), (1e-3, 10), (1e-4, 10)):
        for d in (64, 128):
            M, ref = make_M(d, L)
            rows = [run(L, K, d, p, M, ref) for p in PS]
            R["rows"][f"L{L}_K{K}_d{d}"] = rows
            print(f"\n L={L:g}  K={K}  d={d}   E_K={rows[0]['EK']:.3e}  S_pert={rows[0]['S_pert']:.1f}")
            print(f"   {'p':>3s} {'u':>9s} {'measured':>11s} {'predicted':>11s} {'pred/meas':>10s} {'err/u':>9s}")
            for r in rows:
                print(f"   {r['p']:3d} {r['u']:9.2e} {r['err']:11.3e} {r['pred']:11.3e} {r['ratio']:10.2f} "
                      f"{r['err'] / r['u']:9.2f}")

    print("\n" + "=" * 110)
    print("certified stop K*(u) = min{K : E_K <= u * S_pert(K)} and the error at that K")
    print("=" * 110)
    stop = {}
    for L in (1e-2, 1e-3, 1e-4):
        grid = np.logspace(np.log10(L), 0, 1500)
        Ms = {}
        for d in (64, 128):
            Ms[d] = make_M(d, L)
        for p in PS:
            u = 2.0 ** -p
            K = 2
            while K <= 14:
                co, _ = pe_coeffs(L, K, 0.02, 1.01)
                EK, S, _ = stats(co, grid, K)
                if EK <= u * S:
                    break
                K += 1
            row = {"Kstar": K, "EK": EK, "S_pert": S, "pred": EK + C * u * S}
            for d in (64, 128):
                r = run(L, K, d, p, Ms[d][0], Ms[d][1])
                row[f"err_d{d}"] = r["err"]
                row[f"ratio_d{d}"] = r["ratio"]
            stop[f"L{L}_p{p}"] = row
        print(f"\n L={L:g}")
        print(f"   {'p':>3s} {'K*':>3s} {'E_K*':>10s} {'S_pert':>8s} {'pred':>10s} {'err d=64':>10s} "
              f"{'pred/meas':>10s} {'err d=128':>10s} {'pred/meas':>10s}")
        for p in PS:
            r = stop[f"L{L}_p{p}"]
            print(f"   {p:3d} {r['Kstar']:3d} {r['EK']:10.3e} {r['S_pert']:8.1f} {r['pred']:10.3e} "
                  f"{r['err_d64']:10.3e} {r['ratio_d64']:10.2f} {r['err_d128']:10.3e} {r['ratio_d128']:10.2f}")
    R["stop"] = stop

    out = os.path.join(HERE, "pilot7c_results.json")
    with open(out, "w") as f:
        json.dump(R, f, indent=1, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
