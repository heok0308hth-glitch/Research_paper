#!/usr/bin/env python3
"""
Pilot 21b - per-matrix step counts on the real momentum matrices.

Pilot 21 (stage d) reports a step sweep as a median over 80 matrices.  A per-matrix rule
would need to know whether each matrix wants the same step count, so this script runs
bf16 at K=1..6 on every matrix in the design window of pilot 21 (ell0=1e-2, cushion
0.02, safety 1.01, Frobenius input normalisation) and records the window error, the
smallest K within 1.15x of that matrix's best error, ell and the number of directions
inside the window.

Output: pilot21b_kmin.json (histogram and one row per matrix).  Deterministic; reads
pilot21_momenta.npz written by pilot21_loss_phase_and_scheduler.py --stage b1.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pilot21_loss_phase_and_scheduler import (  # noqa: E402
    CUSH, ELL0, WINDOW, coefs, exact_polar, norm_est, polar, window_err,
)

NPZ = os.path.join(HERE, "pilot21_momenta.npz")
OUT = os.path.join(HERE, "pilot21b_kmin.json")
TOL = 1.15
KS = [1, 2, 3, 4, 5, 6]


def main():
    z = np.load(NPZ)
    mats = [np.asarray(z[f"W{l}"][t], dtype=np.float64)
            for l in range(4) for t in range(z["W0"].shape[0])]
    rows = []
    for i, M in enumerate(mats):
        nev = norm_est(M)
        ref, sv, vt = exact_polar(M)
        floor = WINDOW * nev
        e = {}
        for K in KS:
            X, _ = polar(M, coefs(K, 1.01), "bf16", ["bf16"] * K, nev)
            e[K] = window_err(X, ref, vt, sv, floor)
        best = min(e.values())
        kmin = next(K for K in KS if e[K] <= TOL * best)
        s = sv / nev
        rows.append({
            "l": i // z["W0"].shape[0],
            "t": i % z["W0"].shape[0],
            "e": {str(K): float(e[K]) for K in KS},
            "kmin": int(kmin),
            "ell": float(s[-1]),
            "nwin": int(max(1, int(np.sum(s >= WINDOW)))),
        })
    hist = {}
    for r in rows:
        hist[str(r["kmin"])] = hist.get(str(r["kmin"]), 0) + 1
    out = {"note": "bf16, K=1..6, design ell0=1e-2, cushion 0.02, safety 1.01; "
                   "kmin = smallest K within 1.15x of the matrix's best window error",
           "kmin_hist": {k: hist[k] for k in sorted(hist)}, "rows": rows}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("kmin histogram:", out["kmin_hist"], "of", len(rows))


if __name__ == "__main__":
    main()
