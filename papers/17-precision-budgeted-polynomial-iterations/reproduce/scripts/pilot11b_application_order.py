#!/usr/bin/env python3
"""
Pilot 11b - at fixed design and fixed matrix, how fast does the two-operand
asymmetry of the bit allocation appear?

Section 5.3 claims that the max structure is created by the *second* application
of a step rather than accumulated over K.  The supporting data must hold the
design and the matrix fixed and change only K.  We use the degree-17 minimax step
of pilot4c (ell=0.3, one step on a 64x64 matrix, exact error E_1 = 1.68e-3, small
enough that the rounding is visible at p >= 8) and iterate it for K = 1, 2, 3.

Model exactly as in pilot4c: the state X is stored with pX bits, the intermediate
A = X X^T with pA bits, products in float64.

    X <- q( sum_i a_i A^i X , pX ),   A = q(X X^T, pA).
"""
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "pilot11b_results.json")
PS = (6, 8, 10, 12, 16, 24)


def q(x, p):
    if p is None or p >= 30:
        return np.asarray(x, float)
    x = np.asarray(x, float)
    m, e = np.frexp(x)
    s = float(2 ** (p - 1))
    return np.ldexp(np.round(m * s) / s, e)


def step(coeffs, X, pX, pA):
    A = q(X @ X.T, pA)
    Y = coeffs[0] * X
    P = np.eye(X.shape[0])
    for a in coeffs[1:]:
        P = P @ A
        Y = Y + a * (P @ X)
    return q(Y, pX)


def run(coeffs, M, pX, pA, K):
    X = q(M / np.linalg.norm(M, 2), pX)
    for _ in range(K):
        X = step(coeffs, X, pX, pA)
    return X


def main():
    R = {"config": {"ell": 0.3, "matrix": "64x64, np.random.seed(11) as pilot4c",
                    "design": "degree-17 minimax step of pilot4c"}}
    c4 = json.load(open(os.path.join(HERE, "pilot4c_results.json")))
    coefs = [float(v) for v in c4["design"]["coeffs"]]
    R["config"]["coeffs"] = coefs

    np.random.seed(11)                     # identical matrix to pilot4c
    L, dd = 0.3, 64
    U, _ = np.linalg.qr(np.random.randn(dd, dd))
    V, _ = np.linalg.qr(np.random.randn(dd, dd))
    M = (U * np.logspace(0, np.log10(L), dd)) @ V.T
    ref = U @ V.T

    E = {}
    for K in (1, 2, 3, 4):
        E[K] = float(np.linalg.norm(run(coefs, M, None, None, K) - ref, 2))
    R["E"] = E
    print("=" * 100)
    print("[A] exact-arithmetic error of the iterated degree-17 step")
    print("   " + "  ".join(f"K={k}: {v:.3e}" for k, v in E.items()))
    print(f"   pilot4c E_1 = {c4['E1']:.3e} (reproduction check: "
          f"{'matches' if abs(E[1] - c4['E1']) / c4['E1'] < 1e-6 else 'DIFFERS'})")
    R["E1_matches_pilot4c"] = abs(E[1] - c4["E1"]) / c4["E1"] < 1e-6

    R["table"] = {}
    R["summary"] = {}
    for K in (1, 2, 3, 4):
        tab = {}
        print(f"\n  K={K}   rows pX, cols pA   (errors)")
        print(f"  {'pX\\pA':>6s}" + "".join(f"{p:<10d}" for p in PS))
        for pX in PS:
            row = [float(np.linalg.norm(run(coefs, M, pX, pA, K) - ref, 2)) for pA in PS]
            tab[str(pX)] = row
            print(f"  {pX:6d}" + "".join(f"{v:<10.2e}" for v in row))
        R["table"][str(K)] = tab
        eq = {}
        for tot in (16, 22, 32):
            cand = {f"{pX},{pA}": tab[str(pX)][PS.index(pA)]
                    for pX in PS for pA in PS if pX + pA == tot}
            if not cand:
                continue
            eq[str(tot)] = cand
            base = cand.get("8,8") or min(cand.values())
            r610 = cand.get("6,10")
            r106 = cand.get("10,6")
            s = {"balanced": base, "b_richer": r610, "x_richer": r106,
                 "b_over": (r610 / base) if (r610 and base) else None,
                 "x_over": (r106 / base) if (r106 and base) else None}
            if s["b_over"] and s["x_over"]:
                s["asymmetry"] = max(s["b_over"], s["x_over"]) / min(s["b_over"], s["x_over"])
            R["summary"][f"K{K}_tot{tot}"] = s
            print(f"    total {tot}: " + "  ".join(f"({k}):{v:.2e}" for k, v in sorted(
                cand.items(), key=lambda z: z[1])))
            if s["b_over"] and s["x_over"]:
                print(f"      vs balanced: (6,10) {s['b_over']:.2f}x, (10,6) "
                      f"{s['x_over']:.2f}x -> asymmetry {s['asymmetry']:.2f}x")

    with open(OUT, "w") as f:
        json.dump(R, f, indent=1, default=float)
    print("\nwrote", os.path.basename(OUT))


if __name__ == "__main__":
    main()
