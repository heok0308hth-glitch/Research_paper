#!/usr/bin/env python3
"""
Pilot 4b - Bottleneck vs accumulation: the same bit-allocation table at K=1.

pilot4 measured a (p_X, p_B) table at K=6: with p_X = 8 the error is flat in
p_B (1.07x over 8 extra bits) while with p_X = 16 it improves 71x.  Two readings:

  bottleneck     error ~ max_i 2^-p_i * S_i        (two-operand max structure)
  accumulation   X is rounded once per step, so at K=6 its rounding enters six
                 times and simply matters more than the single rounding of B

At K=1 both operands are rounded exactly once, so accumulation cannot produce
an asymmetry.  If the table is still asymmetric at K=1, and matches
eps_1 + max(2^-p_X S_X, 2^-p_B S_B) with S_X, S_B measured by single-operand
injection, the max reading is supported.

Model: d=64, spectrum in [L,1], X and B stored in p bits, products in float64.
"""
import json
import numpy as np

np.random.seed(4)
R = {}


def q(x, p):
    if p is None or p >= 30:
        return x
    x = np.asarray(x, float)
    m, e = np.frexp(x)
    s = float(2 ** (p - 1))
    return np.ldexp(np.round(m * s) / s, e)


def optimal_quintic(l, u):
    if 1 - 5e-6 <= l / u:
        return (15 / 8) / u, (-10 / 8) / (u ** 3), (3 / 8) / (u ** 5)
    qq, r = (3 * l + u) / 4, (l + 3 * u) / 4
    E, old = np.inf, None
    while not old or abs(old - E) > 1e-15:
        old = E
        A = np.array([[l, l**3, l**5, 1], [qq, qq**3, qq**5, -1],
                      [r, r**3, r**5, 1], [u, u**3, u**5, -1]])
        a, b, c, E = np.linalg.solve(A, np.ones(4))
        qq, r = np.sqrt((-3 * b + np.array([-1, 1]) * np.sqrt(9 * b**2 - 20 * a * c)) / (10 * c))
    return float(a), float(b), float(c)


def pe_coeffs(l0, K, cushion=0.02, safety=1.01):
    u, l, out = 1.0, l0, []
    for it in range(K):
        lo = max(l, cushion * u)
        a, b, c = optimal_quintic(lo, u)
        if cushion * u > l:
            pl = a * l + b * l**3 + c * l**5
            pu = a * u + b * u**3 + c * u**5
            r = 2 / (pl + pu)
            a, b, c = a * r, b * r, c * r
        if it < K - 1:
            a, b, c = a / safety, b / safety**3, c / safety**5
        out.append((a, b, c))
        l = a * l + b * l**3 + c * l**5
        u = 2 - l
    return out


def run(coeffs, M, pX, pB, inject=None, u_inj=0.0):
    """inject = (operand, step_index) -> add a relative perturbation there."""
    X = q(M / np.linalg.norm(M, 2), pX)
    for k, (a, b, c) in enumerate(coeffs):
        if inject == ("X", k):
            X = X * (1.0 + u_inj)
        A = X @ X.T
        B = q(b * A + c * (A @ A), pB)
        if inject == ("B", k):
            B = B * (1.0 + u_inj)
        X = q(a * X + B @ X, pX)
    return X


def table(coeffs, M, ref, ps):
    tab = {}
    for pX in ps:
        tab[str(pX)] = [float(np.linalg.norm(run(coeffs, M, pX, pB) - ref, 2)) for pB in ps]
    return tab


def main():
    ps = [6, 8, 10, 12, 16, 24]
    d = 64
    U, _ = np.linalg.qr(np.random.randn(d, d))
    V, _ = np.linalg.qr(np.random.randn(d, d))
    for L in (0.3,):
        M = (U * np.logspace(0, np.log10(L), d)) @ V.T
        ref = U @ V.T
        print("#" * 96)
        print(f"L={L}  d={d}  spectrum logspace(0,{np.log10(L):.2f})")
        print("#" * 96)
        for K in (1, 2, 3, 6):
            co = pe_coeffs(L, K)
            E = float(np.linalg.norm(run(co, M, None, None) - ref, 2))   # exact-arithmetic error
            tab = table(co, M, ref, ps)
            print(f"\n== K={K}  exact-arithmetic error E_K = {E:.3e}")
            print(f"{'pX\\pB':>6s}" + "".join(f"{p:<11d}" for p in ps))
            for pX in ps:
                print(f"{pX:6d}" + "".join(f"{v:<11.2e}" for v in tab[str(pX)]))
            # flatness of a row / column
            row8 = tab["8"]
            col8 = [tab[str(p)][ps.index(8)] for p in ps]
            print(f"  row pX=8 : pB 8->16 gives {row8[1] / row8[4]:.2f}x")
            print(f"  col pB=8 : pX 8->16 gives {col8[1] / col8[4]:.2f}x")

            # single-operand injection sensitivities
            u = 2.0 ** -12
            SX, SB = [], []
            for k in range(K):
                ex = np.linalg.norm(run(co, M, None, None, ("X", k), u) - run(co, M, None, None))
                eb = np.linalg.norm(run(co, M, None, None, ("B", k), u) - run(co, M, None, None))
                SX.append(float(ex / u))
                SB.append(float(eb / u))
            SXmax, SBmax = max(SX), max(SB)
            print(f"  injected sensitivity: S_X = {SXmax:.3e}  S_B = {SBmax:.3e}"
                  f"   (ratio {SXmax / max(SBmax, 1e-30):.2f})")
            # predicted two-operand max model
            print("  max-form prediction (E_K + max(2^-pX S_X, 2^-pB S_B)):")
            print(f"{'pX\\pB':>6s}" + "".join(f"{p:<11d}" for p in ps))
            pred = {}
            for pX in ps:
                prow = []
                for pB in ps:
                    prow.append(E + max(2.0 ** -pX * SXmax, 2.0 ** -pB * SBmax))
                pred[str(pX)] = prow
                print(f"{pX:6d}" + "".join(f"{v:<11.2e}" for v in prow))
            ratio = [[tab[k][i] / pred[k][i] for i in range(len(ps))] for k in pred]
            print(f"  measured/predicted: min {np.min(ratio):.2f} median {np.median(ratio):.2f}"
                  f" max {np.max(ratio):.2f}")
            R.setdefault(f"L{L}", {})[f"K{K}"] = {
                "E": E, "table": tab, "pred": pred,
                "SX": SX, "SB": SB, "SXmax": SXmax, "SBmax": SBmax,
                "row8_8to16": row8[1] / row8[4], "col8_8to16": col8[1] / col8[4],
                "meas_over_pred": [float(np.min(ratio)), float(np.median(ratio)), float(np.max(ratio))]}

    with open("work/iclr-research/data/pilot4b_results.json", "w") as f:
        json.dump(R, f, indent=1)


if __name__ == "__main__":
    main()
