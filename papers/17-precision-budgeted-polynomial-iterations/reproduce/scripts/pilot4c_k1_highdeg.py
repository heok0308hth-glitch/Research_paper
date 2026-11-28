#!/usr/bin/env python3
"""
Pilot 4c - K=1 separation with a single step accurate enough to expose rounding.

At L=0.3 a quintic step has E_1 = 0.108, which hides the rounding contribution
(see pilot4b).  Here the single step is the degree-17 odd polynomial obtained
from the pilot5 linear program at eps = 1.03 * E_17(L), so E_1 ~ 1.7e-3 and the
rounding terms are visible.

The map is the general odd polynomial of a matrix,
    P(X) = sum_j a_{2j+1} (X X^T)^j X ,
with the two stored intermediates being X (p_X bits) and A = X X^T (p_A bits).
"""
import json
import sys
import numpy as np

sys.path.insert(0, "work/iclr-research/data")
import pilot5_lower_bound as p5

np.random.seed(11)
R = {}


def q(x, p):
    if p is None or p >= 30:
        return x
    x = np.asarray(x, float)
    m, e = np.frexp(x)
    s = float(2 ** (p - 1))
    return np.ldexp(np.round(m * s) / s, e)


def run(coeffs, ks, M, pX, pA):
    X = q(M / np.linalg.norm(M, 2), pX)
    A = q(X @ X.T, pA)
    Y = coeffs[0] * X
    P = np.eye(X.shape[0])
    for a in coeffs[1:]:
        P = P @ A
        Y = Y + a * (P @ X)
    return q(Y, pX)


def main():
    L = 0.3
    d = 17
    eps_req = 1.03 * p5.min_eps(L, d)
    a = p5.solve(L, d, eps_req, "deriv", p5.build(L, d, "deriv"))
    ks = list(range(1, d + 1, 2))
    ec, S, pm, l1 = p5.verify(L, d, a)
    print(f"design: L={L}, d={d}, eps_req={eps_req:.3e} -> verified eps={ec:.3e}, "
          f"max|P'|={S:.3e}, max|P|={pm:.3f}, l1={l1:.3e}")
    print("coeffs:", " ".join(f"{ks[i]}:{a[i]:+.3e}" for i in range(len(ks))))

    dd = 64
    U, _ = np.linalg.qr(np.random.randn(dd, dd))
    V, _ = np.linalg.qr(np.random.randn(dd, dd))
    M = (U * np.logspace(0, np.log10(L), dd)) @ V.T
    ref = U @ V.T
    E1 = float(np.linalg.norm(run(a, ks, M, None, None) - ref, 2))
    print(f"exact-arithmetic error of the single step: {E1:.3e}")

    ps = [6, 8, 10, 12, 16, 24]
    print(f"\n{'pX\\pA':>6s}" + "".join(f"{p:<11d}" for p in ps))
    tab = {}
    for pX in ps:
        row = [float(np.linalg.norm(run(a, ks, M, pX, pA) - ref, 2)) for pA in ps]
        tab[str(pX)] = row
        print(f"{pX:6d}" + "".join(f"{v:<11.2e}" for v in row))
    row8 = tab["8"]
    col8 = [tab[str(p)][ps.index(8)] for p in ps]
    print(f"  row pX=8 : pA 8->16 gives {row8[1] / row8[4]:.2f}x")
    print(f"  col pA=8 : pX 8->16 gives {col8[1] / col8[4]:.2f}x")

    # min over allocations of equal total bits
    print("\n  equal-cost:")
    for tot in (16, 22, 32):
        best = sorted(
            (tab[str(pX)][ps.index(pA)], pX, pA)
            for pX in ps for pA in ps if pX + pA == tot)
        if best:
            print(f"   total {tot}: " + "  ".join(f"({x},{y}):{e:.2e}" for e, x, y in best))
            R.setdefault("equal_cost", {})[str(tot)] = [
                {"pX": x, "pA": y, "err": float(e)} for e, x, y in best]

    R["design"] = {"L": L, "d": d, "eps_req": eps_req, "eps_verified": ec,
                   "S_max": S, "coeffs": [float(v) for v in a], "ks": ks}
    R["E1"] = E1
    R["table"] = tab
    R["row8_8to16"] = row8[1] / row8[4]
    R["col8_8to16"] = col8[1] / col8[4]
    with open("work/iclr-research/data/pilot4c_results.json", "w") as f:
        json.dump(R, f, indent=1)


if __name__ == "__main__":
    main()
