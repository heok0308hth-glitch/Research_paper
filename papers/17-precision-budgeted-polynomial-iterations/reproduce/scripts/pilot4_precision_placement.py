#!/usr/bin/env python3
"""
Pilot 4 - Where inside a polynomial sign iteration should the precision go?

The online loop stores two intermediates: X (the iterate) and
B = b*X X^T + c*(X X^T)^2 (the matrix multiplied by X).
This measures the error for every (p_X, p_B) pair at K=6, i.e. a small
bit-allocation table for the 2-operand case.
"""
import json
import numpy as np

np.random.seed(2)
L = 1e-3
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


def run(coeffs, M, pX, pB):
    X = q(M / np.linalg.norm(M, 2), pX)
    for a, b, c in coeffs:
        A = X @ X.T
        B = q(b * A + c * (A @ A), pB)
        X = q(a * X + B @ X, pX)
    return X


def main():
    d = 64
    U, _ = np.linalg.qr(np.random.randn(d, d))
    V, _ = np.linalg.qr(np.random.randn(d, d))
    M = (U * np.logspace(0, -3, d)) @ V.T
    ref = U @ V.T
    K = 6
    co = pe_coeffs(L, K)

    ps = [6, 8, 10, 11, 13, 16, 24]
    print("=" * 84)
    print(f"[A] ||X_K - UV^T||_2, K={K}, rows = p_X, cols = p_B")
    print("=" * 84)
    print(f"{'pX\\pB':>6s}" + "".join(f"{p:<10d}" for p in ps))
    tab = {}
    for pX in ps:
        row = []
        for pB in ps:
            e = float(np.linalg.norm(run(co, M, pX, pB) - ref, 2))
            row.append(e)
        tab[str(pX)] = row
        print(f"{pX:6d}" + "".join(f"{v:<10.2e}" for v in row))
    R["table"] = tab

    print("\n" + "=" * 84)
    print("[B] marginal value of one extra bit, holding the other operand fixed")
    print("=" * 84)
    for pX in (8, 11):
        row = tab[str(pX)]
        print(f"pX={pX}: " + "".join(
            f"{ps[i]}->{ps[i+1]}:{row[i]/row[i+1]:5.1f}x " for i in range(len(ps) - 1)))
    for pB in (8, 11):
        col = [tab[str(p)][ps.index(pB)] for p in ps]
        print(f"pB={pB}: " + "".join(
            f"{ps[i]}->{ps[i+1]}:{col[i]/col[i+1]:5.1f}x " for i in range(len(ps) - 1)))

    print("\n" + "=" * 84)
    print("[C] equal-cost comparison (total stored bits per matrix = p_X + p_B)")
    print("=" * 84)
    for tot in (16, 22, 32):
        best = []
        for pX in ps:
            for pB in ps:
                if pX + pB == tot:
                    best.append((float(np.linalg.norm(run(co, M, pX, pB) - ref, 2)),
                                 pX, pB))
        best.sort()
        if best:
            print(f"total {tot} bits: " + "  ".join(
                f"({pX},{pB}):{e:.2e}" for e, pX, pB in best))
            R.setdefault("equal_cost", {})[str(tot)] = [
                {"pX": pX, "pB": pB, "err": e} for e, pX, pB in best]

    with open("work/iclr-research/data/pilot4_results.json", "w") as f:
        json.dump(R, f, indent=1)


if __name__ == "__main__":
    main()
