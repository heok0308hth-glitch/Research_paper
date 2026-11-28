#!/usr/bin/env python3
"""
Pilot 5b - How the certificate S(K) and the floor constant scale with L and d.

The two-term law is  err ~ E_K + C * 2^-p * S_K.  This measures, for the shipped
design family (cushion 0.02, safety 1.01, quintic steps):

  S(K)  as a function of L in {1e-2, 1e-3, 1e-4}      -> the exponent of L
  C     as a function of d in {32, 64, 128}            -> matrix-size effect

Kernel model identical to pilot1c: X and B stored in p bits, products in float64.
C is only meaningful where the design has not yet converged (E_K << 2^-p S_K is
the regime where the law is known to fail), so the ratio is reported per K.
"""
import json
import numpy as np

np.random.seed(7)
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


def orbit(coeffs, x0):
    x = np.asarray(x0, float)
    tr = [x.copy()]
    for a, b, c in coeffs:
        x = a * x + b * x**3 + c * x**5
        tr.append(x.copy())
    return np.array(tr)


def sensitivity(coeffs, x0):
    tr = orbit(coeffs, x0)
    base = np.asarray(x0, float)
    best = np.ones_like(base)
    for j in range(len(coeffs)):
        amp = np.ones_like(base)
        for l in range(j, len(coeffs)):
            a, b, c = coeffs[l]
            amp = amp * np.abs(a + 3 * b * tr[l] ** 2 + 5 * c * tr[l] ** 4)
        best = np.maximum(best, amp)
    return best


def kernel(coeffs, M, ref, p):
    X = q(M / np.linalg.norm(M, 2), p)
    for a, b, c in coeffs:
        A = X @ X.T
        B = q(b * A + c * (A @ A), p)
        X = q(a * X + B @ X, p)
    return float(np.linalg.norm(X - ref, 2))


def main():
    Ls = [1e-2, 1e-3, 1e-4]
    ds = [32, 64, 128]
    Ks = list(range(2, 9))
    ps = [10, 12]

    print("=" * 100)
    print("[A] S(K) of the shipped design along the orbit, per L  (independent of d)")
    print("=" * 100)
    Sok = {}
    Eok = {}
    print(f"{'L':>8s}" + "".join(f"K={k:<11d}" for k in Ks))
    for L in Ls:
        g = np.logspace(np.log10(L), 0, 2000)
        Srow, Erow = [], []
        for K in Ks:
            co = pe_coeffs(L, K)
            Srow.append(float(np.max(sensitivity(co, g))))
            Erow.append(float(np.max(np.abs(orbit(co, g)[K] - 1))))
        Sok[L], Eok[L] = Srow, Erow
        print(f"{L:8.0e}" + "".join(f"{v:<11.2e}" for v in Srow))
    print("  exact-arithmetic E_K:")
    for L in Ls:
        print(f"{L:8.0e}" + "".join(f"{v:<11.2e}" for v in Eok[L]))
    R["S_of_L"] = {f"{L}": dict(zip(map(str, Ks), Sok[L])) for L in Ls}
    R["E_of_L"] = {f"{L}": dict(zip(map(str, Ks), Eok[L])) for L in Ls}

    print("\n  log-log slope of S(K) against L (per K):")
    for i, K in enumerate(Ks):
        x = np.log(np.array(Ls))
        y = np.log([Sok[L][i] for L in Ls])
        A = np.vstack([x, np.ones_like(x)]).T
        sl = np.linalg.lstsq(A, y, rcond=None)[0][0]
        print(f"   K={K}: dlogS/dlogL = {sl:5.2f}   "
              + "  ".join(f"{Sok[L][i]:.2e}" for L in Ls))
        R.setdefault("slope_S_L", {})[str(K)] = float(sl)

    print("\n" + "=" * 100)
    print("[B] floor constant C = err / (2^-p * S(K))  over d and L")
    print("=" * 100)
    for d in ds:
        U, _ = np.linalg.qr(np.random.randn(d, d))
        V, _ = np.linalg.qr(np.random.randn(d, d))
        for L in Ls:
            M = (U * np.logspace(0, np.log10(L), d)) @ V.T
            ref = U @ V.T
            for p in ps:
                row, Crow = [], []
                for i, K in enumerate(Ks):
                    co = pe_coeffs(L, K)
                    e = kernel(co, M, ref, p)
                    pred = 2.0 ** -p * Sok[L][i]
                    row.append(e)
                    Crow.append(e / pred)
                print(f"d={d:<4d} L={L:<7.0e} p={p:<3d} err: "
                      + " ".join(f"{v:.2e}" for v in row))
                print(f"{'':19s}  C  : " + " ".join(f"{v:8.2f}" for v in Crow))
                R.setdefault("floor", {})[f"d{d}|L{L}|p{p}"] = {
                    "err": row, "C": Crow, "K": Ks}

    with open("work/iclr-research/data/pilot5b_results.json", "w") as f:
        json.dump(R, f, indent=1)


if __name__ == "__main__":
    main()
