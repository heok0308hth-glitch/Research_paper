#!/usr/bin/env python3
"""
Pilot 1c - How the achievable accuracy of a fixed polynomial design scales with
stored precision, and where the optimal iteration count sits.

d=64 random matrix with logspace(0,-3) spectrum. Kernel model: X and B are
stored in p significant bits, products accumulated in float64, A not stored.
"""
import json
import numpy as np

np.random.seed(1)
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


NS5 = [(15 / 8, -10 / 8, 3 / 8)] * 10


def mat_run(coeffs, M, p):
    X = q(M / np.linalg.norm(M, 2), p)   # spectral normalisation: sigma_max = 1
    for a, b, c in coeffs:
        A = X @ X.T
        B = q(b * A + c * (A @ A), p)
        X = q(a * X + B @ X, p)
    return X


def main():
    d = 64
    U, _ = np.linalg.qr(np.random.randn(d, d))
    V, _ = np.linalg.qr(np.random.randn(d, d))
    sig = np.logspace(0, -3, d)
    M = (U * sig) @ V.T
    ref = U @ V.T

    print("=" * 88)
    print("[A] ||X_K - UV^T||_2 vs stored precision p, K fixed")
    print("    prediction of the two-term model: error ~ u*S(K), u = 2^-p")
    print("=" * 88)
    ps = [6, 7, 8, 9, 10, 11, 13, 16, 20]
    for nm, gen in (("NS5", lambda k: NS5[:k]), ("PE_ref", lambda k: pe_coeffs(L, k))):
        for K in (4, 5, 6, 8):
            row = [float(np.linalg.norm(mat_run(gen(K), M, p) - ref, 2)) for p in ps]
            R.setdefault(nm, {})[K] = dict(zip(map(str, ps), row))
            print(f"{nm:7s} K={K}  " + "".join(f"{p}:{v:7.1e} " for p, v in zip(ps, row)))
        print()

    print("=" * 88)
    print("[B] optimal iteration count K* vs stored precision (PE_ref design)")
    print("=" * 88)
    print(f"{'p':>4s}" + "".join(f"K={k:<2d}     " for k in range(2, 10)))
    kstar = {}
    for p in [8, 10, 12, 14, 16, 24]:
        errs = [float(np.linalg.norm(mat_run(pe_coeffs(L, k), M, p) - ref, 2))
                for k in range(2, 10)]
        kstar[p] = {"errs": errs, "kstar": int(np.argmin(errs)) + 2}
        print(f"{p:4d}" + "".join(f"{v:8.1e} " for v in errs)
              + f"  K*={int(np.argmin(errs)) + 2}")
    R["kstar"] = {str(k): v for k, v in kstar.items()}

    with open("work/iclr-research/data/pilot1c_results.json", "w") as f:
        json.dump(R, f, indent=1)


if __name__ == "__main__":
    main()
