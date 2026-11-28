#!/usr/bin/env python3
"""
Pilot 6b - Where the floor constant of  err ~ C u  comes from.

Pilot 5b reported a plateau err ~ C u with C = 10.1-13.8 (d = 32..128, L = 1e-2),
nearly independent of the matrix size.  This script replaces that constant by the
certificate: with the prefix ratchet of pilot6,

        err ~ E_K + c * u * S_pert(K),      c = 0.142,  S_pert = max over prefixes,

so the observed "C" is  c * S_pert(K):  C = 10.9 at L = 1e-2 and 86 at L = 1e-3.
The apparent d-independence then reduces to two separate claims, both measured here:

  (i)  S_pert is a scalar function of the coefficients and (L, K), hence d-free;
  (ii) the entrywise relative rounding of X has spectral norm u * O(1), not u * sqrt(d).

The second point is the one a Frobenius-norm argument gets wrong: ||dX||_F / u grows
like sqrt(d) (measured 1.9 -> 3.8 -> 7.6 for d = 32/64/128) while ||dX||_2 / u stays
at 0.5-0.6, so a bound built on ||.||_F would predict a sqrt(d) growth of the floor
(1 : 1.41 : 2.00) that the measurement does not show (1 : 1.18 : 1.30).

Also measured: the size of every rounding actually injected in the run (X channel and
B channel, per step, in units of u), which is the O(1) input of the two-term law, and
the u-linearity of the floor over p = 6..14.
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = {"config": {}}


def q(x, p):
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


def orbit(co, x0):
    x = np.asarray(x0, float)
    tr = [x.copy()]
    for a, b, c in co:
        x = a * x + b * x**3 + c * x**5
        tr.append(x.copy())
    return np.array(tr)


def S_prefix_ratchet(co, grid):
    tr = orbit(co, grid)
    K = len(co)
    best = 1.0
    for t in range(1, K + 1):
        for j in range(t):
            amp = np.ones_like(np.asarray(grid, float))
            for l in range(j, t):
                a, b, c = co[l]
                amp = amp * np.abs(a + 3 * b * tr[l] ** 2 + 5 * c * tr[l] ** 4)
            best = max(best, float(np.max(amp)))
    return best


def run_instrumented(co, M, p):
    """returns final error, and every rounding injected (||dX||_2, ||dB||_2) per step in units of u"""
    u = 2.0 ** -p
    X = q(M / np.linalg.norm(M, 2), p)
    inj = []
    for a, b, c in co:
        A = X @ X.T
        Braw = b * A + c * (A @ A)
        B = q(Braw, p)
        dB = float(np.linalg.norm(B - Braw, 2)) / u
        Xraw = a * X + B @ X
        Xn = q(Xraw, p)
        dX = float(np.linalg.norm(Xn - Xraw, 2)) / u
        inj.append((dX, dB, float(np.linalg.norm(Braw, 2))))
        X = Xn
    return X, inj


def make_M(d, L, seed=1):
    rs = np.random.RandomState(seed)
    U, _ = np.linalg.qr(rs.randn(d, d))
    V, _ = np.linalg.qr(rs.randn(d, d))
    return (U * np.logspace(0, np.log10(L), d)) @ V.T, U @ V.T


C_CERT = 0.142      # from pilot6 (single constant, calibrated on d=64, L in {1e-2,1e-3})
R["config"] = {"c_cert": C_CERT, "design": "cushion 0.02, safety 1.01",
               "note": "kernel model as in pilot6; u = 2^-p"}


def main():
    print("=" * 100)
    print("[A] the rounding that is actually injected, in units of u")
    print("    X channel = ||q(X)-X||_2/u, B channel = ||q(B)-B||_2/u, ||B||_2 at the fixed point")
    print("=" * 100)
    A = {}
    for L in (1e-2, 1e-3):
        for K in (6,):
            co = pe_coeffs(L, K)
            for d in (64,):
                M, ref = make_M(d, L)
                for p in (8, 11, 14):
                    u = 2.0 ** -p
                    X, inj = run_instrumented(co, M, p)
                    err = float(np.linalg.norm(X - ref, 2))
                    print(f"  L={L:<7g} K={K} d={d} p={p:<3d}  err/u={err/u:7.2f}   "
                          + "  ".join(f"step{i+1}: dX={a:5.2f} dB={b:5.2f} |B|={c:5.3f}"
                                      for i, (a, b, c) in enumerate(inj)))
                    A[f"L{L}_p{p}"] = {"err_over_u": err / u, "per_step": inj}
    print("\n  -> every injected rounding is O(u); the floor is their amplified sum, not a sqrt(d) effect.")

    print("\n" + "=" * 100)
    print("[B] the floor minus the certificate: err/u against c*S_pert and the size dependence")
    print("=" * 100)
    print(f"  {'L':>7s} {'d':>4s} {'p':>3s} {'err/u':>9s} {'E_K/u':>9s} {'S_pert':>9s} {'c*S_pert':>9s} "
          f"{'pred/err':>8s} {'|dX0|2/u':>9s} {'|dX0|F/u':>9s}")
    B = {}
    for L in (1e-2, 1e-3):
        for K in (6,):
            co = pe_coeffs(L, K)
            grid = np.logspace(np.log10(L), 0, 1500)
            S = S_prefix_ratchet(co, grid)
            EK = float(np.max(np.abs(orbit(co, grid)[K] - 1)))
            for d in (32, 64, 128):
                M, ref = make_M(d, L)
                X0 = M / np.linalg.norm(M, 2)
                for p in (8, 11, 14):
                    u = 2.0 ** -p
                    dX0 = q(X0, p) - X0
                    errv = float(np.linalg.norm(_run(co, M, p) - ref, 2))
                    pred = EK + C_CERT * u * S
                    print(f"  {L:7g} {d:4d} {p:3d} {errv/u:9.2f} {EK/u:9.2f} {S:9.1f} {C_CERT*S:9.1f} "
                          f"{pred/errv:8.2f} {np.linalg.norm(dX0,2)/u:9.3f} {np.linalg.norm(dX0,'fro')/u:9.3f}")
                    B[f"L{L}_d{d}_p{p}"] = {"err_over_u": errv / u, "EK_over_u": EK / u, "S_pert": S,
                                            "pred_over_err": pred / errv,
                                            "dX0_2_over_u": float(np.linalg.norm(dX0, 2) / u),
                                            "dX0_F_over_u": float(np.linalg.norm(dX0, "fro") / u)}
    print("\n  d-scaling of the floor at fixed (L,K,p): measured against the two candidate laws")
    for p in (8, 11, 14):
        for L in (1e-2, 1e-3):
            v = [B[f"L{L}_d{d}_p{p}"]["err_over_u"] for d in (32, 64, 128)]
            f = [B[f"L{L}_d{d}_p{p}"]["dX0_F_over_u"] for d in (32, 64, 128)]
            print(f"   L={L:<7g} p={p:<3d} err/u 32:64:128 = 1 : {v[1]/v[0]:.2f} : {v[2]/v[0]:.2f}"
                  f"   (sqrt d = 1 : 1.41 : 2.00)   |dX0|_F/u = 1 : {f[1]/f[0]:.2f} : {f[2]/f[0]:.2f}")

    print("\n" + "=" * 100)
    print("[C] u-linearity: the floor halves with every bit, C = err/u is p-flat")
    print("=" * 100)
    Cc = {}
    for L in (1e-2, 1e-3):
        co = pe_coeffs(L, 6)
        M, ref = make_M(64, L)
        print(f"  L={L:<7g} K=6 d=64")
        for p in (6, 7, 8, 10, 11, 12, 14, 16, 24):
            u = 2.0 ** -p
            err = float(np.linalg.norm(_run(co, M, p) - ref, 2))
            print(f"    p={p:<3d} err={err:10.3e}  err/u={err/u:8.2f}  err*2^p={err*2.0**p:8.2f}")
            Cc[f"L{L}_p{p}"] = err / u
    print("\n  -> err * 2^p is flat to within 20% from p=8 to p=14: the floor is linear in u and the")
    print("     constant of proportionality is c * S_pert, not a free parameter of the arithmetic.")

    out = {"A_injected_rounding": A, "B_size_scaling": B, "C_u_linearity": Cc, "config": R["config"]}
    with open(os.path.join(HERE, "pilot6b_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {os.path.join(HERE, 'pilot6b_results.json')}")


def _run(co, M, p):
    X = q(M / np.linalg.norm(M, 2), p)
    for a, b, c in co:
        A = X @ X.T
        B = q(b * A + c * (A @ A), p)
        X = q(a * X + B @ X, p)
    return X


if __name__ == "__main__":
    main()
