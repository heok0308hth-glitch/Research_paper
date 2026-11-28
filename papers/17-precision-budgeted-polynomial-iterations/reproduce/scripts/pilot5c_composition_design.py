#!/usr/bin/env python3
"""
Pilot 5c - Accuracy vs sensitivity at the composition level, and whether the
shipped design is precision-optimal.

Three questions:
 (1) Along a one-parameter family of designs (LSQ5 -> PE_ref), is the
     (eps_K, S_K) frontier monotone?  If not, "accuracy costs sensitivity" fails.
 (2) Over the (cushion, safety) design grid, is the two-term law
     err ~ eps_K + 2^-p * S_K predictive of the real bf16 kernel error?
 (3) Does minimising eps_K + 2^-p * S_K over the grid beat the shipped
     (cushion 0.02, safety 1.01) design in the real kernel?

Model: d=64, spectrum logspace(0,-3), X and B stored in p bits, products in
float64 (identical to pilot1c / pilot4).
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


def lsq_coeffs(l0, K, n=1200):
    u, l, out = 1.0, l0, []
    for _ in range(K):
        x = np.linspace(l, u, n)
        A = np.stack([x, x**3, x**5], 1)
        a, b, c = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
        out.append((a, b, c))
        l = a * l + b * l**3 + c * l**5
        u = 2 - l
    return out


G1 = np.logspace(np.log10(L), 0, 1500)
G2 = np.logspace(np.log10(L), 0, 4000)


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


def metrics(coeffs, K):
    tr = orbit(coeffs, G2)
    E = float(np.max(np.abs(tr[K] - 1)))
    S = float(np.max(sensitivity(coeffs, G1)))
    return E, S


def kernel(coeffs, M, ref, p):
    X = q(M / np.linalg.norm(M, 2), p)
    for a, b, c in coeffs:
        A = X @ X.T
        B = q(b * A + c * (A @ A), p)
        X = q(a * X + B @ X, p)
    return float(np.linalg.norm(X - ref, 2))


def main():
    d = 64
    U, _ = np.linalg.qr(np.random.randn(d, d))
    V, _ = np.linalg.qr(np.random.randn(d, d))
    M = (U * np.logspace(0, -3, d)) @ V.T
    ref = U @ V.T

    # ---------- (1) one-parameter family
    print("=" * 96)
    print("[1] monotonicity along LSQ5 -> PE_ref interpolation, K=6")
    print("=" * 96)
    K = 6
    lq, pe = lsq_coeffs(L, K), pe_coeffs(L, K)
    fam = []
    for th in np.linspace(0, 1, 11):
        co = [tuple((1 - th) * np.array(a) + th * np.array(b)) for a, b in zip(lq, pe)]
        E, S = metrics(co, K)
        fam.append({"theta": float(th), "eps": E, "S": S})
        print(f"  theta={th:4.2f}  eps={E:.3e}  S={S:.3e}  eps+2^-8*S={E + 2**-8 * S:.3e}")
    R["family_interp_K6"] = fam
    # local slope of log S vs log eps
    sl = []
    for i in range(len(fam) - 1):
        de = np.log(fam[i + 1]["eps"]) - np.log(fam[i]["eps"])
        ds = np.log(fam[i + 1]["S"]) - np.log(fam[i]["S"])
        sl.append(ds / de)
    print("  local dlogS/dlogeps: " + " ".join(f"{v:6.2f}" for v in sl))
    R["family_interp_local_slope"] = [float(v) for v in sl]

    # ---------- (2)(3) design grid
    print("\n" + "=" * 96)
    print("[2] design grid: two-term prediction vs measured bf16/fp16 kernel error")
    print("=" * 96)
    cus = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    saf = [1.0, 1.005, 1.01, 1.02, 1.05, 1.1, 1.2]
    for K in (5, 6, 7):
        rows = []
        for c in cus:
            for s in saf:
                co = pe_coeffs(L, K, cushion=c, safety=s)
                E, S = metrics(co, K)
                if not np.isfinite(E) or not np.isfinite(S):
                    continue
                e8 = kernel(co, M, ref, 8)
                e11 = kernel(co, M, ref, 11)
                rows.append({"cushion": c, "safety": s, "eps": E, "S": S,
                             "pred8": E + 2.0 ** -8 * S, "err8": e8,
                             "pred11": E + 2.0 ** -11 * S, "err11": e11})
        rows.sort(key=lambda r: r["pred8"])
        best_pred = rows[0]
        best_meas = min(rows, key=lambda r: r["err8"])
        ship = [r for r in rows if r["cushion"] == 0.02 and r["safety"] == 1.01][0]
        print(f"\n K={K}   ({len(rows)} designs)")
        print(f"   shipped (c=0.02,s=1.01)   eps={ship['eps']:.2e} S={ship['S']:.2e}"
              f"  pred8={ship['pred8']:.2e}  err8={ship['err8']:.2e}  err11={ship['err11']:.2e}")
        print(f"   best by prediction        eps={best_pred['eps']:.2e} S={best_pred['S']:.2e}"
              f"  pred8={best_pred['pred8']:.2e}  err8={best_pred['err8']:.2e}"
              f"  (c={best_pred['cushion']}, s={best_pred['safety']})")
        print(f"   best measured (err8)      eps={best_meas['eps']:.2e} S={best_meas['S']:.2e}"
              f"  pred8={best_meas['pred8']:.2e}  err8={best_meas['err8']:.2e}"
              f"  (c={best_meas['cushion']}, s={best_meas['safety']})")
        print(f"   gain of prediction-best over shipped (measured): "
              f"{ship['err8'] / best_pred['err8']:.2f}x")
        # rank correlation between prediction and measurement
        import scipy.stats as st
        rho = st.spearmanr([r["pred8"] for r in rows], [r["err8"] for r in rows]).statistic
        rho11 = st.spearmanr([r["pred11"] for r in rows], [r["err11"] for r in rows]).statistic
        print(f"   Spearman(pred, measured): bf16 {rho:.3f}  fp16 {rho11:.3f}")
        R[f"grid_K{K}"] = {"rows": rows, "spearman8": float(rho), "spearman11": float(rho11),
                           "shipped": ship, "best_pred": best_pred, "best_meas": best_meas}
        # show a few top designs by measurement
        for r in sorted(rows, key=lambda r: r["err8"])[:6]:
            print(f"     c={r['cushion']:<6g} s={r['safety']:<6g} eps={r['eps']:.2e}"
                  f" S={r['S']:.2e} pred8={r['pred8']:.2e} err8={r['err8']:.2e}")

    with open("work/iclr-research/data/pilot5c_results.json", "w") as f:
        json.dump(R, f, indent=1)


if __name__ == "__main__":
    main()
