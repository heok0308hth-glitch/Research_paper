#!/usr/bin/env python3
"""
Pilot 1b - Precision/optimality Pareto front for quintic matrix-sign iterations.

Compares
  NS5            fixed Newton-Schulz quintic (15/8, -10/8, 3/8)
  LSQ5           L2 (least-squares) fit of 1 on the tracked interval
  PE_greedy      Polar Express Theorem 3.1 greedy minimax, cushion=0, no safety
  PE_ref         the shipped Polar Express design (cushion=0.02, safety 1e-2)
  PE_c<c>        greedy minimax with cushion c (design family)
Metrics
  eps_exact(K)   max_{sigma in [1e-3,1]} |P_K(sigma)-1| in float64
  S(K)           max amplification of one rounding injected at step j:
                 max_{j,sigma} prod_{l>=j} |p_l'(sigma_l)|
  floor_b(K)     faithful scalar model, values stored in b significant bits
  pred_b(K)      eps_exact(K) + 2^-b * S(K)
"""
import json
import numpy as np

np.random.seed(0)
L = 1e-3
GRID = np.logspace(np.log10(L), 0, 600)


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


def pe_coeffs(l0, K, cushion=0.0, safety=1.0):
    """Direct port of the authors' optimal_composition."""
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
            l = a * l + b * l**3 + c * l**5
            u = 2 - l
        else:
            l = a * l + b * l**3 + c * l**5
            u = 2 - l
        out.append((a, b, c))
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


NS5 = [(15 / 8, -10 / 8, 3 / 8)] * 12


def orbit(coeffs, x0):
    x = np.asarray(x0, float)
    tr = [x.copy()]
    for a, b, c in coeffs:
        x = a * x + b * x**3 + c * x**5
        tr.append(x)
    return np.array(tr)


def sensitivity(coeffs, x0):
    tr = orbit(coeffs, x0)
    best = np.ones_like(np.asarray(x0, float))
    for j in range(len(coeffs)):
        amp = np.ones_like(np.asarray(x0, float))
        for l in range(j, len(coeffs)):
            a, b, c = coeffs[l]
            amp *= np.abs(a + 3 * b * tr[l] ** 2 + 5 * c * tr[l] ** 4)
        best = np.maximum(best, amp)
    return best


def faithful(coeffs, x0, p):
    """One rounding per iteration (the value is stored in p bits)."""
    x = np.asarray(x0, float)
    tr = [x.copy()]
    for a, b, c in coeffs:
        x = q(a * x + b * x**3 + c * x**5, p)
        tr.append(x.copy())
    return np.array(tr)


def main():
    K = 8
    designs = {
        "NS5": NS5,
        "LSQ5": lsq_coeffs(L, K),
        "PE_greedy": pe_coeffs(L, K, cushion=0.0, safety=1.0),
        "PE_ref": pe_coeffs(L, K, cushion=0.02, safety=1.01),
    }
    for c in (0.005, 0.05, 0.1):
        designs[f"PE_c{c}"] = pe_coeffs(L, K, cushion=c, safety=1.01)

    out = {}
    print("=" * 100)
    print(f"[A] exact max error over sigma in [1e-3,1] (float64) and orbit sensitivity")
    print("=" * 100)
    print(f"{'design':12s}" + "".join(f"K={k:<7d}" for k in range(1, K + 1)))
    for nm, co in designs.items():
        tr = orbit(co, GRID)
        e = [float(np.max(np.abs(tr[k] - 1))) for k in range(1, K + 1)]
        out.setdefault("exact", {})[nm] = e
        print(f"{nm:12s}" + "".join(f"{v:8.1e}" for v in e))
    print()
    print(f"{'design':12s}" + "".join(f"K={k:<7d}" for k in range(1, K + 1)))
    for nm, co in designs.items():
        s = [float(np.max(sensitivity(co[:k], GRID))) for k in range(1, K + 1)]
        out.setdefault("sens", {})[nm] = s
        print(f"{nm:12s}" + "".join(f"{v:8.1e}" for v in s))

    print()
    print("=" * 100)
    print("[B] faithful rounding model: measured floor vs two-term prediction")
    print("    pred = eps_exact + 2^-b * S   (u = 2^-b for p = b+1 significant bits)")
    print("=" * 100)
    for pname, p in (("bf16", 8), ("fp16", 11), ("fp32", 24)):
        print(f"\n-- {pname}: measured max|P_K-1| (top) / predicted (bottom)")
        print(f"{'design':12s}" + "".join(f"K={k:<8d}" for k in range(1, K + 1)))
        for nm, co in designs.items():
            meas = [float(np.max(np.abs(faithful(co[:k], GRID, p)[k] - 1)))
                    for k in range(1, K + 1)]
            pred = [out["exact"][nm][k - 1] + 2.0 ** (-p) * out["sens"][nm][k - 1]
                    for k in range(1, K + 1)]
            kst = int(np.argmin(meas)) + 1
            print(f"{nm:12s}" + "".join(f"{v:8.1e}" for v in meas)
                  + f"  k*={kst}")
            print(f"{'':12s}" + "".join(f"{v:8.1e}" for v in pred))
            out.setdefault("floor", {}).setdefault(pname, {})[nm] = {
                "meas": meas, "pred": pred, "kstar": kst}
    with open("work/iclr-research/data/pilot1b_results.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
