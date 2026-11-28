#!/usr/bin/env python3
"""
Pilot 5 - Is there an accuracy/sensitivity lower bound for one polynomial step?

For an odd polynomial P of degree d, on [L,1]:

    eps(P) = max_{[L,1]} |P(x) - 1|          (accuracy of one step)
    S(P)   = max_{[L,1]} |P'(x)|             (amplification of one rounding, K=1)

Question (Theorem 2 draft): must S(P) >= c * eps(P)^(-alpha)?

Each point of the frontier is the exact solution of a linear program

    min  t
    s.t. |P(x_i) - 1| <= eps          on a grid of [L,1]
         |P(y_j)|     <= 2            on a grid of [0,1]
         |P'(z_k)|    <= t            on a grid of [L,1]

and, separately, min sum|a_k| under the same accuracy/boundedness constraints.
The LP solution is re-checked on a 20x finer grid (the reported eps/S are the
verified values, not the LP's internal ones).
"""
import json
import numpy as np
from scipy.optimize import linprog
import scipy.sparse as sp

R = {"note": "eps/S values are re-verified on a 20x finer grid than the LP grid"}
BOUND = 2.0


def make_grids(L, n1=400, n2=400, nb=400):
    xg = np.unique(np.concatenate([np.linspace(L, 1, n1), np.logspace(np.log10(L), 0, n2)]))
    return xg, np.linspace(0, 1, nb)


def fine_grids(L):
    xg = np.unique(np.concatenate([np.linspace(L, 1, 8001), np.logspace(np.log10(L), 0, 4001)]))
    return xg, np.linspace(0, 1, 8001)


def build(L, d, mode="deriv"):
    xg, xb = make_grids(L)
    ks = np.arange(1, d + 1, 2)
    V = xg[:, None] ** ks[None, :]
    Vb = xb[:, None] ** ks[None, :]
    D = ks[None, :] * xg[:, None] ** (ks[None, :] - 1)
    n = len(ks)
    zx = sp.csr_matrix((len(xg), 1))
    zb = sp.csr_matrix((len(xb), 1))
    if mode == "deriv":
        blocks = [sp.hstack([V, zx]), sp.hstack([-V, zx]),
                  sp.hstack([Vb, zb]), sp.hstack([-Vb, zb])]
        ones = sp.csr_matrix(np.ones((len(xg), 1)))
        blocks += [sp.hstack([D, -ones]), sp.hstack([-D, -ones])]
        nv = n + 1
    else:                                   # l1: extra n slacks
        zx2 = sp.csr_matrix((len(xg), n))
        zb2 = sp.csr_matrix((len(xb), n))
        I = sp.identity(n, format="csr")
        blocks = [sp.hstack([V, zx2]), sp.hstack([-V, zx2]),
                  sp.hstack([Vb, zb2]), sp.hstack([-Vb, zb2]),
                  sp.hstack([I, -I]), sp.hstack([-I, -I])]
        nv = 2 * n
    A = sp.vstack(blocks).tocsr()
    return A, len(xg), len(xb), ks, nv, n


def solve(L, d, eps, mode="deriv", pre=None):
    A, nx, nb, ks, nv, n = pre if pre else build(L, d, mode)
    base = [np.ones(nx) + eps, -np.ones(nx) + eps, BOUND * np.ones(nb), BOUND * np.ones(nb)]
    if mode == "deriv":
        b = np.concatenate(base + [np.zeros(2 * nx)])
        c = np.zeros(nv); c[-1] = 1.0
        bnds = [(None, None)] * n + [(0, None)]
    else:
        b = np.concatenate(base + [np.zeros(2 * n)])
        c = np.zeros(nv); c[:n] = 1.0
        bnds = [(None, None)] * n + [(0, None)] * n
    r = linprog(c, A_ub=A, b_ub=b, bounds=bnds, method="highs")
    if r.status != 0:
        return None
    return r.x[:n]


def verify(L, d, a, fg=None):
    xg, xb = fg if fg else fine_grids(L)
    ks = np.arange(1, d + 1, 2)
    P = a @ (xg[None, :] ** ks[:, None])
    Dp = (ks * a) @ (xg[None, :] ** (ks[:, None] - 1))
    Pb = a @ (xb[None, :] ** ks[:, None])
    return (float(np.max(np.abs(P - 1))), float(np.max(np.abs(Dp))),
            float(np.max(np.abs(Pb))), float(np.abs(a).sum()))


def min_eps(L, d, fg=None, lo=1e-7, hi=3.0):
    pre = build(L, d, "deriv")
    for _ in range(50):
        mid = np.sqrt(lo * hi)
        if solve(L, d, mid, "deriv", pre) is not None:
            hi = mid
        else:
            lo = mid
        if hi - lo < 2e-9 * hi:
            break
    return hi


def main():
    print("=" * 96)
    print("[A] minimal achievable single-step accuracy E_d(L)  (|P-1|<=eps on [L,1], |P|<=2 on [0,1])")
    print("=" * 96)
    print(f"{'d':>4s}" + "".join(f"{f'L={L:g}':>14s}" for L in (0.3, 0.1, 0.03)))
    Etab = {}
    for d in (3, 5, 7, 9, 13, 17):
        row = []
        for L in (0.3, 0.1, 0.03):
            e = min_eps(L, d)
            Etab[f"{d}|{L}"] = e
            row.append(e)
        print(f"{d:4d}" + "".join(f"{v:14.3e}" for v in row))
    R["E_d"] = Etab

    print("\n" + "=" * 96)
    print("[B] frontier: minimal S = max|P'| on [L,1] subject to max|P-1| <= eps")
    print("=" * 96)
    for L in (0.3, 0.1, 0.03):
        fg = fine_grids(L)
        print(f"\n-- L={L:g}")
        print(f"{'d':>4s}{'eps':>11s}{'S_min':>11s}{'eps_chk':>11s}{'|P|max':>9s}"
              f"{'l1_min':>11s}{'eps_chk':>11s}")
        for d in (3, 5, 7, 9, 13, 17):
            Ed = Etab[f"{d}|{L}"]
            eps_list = np.geomspace(min(Ed * 1.02, 0.99), 1.05, 11)
            pred = build(L, d, "deriv")
            prel = build(L, d, "l1")
            rows = []
            for eps in eps_list:
                a = solve(L, d, eps, "deriv", pred)
                if a is None:
                    continue
                ec, S, pm, l1 = verify(L, d, a, fg)
                al = solve(L, d, eps, "l1", prel)
                l1m, ecl = (verify(L, d, al, fg)[3], verify(L, d, al, fg)[0]) if al is not None else (None, None)
                rows.append({"eps": float(eps), "eps_verified": ec, "S_min": S,
                             "P_max": pm, "l1_at_Smin": l1,
                             "l1_min": l1m, "eps_verified_l1": ecl})
                print(f"{d:4d}{eps:11.2e}{S:11.3e}{ec:11.2e}{pm:9.2f}"
                      f"{l1m:11.3e}{ecl:11.2e}")
            R.setdefault("frontier", {}).setdefault(f"L{L}", {})[str(d)] = rows

    print("\n" + "=" * 96)
    print("[C] log-log slope of the frontier: S_min ~ eps^(-alpha), fitted on the achievable range")
    print("=" * 96)
    fits = {}
    for L in (0.3, 0.1, 0.03):
        for d in (3, 5, 7, 9, 13, 17):
            rows = R["frontier"][f"L{L}"][str(d)]
            rows = [r for r in rows if r["S_min"] and r["eps_verified"]]
            if len(rows) < 3:
                continue
            x = np.log([r["eps_verified"] for r in rows])
            y = np.log([r["S_min"] for r in rows])
            A = np.vstack([x, np.ones_like(x)]).T
            sl, ic = np.linalg.lstsq(A, y, rcond=None)[0]
            res = y - (A @ np.array([sl, ic]))
            fits[f"L{L}|d{d}"] = {"alpha": float(-sl), "C": float(np.exp(ic)),
                                  "rms_log": float(np.sqrt(np.mean(res ** 2))),
                                  "eps_lo": float(min(r["eps_verified"] for r in rows)),
                                  "eps_hi": float(max(r["eps_verified"] for r in rows))}
            print(f"L={L:<5g} d={d:<3d} alpha={-sl:6.3f}  C={np.exp(ic):9.3e}"
                  f"  rms={np.sqrt(np.mean(res**2)):.3f}"
                  f"  range=[{fits[f'L{L}|d{d}']['eps_lo']:.1e},{fits[f'L{L}|d{d}']['eps_hi']:.1e}]")
    R["fits"] = fits

    print("\n" + "=" * 96)
    print("[D] degree-free test: at a fixed eps, does S_min fall as d grows?")
    print("=" * 96)
    cross = {}
    for L in (0.3, 0.1):
        for eps in (0.6, 0.45, 0.35, 0.25):
            vals = {}
            for d in (3, 5, 7, 9, 13, 17):
                if eps <= Etab[f"{d}|{L}"] * 1.0001:
                    continue
                a = solve(L, d, eps, "deriv", build(L, d, "deriv"))
                if a is None:
                    continue
                ec, S, pm, l1 = verify(L, d, a)
                vals[str(d)] = {"eps": ec, "S_min": S, "l1": l1}
            if len(vals) >= 2:
                cross[f"L{L}|eps{eps}"] = vals
                print(f"L={L:g} eps={eps:.2f}: " + "  ".join(
                    f"d={d}:S={v['S_min']:.2e}" for d, v in vals.items()))
    R["cross_d"] = cross

    with open("work/iclr-research/data/pilot5_results.json", "w") as f:
        json.dump(R, f, indent=1)


if __name__ == "__main__":
    main()
