#!/usr/bin/env python3
"""
Pilot 6 - Pinpointing and repairing the failure of the rounding certificate.

The two-term law of the shipped design family (cushion 0.02, safety 1.01) is

        ||X_K - UV^T||_2  ~  E_K + u * S(K),       u = 2^-bits,

with S(K) = max over sigma in [L,1] and over injection points of the product of
|p'(x_l)| along the exact orbit.  S(K) collapses to 1.0 once the orbit is fully
converged (L=1e-3, K>=7 at 8 bits) and the law then under-predicts the measured
error by a factor 60.  This script

  [A] diagnoses the collapse (which channel carries the floor, and why the
      diagonal derivative cannot see it),
  [B] races five computable replacements, all of which use only the
      coefficients and (L, K, bits),
  [C] reports the winner, a held-out matrix size, and a held-out dynamic range
      L=1e-4 that takes no part in the calibration.

Kernel model as in pilot1c/pilot5b: X and B stored in p significant bits, A=XX^T
not stored, all products accumulated in float64.
"""
import json
import os
import numpy as np

R = {"config": {}, "diagnosis": {}, "variants": {}, "summary": {}}
CAP = 1e12
DESIGNS = {"ref": (0.02, 1.01), "cush05": (0.05, 1.01), "cush10": (0.10, 1.01)}


# ---------------------------------------------------------------- building blocks
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
    u, l, out, win = 1.0, l0, [], []
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
        win.append((lo, u))
        l = a * l + b * l**3 + c * l**5
        u = 2 - l
    return out, win


def design(L, K, name):
    cu, sa = DESIGNS[name]
    return pe_coeffs(L, K, cu, sa)


def deriv(a, b, c, x):
    return a + 3 * b * x ** 2 + 5 * c * x ** 4


def gain(a, b, c, x, y):
    """off diagonal (i != j) gain of aX + b(XX^T)X + c(XX^T)^2 X on u_i v_j^T"""
    return a + b * (x * x + y * y) + c * (x ** 4 + y * y * (x * x + y * y))


def orbit(co, x0):
    x = np.asarray(x0, float)
    tr = [x.copy()]
    for a, b, c in co:
        x = a * x + b * x ** 3 + c * x ** 5
        tr.append(x.copy())
    return np.array(tr)


def sensitivity(co, x0):
    """max over sigma and injection point of the product of |p'| from j to K"""
    tr = orbit(co, x0)
    base = np.asarray(x0, float)
    best = np.ones_like(base)
    for j in range(len(co)):
        amp = np.ones_like(base)
        for l in range(j, len(co)):
            a, b, c = co[l]
            amp = amp * np.abs(deriv(a, b, c, tr[l]))
        best = np.maximum(best, amp)
    return best


def prefix_curve(co, grid):
    """for the *given* design: the sensitivity of every prefix of the trajectory.
    curve[t] = S of the first t coefficients of this design (t = 1..K)."""
    tr = orbit(co, grid)
    K = len(co)
    curve = []
    for t in range(1, K + 1):
        best = np.ones_like(np.asarray(grid, float))
        for j in range(t):
            amp = np.ones_like(best)
            for l in range(j, t):
                a, b, c = co[l]
                amp = amp * np.abs(deriv(a, b, c, tr[l]))
            best = np.maximum(best, amp)
        curve.append(float(np.max(best)))
    return curve


def mat_run(co, M, p, rndX=True, rndB=True, only=None):
    X = q(M / np.linalg.norm(M, 2), p) if (rndX and only in (None, 0)) else M / np.linalg.norm(M, 2)
    for i, (a, b, c) in enumerate(co, start=1):
        act = only is None or only == i
        A = X @ X.T
        B = b * A + c * (A @ A)
        if rndB and act:
            B = q(B, p)
        X = a * X + B @ X
        if rndX and act:
            X = q(X, p)
    return X


def make_M(d, L, seed=1):
    rs = np.random.RandomState(seed)
    U, _ = np.linalg.qr(rs.randn(d, d))
    V, _ = np.linalg.qr(rs.randn(d, d))
    return (U * np.logspace(0, np.log10(L), d)) @ V.T, U @ V.T


# ---------------------------------------------------------------- candidate definitions
def s_extended(co, grid, u, c):
    hi = 1 + c * u
    g = np.concatenate([grid, np.linspace(1 - 2 * u, hi, 600)])
    g = g[(g >= grid[0]) & (g <= hi)]
    with np.errstate(over="ignore", invalid="ignore"):
        v = float(np.max(sensitivity(co, g)))
    return CAP if (not np.isfinite(v)) or v > CAP else v


def s_disturbed(co, grid, u, frac=1.0):
    tr = orbit(co, grid)
    K = len(co)
    best = np.ones_like(np.asarray(grid, float))
    with np.errstate(over="ignore", invalid="ignore"):
        for j in range(K):
            for sgn in (1.0, -1.0):
                xx = tr[j] + sgn * frac * u
                amp = np.ones_like(best)
                for l in range(j, K):
                    a, b, c = co[l]
                    amp = amp * np.abs(deriv(a, b, c, xx))
                    xx = a * xx + b * xx**3 + c * xx**5
                best = np.maximum(best, amp)
    v = float(np.max(best))
    return CAP if (not np.isfinite(v)) or v > CAP else v


def s_truncated(co, win, grid, u, c):
    tr = orbit(co, grid)
    K = len(co)
    keep = [bool(win[l][1] - win[l][0] >= c * u) for l in range(K)]
    best = np.ones_like(np.asarray(grid, float))
    for j in range(K):
        amp = np.ones_like(best)
        for l in range(j, K):
            if not keep[l]:
                continue
            a, b, c_ = co[l]
            amp = amp * np.abs(deriv(a, b, c_, tr[l]))
        best = np.maximum(best, amp)
    v = float(np.max(best))
    return CAP if (not np.isfinite(v)) or v > CAP else v


def pair_grid(L, n=80):
    g = np.logspace(np.log10(L), 0, n)
    X, Y = np.meshgrid(g, g)
    a, b = X.ravel(), Y.ravel()
    for rr in (1.002, 1.02, 1.05, 1.1, 1.3, 1.6, 2.0, 4.0, 8.0):
        y2 = g * rr
        m = y2 <= 1.0
        a = np.concatenate([a, g[m]])
        b = np.concatenate([b, y2[m]])
    return a, b


def s_pair(co, pairs, K=None):
    K = K or len(co)
    x = np.asarray(pairs[0], float).copy()
    y = np.asarray(pairs[1], float).copy()
    tr = [(x.copy(), y.copy())]
    for a, b, c in co:
        x, y = a * x + b * x**3 + c * x**5, a * y + b * y**3 + c * y**5
        tr.append((x.copy(), y.copy()))
    best = np.ones_like(tr[0][0])
    with np.errstate(over="ignore", invalid="ignore"):
        for j in range(K):
            amp = np.ones_like(best)
            for l in range(j, K):
                a, b, c = co[l]
                amp = amp * np.abs(gain(a, b, c, tr[l][0], tr[l][1]))
            best = np.maximum(best, amp)
    v = float(np.max(best))
    return CAP if (not np.isfinite(v)) or v > CAP else v


# ---------------------------------------------------------------- driver
def main():
    Ls = [1e-2, 1e-3]
    ds = [64, 128]
    ps = [8, 11]
    Ks = list(range(2, 10))
    R["config"] = {"designs": {k: list(v) for k, v in DESIGNS.items()},
                   "Ls": Ls, "ds": ds, "bits": ps, "Ks": Ks,
                   "kernel": "X,B stored in p significant bits, A not stored, products float64",
                   "calibration": "C per definition = geometric midpoint of the band of err/(u*S) "
                                  "over floor-dominated cells (E_K <= 0.05 err) at d=64; "
                                  "d=128 and L=1e-4 are held out"}

    # ---------------- A. diagnosis
    print("=" * 104)
    print("[A] why the diagonal certificate collapses")
    print("=" * 104)
    diag_out = {}
    for L in (1e-3, 1e-2):
        gg = np.logspace(np.log10(L), 0, 1500)
        for K in (5, 7):
            co, _ = design(L, K, "ref")
            tr = orbit(co, np.array([L]))[:, 0]
            y0 = min(1.3 * L, 1.0)
            tp = orbit(co, np.array([L, y0]))
            dd = [abs(deriv(*co[l], tr[l])) for l in range(K)]
            oo = [abs(gain(*co[l], tp[l][0], tp[l][1])) for l in range(K)]
            diag_out[f"L{L}_K{K}"] = {"orbit": tr.tolist(), "diag": dd, "off": oo,
                                      "prod_diag": float(np.prod(dd)), "prod_off": float(np.prod(oo))}
            print(f"  L={L:<7g} K={K}  closest pair ({L:g}, {y0:.3g})")
            print("     |p'(x_l)|   : " + " ".join(f"{v:7.3f}" for v in dd) + f"   product={np.prod(dd):9.2e}")
            print("     |g(x_l,y_l)| : " + " ".join(f"{v:7.3f}" for v in oo) + f"   product={np.prod(oo):9.2e}")

    print("\n  channel decomposition: share of the floor by injection point (units of u)")
    ch = {}
    for L in (1e-3, 1e-2):
        M, ref = make_M(64, L)
        for K in (5, 9):
            co, _ = design(L, K, "ref")
            for p in (8, 11):
                u = 2.0 ** -p
                full = float(np.linalg.norm(mat_run(co, M, p) - ref, 2)) / u
                x0 = float(np.linalg.norm(mat_run(co, M, p, rndB=False, only=0) - ref, 2)) / u
                per = [float(np.linalg.norm(mat_run(co, M, p, only=j) - ref, 2)) / u for j in range(1, K + 1)]
                ch[f"L{L}_K{K}_p{p}"] = {"full": full, "X0": x0, "per_step": per}
                print(f"   L={L:<7g} K={K} p={p:<3d} full={full:7.2f}u  X0={x0:7.2f}u  steps="
                      + " ".join(f"{v:5.1f}" for v in per))
    R["diagnosis"] = {"prefix_derivatives": diag_out, "channel_decomposition": ch}

    # ---------------- B. the race
    print("\n" + "=" * 104)
    print("[B] five computable definitions over 192 cells (L K bits design d)")
    print("=" * 104)
    data = []
    for L in Ls:
        grid = np.logspace(np.log10(L), 0, 1500)
        pairs = pair_grid(L)
        for nm in DESIGNS:
            S = {}
            run_d = run_p = 1.0
            for K in Ks:
                co, win = design(L, K, nm)
                EK = float(np.max(np.abs(orbit(co, grid)[K] - 1)))
                pc = prefix_curve(co, grid)
                sd = pc[-1]
                sp = s_pair(co, pairs)
                run_d = max(run_d, sd)
                run_p = max(run_p, sp)
                S[K] = {"EK": EK, "S": min(sd, CAP), "prefix_run": min(max(pc), CAP),
                        "pair": sp, "pair_run": run_p, "ext": {}, "dist": {}, "trunc": {}}
                for p in ps:
                    u = 2.0 ** -p
                    for c in (1, 4, 16):
                        S[K]["ext"][(p, c)] = s_extended(co, grid, u, c)
                    S[K]["dist"][p] = s_disturbed(co, grid, u)
                    for c in (1, 2, 4):
                        S[K]["trunc"][(p, c)] = s_truncated(co, win, grid, u, c)
            for K in Ks:
                co, _ = design(L, K, nm)
                for d in ds:
                    M, ref = make_M(d, L)
                    for p in ps:
                        u = 2.0 ** -p
                        err = float(np.linalg.norm(mat_run(co, M, p) - ref, 2))
                        data.append({"L": L, "design": nm, "K": K, "d": d, "p": p, "u": u,
                                     "err": err, "EK": S[K]["EK"], "S": S[K]["S"],
                                     "S_prefix_run": S[K]["prefix_run"], "S_pair": S[K]["pair"],
                                     "S_pair_run": S[K]["pair_run"],
                                     "S_ext1": S[K]["ext"][(p, 1)], "S_ext4": S[K]["ext"][(p, 4)],
                                     "S_ext16": S[K]["ext"][(p, 16)], "S_dist": S[K]["dist"][p],
                                     "S_trunc1": S[K]["trunc"][(p, 1)], "S_trunc2": S[K]["trunc"][(p, 2)],
                                     "S_trunc4": S[K]["trunc"][(p, 4)]})
    print(f"  {len(data)} cells measured")

    variants = [("0 current S(K)", "S"),
                ("1 extended window c=1", "S_ext1"),
                ("1 extended window c=4", "S_ext4"),
                ("1 extended window c=16", "S_ext16"),
                ("2 disturbed orbit (u)", "S_dist"),
                ("3 truncation c=1", "S_trunc1"),
                ("3 truncation c=2", "S_trunc2"),
                ("3 truncation c=4", "S_trunc4"),
                ("4 pair sensitivity", "S_pair"),
                ("5 prefix ratchet", "S_prefix_run"),
                ("5b pair ratchet", "S_pair_run")]

    print(f"\n  {'definition':24s} {'C':>8s} {'band(cal)':>16s} {'cal':>9s} {'d=128':>9s} {'all192':>9s} {'max dev':>8s}")
    summary = {}
    for vname, key in variants:
        cal = [r["err"] / (r["u"] * r[key]) for r in data
               if r["d"] == 64 and r["EK"] <= 0.05 * r["err"] and r[key] > 0 and np.isfinite(r[key])]
        lo, hi = min(cal), max(cal)
        C = float(np.sqrt(lo * hi))
        rows = [(r, (r["EK"] + C * r["u"] * r[key]) / r["err"]) for r in data]
        def cnt(sel):
            rr = np.array([q for r, q in rows if sel(r)])
            return int(np.sum((rr >= 0.5) & (rr <= 2.0))), len(rr), float(np.max(np.maximum(rr, 1 / rr)))
        n_cal, nn_cal, dev_cal = cnt(lambda r: r["d"] == 64)
        n_128, nn_128, dev_128 = cnt(lambda r: r["d"] == 128)
        n_all, nn_all, dev_all = cnt(lambda r: True)
        summary[vname] = {"key": key, "C": C, "band": [lo, hi], "cal_pass": n_cal, "cal_n": nn_cal,
                          "d128_pass": n_128, "d128_n": nn_128, "all_pass": n_all, "all_n": nn_all,
                          "max_deviation": dev_all}
        print(f"  {vname:24s} {C:8.4f} {lo:6.3f}-{hi:<9.3f} {n_cal:4d}/{nn_cal:<4d} {n_128:4d}/{nn_128:<4d} "
              f"{n_all:5d}/{nn_all:<3d} {dev_all:8.2f}")
        R["variants"][vname] = summary[vname]

    # ---------------- C. winner detail
    win = "5 prefix ratchet"
    Cw = summary[win]["C"]
    print("\n" + "=" * 104)
    print(f"[C] winner: err ~ E_K + C u S_pert(K),  S_pert(K) = max over prefixes K' <= K of S(K'),  C = {Cw:.4f}")
    print("=" * 104)
    print(f"  {'L':>7s} {'design':>7s} {'K':>2s} {'d':>4s} {'p':>3s} {'err':>10s} {'pred':>10s} {'pred/err':>8s} "
          f"{'err/u':>8s} {'S_pert':>9s} {'S(K)':>8s}")
    worst = 0.0
    for r in data:
        pred = r["EK"] + Cw * r["u"] * r["S_prefix_run"]
        ratio = pred / r["err"]
        worst = max(worst, abs(np.log(ratio)))
        print(f"  {r['L']:7g} {r['design']:>7s} {r['K']:2d} {r['d']:4d} {r['p']:3d} {r['err']:10.3e} {pred:10.3e} "
              f"{ratio:8.2f} {r['err']/r['u']:8.2f} {r['S_prefix_run']:9.1f} {r['S']:8.2f}")
    print(f"  worst deviation from 1 over all cells: factor {np.exp(worst):.2f}")

    resid = {}
    print("\n  residual band of err/(u S_pert) by L (what a single C must absorb):")
    for L in Ls:
        v = [r["err"] / (r["u"] * r["S_prefix_run"]) for r in data if r["L"] == L and r["EK"] <= 0.05 * r["err"]]
        resid[L] = [float(np.min(v)), float(np.median(v)), float(np.max(v))]
        print(f"   L={L:<7g} n={len(v):2d} min={np.min(v):.4f} med={np.median(v):.4f} max={np.max(v):.4f}")

    # ---------------- D. held-out dynamic range L=1e-4
    print("\n" + "=" * 104)
    print("[D] held-out dynamic range L=1e-4 (not used anywhere above)")
    print("=" * 104)
    Lx = 1e-4
    grid = np.logspace(np.log10(Lx), 0, 1500)
    pairs = pair_grid(Lx)
    ext = []
    for nm in DESIGNS:
        S = {}
        run = 1.0
        for K in Ks:
            co, _ = design(Lx, K, nm)
            EK = float(np.max(np.abs(orbit(co, grid)[K] - 1)))
            pc = prefix_curve(co, grid)
            run = max(run, max(pc))
            S[K] = {"EK": EK, "S": pc[-1], "run": min(run, CAP)}
        for K in Ks:
            co, _ = design(Lx, K, nm)
            for d in ds:
                M, ref = make_M(d, Lx)
                for p in ps:
                    u = 2.0 ** -p
                    err = float(np.linalg.norm(mat_run(co, M, p) - ref, 2))
                    ext.append({"L": Lx, "design": nm, "K": K, "d": d, "p": p, "u": u, "err": err,
                                "EK": S[K]["EK"], "S_pert": S[K]["run"], "S": S[K]["S"]})
    print(f"  {'design':>7s} {'K':>2s} {'d':>4s} {'p':>3s} {'err':>10s} {'pred':>10s} {'pred/err':>8s} {'err/u':>8s} {'S_pert':>9s}")
    ok = 0
    for r in ext:
        pred = r["EK"] + Cw * r["u"] * r["S_pert"]
        ratio = pred / r["err"]
        ok += 0.5 <= ratio <= 2.0
        print(f"  {r['design']:>7s} {r['K']:2d} {r['d']:4d} {r['p']:3d} {r['err']:10.3e} {pred:10.3e} "
              f"{ratio:8.2f} {r['err']/r['u']:8.2f} {r['S_pert']:9.1f}")
    print(f"  L=1e-4 held out: {ok}/{len(ext)} inside 2x")

    gchk = {}
    for n in (60, 110):
        px, py = pair_grid(1e-3, n)
        co, _ = design(1e-3, 6, "ref")
        gchk[n] = s_pair(co, (px, py))
    print(f"\n  S_pair grid resolution check (L=1e-3,K=6): n=60 {gchk[60]:.3f}  n=110 {gchk[110]:.3f}")

    R["summary"] = {"C_winner": Cw, "residual_band_by_L": {str(k): v for k, v in resid.items()},
                    "heldout_L1e-4": {"pass": ok, "n": len(ext)},
                    "S_pair_grid_check": gchk, "n_cells": len(data)}
    R["data"] = data
    R["extrapolation_L1e-4"] = ext
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pilot6_results.json")
    with open(out, "w") as f:
        json.dump(R, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
