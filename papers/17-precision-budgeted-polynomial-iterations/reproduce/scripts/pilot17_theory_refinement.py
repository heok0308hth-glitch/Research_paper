#!/usr/bin/env python3
"""
Pilot 17 - the last verification pass.

 [1] Is the certificate spectrum-blind?  At fixed (ell, K, design) the ratchet is a
     function of the design alone, so the prediction c*u*S_pert is identical for every
     spectrum with the same minimum singular value.  Re-run the 48 held-out cells of
     pilot 16 in four spectrum families and measure the prediction/measurement ratio
     per family, together with the ratio of the measured floor between families at
     matched (d, ell, p).

 [2] Where the family dependence lives.  Keep every rounding error the kernel actually
     injects, propagate each one linearly to the output, and split the first-order
     subspace response into the input cast and the in-loop roundings.  Compare with the
     measurement.  The cast response of Theorem 3 is the response to the perturbation
     E = q(Mhat,p) - Mhat, so this separates "a quantity the scalar orbit cannot see"
     from "the orbit".

 [3] The two-channel (pair) recursion.  Replace the scalar orbit by the pair orbit with
     the exact off-diagonal gain g(a,b,c,x,y), ratcheted over prefixes, and measure the
     constant it leaves on each family.  Also test spectrum-only predictors of the cast
     response: the pair-weight operator W = [2/(s_i+s_j)] and the modelled entry variance
     of the cast error.

 [4] The real Muon momenta (128x32, 100 of the 500 recorded in pilot 11): the same split,
     to see whether a real spectrum behaves like the fitted family or like a plateau.

 [5] The stability boundary.  The damped sequence maps the top of its window exactly onto
     the top of the next window (zero slack) when safety = 1; damping by s buys a slack.
     Compute the slack exactly from the coefficients, measure the divergence census over
     a (p, ell, safety) grid, and locate the critical safety.

Kernel model as in pilots 1-16.  Writes pilot17_results.json.
"""
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pilot7_pert_aware_design import q, pe_coeffs, stats, prefix_curve
from pilot6_s_pert import orbit, deriv, gain

R = {}
C_CERT = 0.142
K_DEF, CUSH, SAFE = 6, 0.02, 1.01


def n2(X):
    return float(np.linalg.norm(X, 2))


def spectra(kind, d, L, seed=7):
    rs = np.random.RandomState(seed)
    U, _ = np.linalg.qr(rs.randn(d, d))
    V, _ = np.linalg.qr(rs.randn(d, d))
    if kind == "logspace":
        sv = np.logspace(0, np.log10(L), d)
    elif kind == "plateau":
        sv = np.ones(d)
        sv[d // 2:] = L
    elif kind == "gap":
        h = d // 2
        a = np.logspace(0, 0.5 * np.log10(L), h)
        b = np.logspace(0.5 * np.log10(L) - 2, np.log10(L), d - h)
        sv = np.concatenate([a, np.clip(b, L, 1.0)])
        sv = sv / sv[0]
    elif kind == "power":
        sv = 1.0 / (1.0 + np.arange(d))
        sv = sv / sv[0]
        sv = np.maximum(sv, L)
    else:
        raise ValueError(kind)
    M = (U * sv) @ V.T
    return M, U @ V.T, sv


def kernel(co, X0, p):
    X = q(X0, p)
    for a, b, c in co:
        A = X @ X.T
        B = q(b * A + c * (A @ A), p)
        X = q(a * X + B @ X, p)
    return X


# ------------------------------------------------------------------ [1]
def section1():
    print("=" * 104)
    print("[1] is the certificate spectrum-blind?  the same 48 held-out cells, four families")
    print("=" * 104)
    out = {}
    print(f"  {'family':>9} {'d':>4} {'ell':>6} {'p':>3} {'S':>8} {'pred':>9} {'measured':>9} {'pred/meas':>9} {'err/u':>7}")
    for d in (48, 96, 256):
        for kind in ("logspace", "plateau", "gap", "power"):
            for L in (1e-1, 1e-2):
                for p in (8, 12):
                    M, ref, sv = spectra(kind, d, L)
                    co, _ = pe_coeffs(L, K_DEF, CUSH, SAFE)
                    X0 = M / n2(M)
                    Xk = kernel(co, X0, p)
                    e_state = n2(Xk - ref)
                    Wk, _, Vtk = np.linalg.svd(Xk, full_matrices=False)
                    e = n2(Wk @ Vtk - ref)
                    grid = np.logspace(np.log10(L), 0, 400)
                    EK, S, _ = stats(co, grid, K_DEF)
                    u = 2.0 ** -p
                    pred = EK + C_CERT * u * S
                    out[f"{kind}_d{d}_L{L:g}_p{p}"] = dict(
                        d=d, spectrum=kind, ell=L, p=p, err=e, err_state=e_state, EK=EK, S=S,
                        pred=pred, ratio=pred / e if e else None, err_over_u=e / u)
                    print(f"  {kind:>9} {d:4d} {L:6g} {p:3d} {S:8.1f} {pred:9.3e} {e:9.3e} "
                          f"{pred / e if e else float('nan'):9.3f} {e / u:7.2f}")
    R["s1_cells"] = out
    # family aggregates
    agg = {}
    for kind in ("logspace", "plateau", "gap", "power"):
        v = np.array([c["ratio"] for c in out.values() if c["spectrum"] == kind], float)
        agg[kind] = {"n": int(v.size), "median": float(np.median(v)),
                     "min": float(v.min()), "max": float(v.max()),
                     "factor_spread": float(v.max() / v.min())}
    # S must be identical across families at fixed (d, ell, p) -- it is a function of
    # (ell, K, design) only; check exactly, and the family ratio of the measured floor.
    fam_ratio = {}
    for d in (48, 96, 256):
        for L in (1e-1, 1e-2):
            for p in (8, 12):
                key = f"d{d}_L{L:g}_p{p}"
                base = out[f"logspace_d{d}_L{L:g}_p{p}"]
                fam_ratio[key] = {k: out[f"{k}_d{d}_L{L:g}_p{p}"]["err"] / base["err"]
                                  for k in ("plateau", "gap", "power")}
                fam_ratio[key]["S_all_equal"] = len({round(out[f"{k}_d{d}_L{L:g}_p{p}"]["S"], 9)
                                                     for k in ("logspace", "plateau", "gap", "power")}) == 1
    R["s1_family"] = agg
    R["s1_floor_ratio_vs_logspace"] = fam_ratio
    pr = np.array([v["plateau"] for v in fam_ratio.values()], float)
    print(f"\n  S identical across the four families in all {len(fam_ratio)} matched cells: "
          f"{all(v['S_all_equal'] for v in fam_ratio.values())}")
    print(f"  measured floor, plateau/logspace at matched (d,ell,p): {pr.min():.2f}-{pr.max():.2f}"
          f" (median {np.median(pr):.2f})")
    for kind in ("logspace", "plateau", "gap", "power"):
        a = agg[kind]
        print(f"  {kind:>9}: pred/meas median {a['median']:.3f}, range [{a['min']:.3f}, {a['max']:.3f}], "
              f"spread {a['factor_spread']:.1f}x")


# ------------------------------------------------------------------ [2]
def kernel_record(co, X0, p):
    """run the kernel and keep every rounding error it injects.

    Y[s] is the state at the start of step s+1 (Y[0] = the input cast); an injection is
    (kind, step, error) with step 1-based, and the B error enters the step's output as
    eta @ Y[step-1]."""
    X = q(X0, p)
    Y = [X.copy()]
    inj = [("cast", 0, q(X0, p) - X0)]
    for i, (a, b, c) in enumerate(co, start=1):
        A = X @ X.T
        Bex = b * A + c * (A @ A)
        B = q(Bex, p)
        inj.append(("B", i, B - Bex))
        Xex = a * X + B @ X
        Xn = q(Xex, p)
        inj.append(("X", i, Xn - Xex))
        Y.append(Xn.copy())
        X = Xn
    return X, Y, inj


def propagate(co, Y, dX, s):
    """linearised propagation of a perturbation added to Y[s] through steps s+1..K."""
    for m in range(s, len(co)):
        a, b, c = co[m]
        X = Y[m]
        A = X @ X.T
        dA = dX @ X.T + X @ dX.T
        dX = a * dX + (b * dA + c * (dA @ A + A @ dA)) @ X + (b * A + c * (A @ A)) @ dX
    return dX


def skew_resp(dX, Q):
    """polar-factor (subspace) response of a perturbation of a near-polar state Q."""
    G = Q.T @ dX
    return float(np.linalg.norm(dX - Q @ ((G + G.T) / 2.0), 2))


def section2():
    print("\n" + "=" * 104)
    print("[2] decomposition of the first-order subspace response: input cast vs in-loop roundings")
    print("=" * 104)
    out = {}
    print(f"  {'family':>9} {'ell':>6} {'p':>3} {'measured':>9} {'cast':>9} {'in-loop':>9} {'total':>9} "
          f"{'cast share':>10} {'total/meas':>10}")
    for d in (48, 96):
        for kind in ("logspace", "plateau", "gap", "power"):
            for L in (1e-1, 1e-2):
                for p in (8,):
                    M, ref, sv = spectra(kind, d, L)
                    co, _ = pe_coeffs(L, K_DEF, CUSH, SAFE)
                    X0 = M / n2(M)
                    X, Y, inj = kernel_record(co, X0, p)
                    Wk, _, Vtk = np.linalg.svd(X, full_matrices=False)
                    e = n2(Wk @ Vtk - ref)
                    parts = {"cast": np.zeros((d, d)), "B": np.zeros((d, d)), "X": np.zeros((d, d))}
                    per = {}
                    for kind_i, at, E in inj:
                        if kind_i == "cast":
                            dX = propagate(co, Y, E, 0)
                            key = "cast"
                        else:
                            dX = propagate(co, Y, E @ Y[at - 1] if kind_i == "B" else E, at)
                            key = kind_i
                        parts[key] = parts[key] + dX
                        per[f"{kind_i}{at}"] = skew_resp(dX, ref)
                    tot = parts["cast"] + parts["B"] + parts["X"]
                    e_cast = skew_resp(parts["cast"], ref)
                    e_loop = skew_resp(parts["B"] + parts["X"], ref)
                    e_tot = skew_resp(tot, ref)
                    out[f"{kind}_d{d}_L{L:g}_p{p}"] = dict(
                        d=d, spectrum=kind, ell=L, p=p, measured=e, cast=e_cast, inloop=e_loop,
                        total=e_tot, cast_share=e_cast / e_tot if e_tot else None,
                        total_over_measured=e_tot / e if e else None,
                        per_injection=per)
                    print(f"  {kind:>9} {L:6g} {p:3d} {e:9.3e} {e_cast:9.3e} {e_loop:9.3e} {e_tot:9.3e} "
                          f"{e_cast / e_tot:10.3f} {e_tot / e:10.3f}")
    R["s2_decomposition"] = out
    for kind in ("logspace", "plateau", "gap", "power"):
        cs = np.array([c["cast_share"] for c in out.values() if c["spectrum"] == kind])
        print(f"  {kind:>9}: cast share of the first-order response {cs.min():.3f}-{cs.max():.3f}")


# ------------------------------------------------------------------ [3]
def pair_ratchet(co, sv):
    """prefix ratchet of the two-channel (pair) recursion, over the actual spectrum."""
    K = len(co)
    x = np.repeat(sv, len(sv))
    y = np.tile(sv, len(sv))
    tr = [(x.copy(), y.copy())]
    for a, b, c in co:
        x, y = a * x + b * x ** 3 + c * x ** 5, a * y + b * y ** 3 + c * y ** 5
        tr.append((x.copy(), y.copy()))
    best = 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(1, K + 1):
            for j in range(t):
                amp = np.ones_like(tr[0][0])
                for l in range(j, t):
                    a, b, c = co[l]
                    amp = amp * np.abs(gain(a, b, c, tr[l][0], tr[l][1]))
                m = float(np.nanmax(amp))
                if m > best:
                    best = m
    return best


def section3():
    print("\n" + "=" * 104)
    print("[3] two-channel (pair) recursion, and spectrum-only predictors of the cast response")
    print("=" * 104)
    out = {}
    print(f"  {'family':>9} {'d':>4} {'ell':>6} {'p':>3} {'S':>8} {'S_pair':>9} {'c_raw':>6} {'c_pair':>7} "
          f"{'|R|2':>9} {'rmsF/vhat':>9} {'P_F':>9} {'|R|2/P_F':>8}")
    for d in (48, 96, 256):
        for kind in ("logspace", "plateau", "gap", "power"):
            for L in (1e-1, 1e-2):
                for p in (8,):
                    M, ref, sv = spectra(kind, d, L)
                    co, _ = pe_coeffs(L, K_DEF, CUSH, SAFE)
                    X0 = M / n2(M)
                    u = 2.0 ** -p
                    e = n2(kernel(co, X0, p) - ref)
                    grid = np.logspace(np.log10(L), 0, 400)
                    EK, S, _ = stats(co, grid, K_DEF)
                    Sp = pair_ratchet(co, sv)
                    E0 = q(X0, p) - X0
                    U = np.linalg.svd(M, full_matrices=False)[0]
                    Vt = np.linalg.svd(M, full_matrices=False)[2]
                    F = U.T @ E0 @ Vt.T
                    W = 1.0 / (sv[:, None] + sv[None, :])
                    Rm = (F - F.T) * W
                    nR = n2(Rm)
                    vhat = u / np.sqrt(3.0) * np.sqrt((sv ** 2).sum()) / d
                    rmsF = float(np.sqrt((F ** 2).mean()))
                    PF = vhat * float(np.sqrt((W ** 2).sum()))
                    P2 = vhat * float(np.linalg.norm(W, 2))
                    out[f"{kind}_d{d}_L{L:g}"] = dict(
                        d=d, spectrum=kind, ell=L, p=p, err=e, S=S, S_pair=Sp,
                        c_raw=(e - EK) / (u * S), c_pair=(e - EK) / (u * Sp),
                        R=nR, rmsF=rmsF, vhat=vhat, P_F=PF, P_2=P2,
                        ratio_PF=nR / PF, ratio_P2=nR / P2)
                    print(f"  {kind:>9} {d:4d} {L:6g} {p:3d} {S:8.1f} {Sp:9.1f} "
                          f"{out[f'{kind}_d{d}_L{L:g}']['c_raw']:6.3f} "
                          f"{out[f'{kind}_d{d}_L{L:g}']['c_pair']:7.4f} {nR:9.2e} "
                          f"{rmsF / vhat:9.2f} {PF:9.2e} {nR / PF:8.2f} {nR / P2:8.2f}")
    R["s3_pair"] = out
    for key in ("c_raw", "c_pair", "ratio_PF", "ratio_P2"):
        v = np.array([c[key] for c in out.values()], float)
        print(f"\n  {key}: overall [{v.min():.3f}, {v.max():.3f}] x{v.max() / v.min():.1f}")
        for kind in ("logspace", "plateau", "gap", "power"):
            w = np.array([c[key] for c in out.values() if c["spectrum"] == kind], float)
            print(f"    {kind:>9}: median {np.median(w):8.3f}  [{w.min():.3f}, {w.max():.3f}]")
    # S_pair/S
    rat = np.array([c["S_pair"] / c["S"] for c in out.values()], float)
    print(f"\n  S_pair / S: {rat.min():.2f}-{rat.max():.2f}")


# ------------------------------------------------------------------ [4]
def section4():
    print("\n" + "=" * 104)
    print("[4] the real Muon momenta (128x32): cast share and the constant, 100 of 500")
    print("=" * 104)
    npz = np.load(os.path.join(HERE, "pilot11_momentum.npz"))
    upds = npz["upds"]
    decays = npz["decays"] if "decays" in npz.files else None
    ell_win = 0.1485
    co, _ = pe_coeffs(ell_win, K_DEF, CUSH, SAFE)
    grid = np.logspace(np.log10(ell_win), 0, 400)
    EK, S, _ = stats(co, grid, K_DEF)
    out = {}
    print(f"  {'idx':>4} {'dyn':>6} {'measured':>9} {'cast':>9} {'in-loop':>9} {'total':>9} "
          f"{'cast share':>10} {'c_raw':>7}")
    rows = []
    for idx in range(0, 500, 5):
        M = upds[idx].astype(float)
        s = np.linalg.svd(M, compute_uv=False)
        Mn = n2(M)
        if not np.isfinite(Mn) or Mn == 0:
            continue
        X0 = M / Mn
        ref = np.linalg.svd(X0, full_matrices=False)
        Q = ref[0] @ ref[2]
        m, n = X0.shape
        p = 8
        u = 2.0 ** -p
        X, Y, inj = kernel_record(co, X0, p)
        Wk, _, Vtk = np.linalg.svd(X, full_matrices=False)
        e = n2(Wk @ Vtk - Q)
        parts = {"cast": np.zeros_like(X0), "B": np.zeros_like(X0), "X": np.zeros_like(X0)}
        for kind_i, at, E in inj:
            if kind_i == "cast":
                dX = propagate(co, Y, E, 0)
            else:
                dX = propagate(co, Y, E @ Y[at - 1] if kind_i == "B" else E, at)
            parts[kind_i] = parts[kind_i] + dX
        tot = parts["cast"] + parts["B"] + parts["X"]
        e_cast = skew_resp(parts["cast"], Q)
        e_loop = skew_resp(parts["B"] + parts["X"], Q)
        e_tot = skew_resp(tot, Q)
        rec = dict(idx=idx, dyn=float(s[-1] / s[0]), measured=e, cast=e_cast, inloop=e_loop,
                   total=e_tot, cast_share=e_cast / e_tot if e_tot else None,
                   c_raw=(e - EK) / (u * S), S=S)
        out[idx] = rec
        rows.append(rec)
        if idx % 50 == 0:
            print(f"  {idx:4d} {rec['dyn']:6.3f} {e:9.3e} {e_cast:9.3e} {e_loop:9.3e} {e_tot:9.3e} "
                  f"{rec['cast_share']:10.3f} {rec['c_raw']:7.3f}")
    R["s4_real"] = out
    def band(key):
        v = np.array([r[key] for r in rows], float)
        v = v[np.isfinite(v)]
        return {"n": int(v.size), "median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
                "p90": float(np.percentile(v, 90)), "min": float(v.min()), "max": float(v.max())}
    R["s4_summary"] = {"ell_used": ell_win, "S": S, "cast_share": band("cast_share"),
                       "c_raw": band("c_raw"),
                       "total_over_measured": band("total"), "n_cells": len(rows)}
    b = R["s4_summary"]
    print(f"\n  {len(rows)} matrices at K={K_DEF}, p=8, ell={ell_win}: S={S:.1f}")
    print(f"  cast share of the first-order response: median {b['cast_share']['median']:.3f} "
          f"[{b['cast_share']['min']:.3f}, {b['cast_share']['max']:.3f}]")
    print(f"  c_raw = err/(u S): median {b['c_raw']['median']:.3f} "
          f"p10-p90 {b['c_raw']['p10']:.3f}-{b['c_raw']['p90']:.3f}")


# ------------------------------------------------------------------ [5]
def slack_of(co, L):
    """exact-arithmetic slack at the top of every window, and the edge derivative."""
    x = [L]
    for a, b, c in co:
        x.append(a * x[-1] + b * x[-1] ** 3 + c * x[-1] ** 5)
    sl, dv = [], []
    for t, (a, b, c) in enumerate(co):
        u = 1.0 if t == 0 else 2 - x[t]
        p_u = a * u + b * u ** 3 + c * u ** 5
        sl.append(2 - x[t + 1] - p_u)
        dv.append(abs(deriv(a, b, c, u)))
    return np.array(sl), np.array(dv)


DIV_THR = 5.0   # the error cap is 2, so a floor above it is saturation, not divergence


def error_of(co, X0, ref, p):
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            e = n2(kernel(co, X0, p) - ref)
        return e
    except Exception:
        return float("inf")


def diverges(co, X0, ref, p):
    e = error_of(co, X0, ref, p)
    return (not np.isfinite(e)) or e > DIV_THR


def section5():
    print("\n" + "=" * 104)
    print("[5] the stability boundary: exact slack, divergence census, critical safety")
    print("=" * 104)
    out = {"slack": {}, "census": {}, "critical": {}}
    print("  exact-arithmetic slack at the top of the window (first steps) and the gain there:")
    for L in (1e-1, 1e-2, 1e-3):
        for s in (1.0, 1.002, 1.005, 1.01):
            co, _ = pe_coeffs(L, K_DEF, CUSH, s)
            sl, dv = slack_of(co, L)
            out["slack"][f"L{L:g}_s{s:g}"] = {"slack": [float(v) for v in sl],
                                              "edge_deriv": [float(v) for v in dv]}
            print(f"   L={L:6g} safety={s:<6}: slack_t = " + " ".join(f"{v:+.2e}" for v in sl[:4])
                  + f"   |p'(u_t)| = " + " ".join(f"{v:.1f}" for v in dv[:4]))
    print("\n  divergence census, 12 spectra (d in {48,64}, six seeds), K=6:")
    cells = {}
    for L in (1e-1, 1e-2, 1e-3):
        cells[L] = []
        for d in (48, 64):
            for seed in range(6):
                M, ref, sv = spectra("logspace", d, L, seed)
                cells[L].append((M / n2(M), ref))
    for p in (4, 5, 6, 8, 10, 12, 16):
        for L in (1e-1, 1e-2, 1e-3):
            for s in (1.0, 1.002, 1.005, 1.008, 1.01):
                co, _ = pe_coeffs(L, K_DEF, CUSH, s)
                k = sum(diverges(co, X0, ref, p) for X0, ref in cells[L])
                emax = max(error_of(co, X0, ref, p) for X0, ref in cells[L])
                epred = max((C_CERT * 2.0 ** -p * stats(co, np.logspace(np.log10(L), 0, 400), K_DEF)[1])
                            for _ in [0])
                out["census"][f"p{p}_L{L:g}_s{s:g}"] = dict(p=p, ell=L, safety=s, n=k, of=len(cells[L]),
                                                           max_err=float(emax), floor_pred=float(epred))
            print(f"   p={p:2d} L={L:6g}: " + "  ".join(
                f"s={s:g}:{out['census'][f'p{p}_L{L:g}_s{s:g}']['n']:2d}/12"
                for s in (1.0, 1.002, 1.005, 1.008, 1.01)))
    print("\n  critical safety (smallest safety with zero divergences in the census), bisection:")
    for p in (4, 5, 6, 8, 10, 12, 16):
        for L in (1e-1, 1e-2, 1e-3):
            lo, hi = 1.0, 1.05
            def ok(s):
                co, _ = pe_coeffs(L, K_DEF, CUSH, s)
                return sum(diverges(co, X0, ref, p) for X0, ref in cells[L]) == 0
            if ok(1.0):
                out["critical"][f"p{p}_L{L:g}"] = dict(p=p, ell=L, s_crit=1.0, note="stable at 1.0")
                print(f"   p={p:2d} L={L:6g}: stable at safety 1.0")
                continue
            if not ok(hi):
                out["critical"][f"p{p}_L{L:g}"] = dict(p=p, ell=L, s_crit=None, note="diverges at 1.05")
                print(f"   p={p:2d} L={L:6g}: not stable even at 1.05")
                continue
            for _ in range(14):
                mid = 0.5 * (lo + hi)
                if ok(mid):
                    hi = mid
                else:
                    lo = mid
            co_hi, _ = pe_coeffs(L, K_DEF, CUSH, hi)
            sl, dv = slack_of(co_hi, L)
            out["critical"][f"p{p}_L{L:g}"] = dict(p=p, ell=L, s_crit=float(hi),
                                                   slack_first=float(sl[0]),
                                                   edge_deriv_first=float(dv[0]),
                                                   slack_over_halfu_gain=float(sl[0] / (2.0 ** -(p + 1) * dv[0])))
            print(f"   p={p:2d} u={2 ** -p:.5f} L={L:6g}: critical safety {hi:.5f}  "
                  f"slack {sl[0]:.3e}  (s-1)/(u/2) {(hi - 1) / 2 ** -(p + 1):.2f}  "
                  f"slack/((u/2)|p'|) {sl[0] / (2.0 ** -(p + 1) * dv[0]):.2f}")
    R["s5_stability"] = out


def main():
    R["config"] = {"K": K_DEF, "cushion": CUSH, "safety": SAFE, "c": C_CERT,
                   "numpy": np.__version__}
    section1()
    section2()
    section3()
    section4()
    section5()
    path = os.path.join(HERE, "pilot17_results.json")
    with open(path, "w") as f:
        json.dump(R, f, indent=1, default=float)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
