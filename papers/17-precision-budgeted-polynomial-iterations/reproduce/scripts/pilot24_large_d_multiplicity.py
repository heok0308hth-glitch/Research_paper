#!/usr/bin/env python3
"""
Pilot 24 - three follow-up measurements on the CPU side.

 [A] - is the multiplicity orthogonal at large d?  Pilot 22 measured the
     quadrature of the 13 first-order responses (cast + two per step) against the
     measured floor on 16 cells at d = 64, 96 and got 0.97-1.08.  Here the same
     quantity is followed to d = 128, 256, 512 over four spectrum families, two
     dynamic ranges and three seeds (96 cells, 24 of them at d = 512), together with
     the mean and the largest |cosine| between the 13 response matrices, which is
     the assumption the quadrature rests on.

 [B] - the per-injection coefficients of pilot 22, as quartiles.  The text quotes a
     median of 0.57 for kappa_inj / (2 sqrt2 g(state)) with a range of 0.003-2.6; a
     range that wide hides where the mass is.  Quartiles, an IQR, and the same split by
     channel (input cast against in-loop roundings), by family and by d, read from
     pilot22_results.json; the coefficients of the 96 cells of [A] are pooled the same
     way so the table is not a 16-cell artefact.

 [C] - the ratchet, and what it would take to promote it to a theorem.  S_pert(K) is
     the prefix maximum of the orbit sensitivity; the floor it predicts is checked in
     the text with a calibrated 0.142.  Which conditions make it a bound rather than a
     calibration?
       (c1) the first-order bound is exact for the channel it is written for: a rounding
            injected at step j reaches the output multiplied by the suffix product
            prod_{l>j} |p_l'|, so prod over injections is bounded by the number of
            injections times the largest suffix product, and the largest suffix product
            is bounded by S(K) - and hence by S_pert(K) - with no condition at all.
            Written down and checked: err <= (2K+1) u S_pert for every cell.
       (c2) that bound is for the singular-value channel.  The measured floor is
            subspace, whose gain is the off-diagonal one, gamma_l(x,y) = a + b(x^2+y^2)
            + c(x^4 + x^2 y^2 + y^4), a function of the pair and not of p_l'.  The pair
            ratchet G_pert replaces the p_l' by gamma_l.  Is gamma dominated by p_l' on
            the pair's interval?  Measured ratio over all reachable pairs and steps:
            unbounded at a Newton-terminated step (p' -> 0 while gamma -> 1/2), so no
            spectral or monotonicity condition can turn S_pert into a bound for the
            subspace floor.  What survives is a conditional statement with the condition
            measured: G_pert <= S_pert in this many of these cells, and err <= (2K+1)
            u G_pert in all of them.
       (c3) the counterexample, in closed form: the Newton step (a,b,c) = (3/2,-1/2,0)
            has p'(1) = 0 and gamma(1,1) = 1/2; the shipped family at ell = 1e-3, K = 7
            inherits it (p'(1) = 0, gamma(1,1) as measured) and the terminal sensitivity
            collapses to ~2e-3 while the measured floor stays at 0.26 - the 66x gap the
            text attributes to the ratchet.

Kernel model as in pilots 1-23: state and intermediates stored in p significant bits,
products accumulated in float64, exact matrix products in between.  Writes
pilot24_results.json.
"""
import gc
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pilot18_c_scaling_and_e5m2_fix as p18  # noqa: E402
from pilot18_c_scaling_and_e5m2_fix import spectra, n2, sub_err  # noqa: E402
from pilot7_pert_aware_design import pe_coeffs, deriv, orbit, prefix_curve  # noqa: E402
from pilot6_s_pert import gain  # noqa: E402
from pilot17_theory_refinement import kernel_record, propagate, skew_resp  # noqa: E402

RES = os.path.join(HERE, "pilot24_results.json")
P22 = os.path.join(HERE, "pilot22_results.json")
if not os.path.exists(P22):                      # archive layout: results live in ../results
    P22 = os.path.join(os.path.dirname(HERE), "results", "pilot22_results.json")
R = {}
K_DEF, CUSH, SAFE = 6, 0.02, 1.01
SQ2 = float(np.sqrt(2.0))
TWO_SQ2 = 2.0 * SQ2
CAP = 2.0


def save():
    tmp = RES + ".tmp"
    with open(tmp, "w") as f:
        json.dump(R, f, indent=1)
    os.replace(tmp, RES)


def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else None


def span(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return (float(min(xs)), float(max(xs))) if xs else (None, None)


def rms(x):
    x = np.asarray(x, float)
    return float(np.sqrt((x ** 2).mean()))


def quant(xs):
    """min, Q1, median, Q3, max, IQR on the finite entries."""
    xs = np.array([x for x in xs if x is not None and np.isfinite(x)], float)
    if not xs.size:
        return None
    q = np.percentile(xs, [0, 25, 50, 75, 100])
    return dict(n=int(xs.size), min=float(q[0]), q1=float(q[1]), median=float(q[2]),
                q3=float(q[3]), max=float(q[4]), iqr=float(q[3] - q[1]))


# ================================================================== [A]
def one_cell(kind, d, L, seed, p=8):
    """the kernel record of one cell, its 13 injections and their responses."""
    M, ref, sv = spectra(kind, d, L, seed)
    co, win = pe_coeffs(L, K_DEF, CUSH, SAFE)
    X0 = M / n2(M)
    s0 = sv / sv.max()
    X, Y, inj = kernel_record(co, X0, p)
    e_meas = sub_err(X, ref)
    s = s0.copy()
    orbit_s = [s.copy()]
    for (a, b, c) in co:
        s = a * s + b * s ** 3 + c * s ** 5
        orbit_s.append(s.copy())

    def wscale(sv_t):
        s_ = sv_t / sv_t[0]
        D = 1.0 / (s_[:, None] + s_[None, :])
        return float(np.linalg.norm(D, "fro") / np.sqrt(s_.size))

    scale0 = wscale(sv)
    resp, quad, vecs = [], 0.0, []
    for k_inj, at, E in inj:
        dX = propagate(co, Y, E if k_inj != "B" else E @ Y[at - 1], at)
        r_ = skew_resp(dX, ref)
        rho = rms(E)
        scale_t = wscale(orbit_s[at]) if orbit_s[at][0] > 0 else None
        resp.append(dict(kind=k_inj, at=at, resp=r_, rho=rho, scale_t=scale_t,
                         kappa=(r_ / (rho * scale_t)) if (rho and scale_t) else None))
        vecs.append(dX.ravel())
        quad += r_ ** 2
    quad = float(np.sqrt(quad))
    # cosines between the 13 response matrices, mean |cos| and max |cos|
    V = np.stack(vecs, axis=0)
    Vn = V / np.maximum(np.linalg.norm(V, axis=1), 1e-300)[:, None]
    C = np.abs(Vn @ Vn.T)
    np.fill_diagonal(C, 0.0)
    co_pos = C[np.triu_indices(C.shape[0], 1)]
    del V, Vn, vecs
    u = 2.0 ** -p
    denom = u * rms(X0) * scale0
    cast = [r for r in resp if r["kind"] == "cast"][0]
    loop = [r for r in resp if r["kind"] != "cast"]
    out = dict(d=d, spectrum=kind, ell=L, seed=seed, p=p,
               C_total=e_meas / denom, C_cast=cast["resp"] / denom,
               C_loop=float(np.sqrt(sum(r["resp"] ** 2 for r in loop))) / denom,
               C_quad=quad / denom, measured=e_meas, quadrature=quad,
               quadrature_over_measured=quad / e_meas if e_meas else None,
               cast_share=cast["resp"] / (quad if quad else np.nan),
               n_inj=len(resp),
               cos_mean=float(co_pos.mean()) if len(co_pos) else None,
               cos_max=float(co_pos.max()) if len(co_pos) else None,
               kappa=[r["kappa"] for r in resp],
               resp_frac=[r["resp"] / quad for r in resp] if quad else [],
               injections=resp)
    return out


def sectionA():
    print("=" * 122)
    print("[A] - is the multiplicity still orthogonal at large d?")
    print("    quadrature of the 13 first-order responses against the measured floor;")
    print("    cos = |cosine| between the 13 response matrices, which is what quadrature assumes")
    print("=" * 122)
    dims = (64, 128, 256, 512)
    fams = ("logspace", "plateau", "gap", "power")
    ells = (1e-1, 1e-2)
    seeds = (0, 1, 2)
    cells = []
    for d in dims:
        for kind in fams:
            for L in ells:
                for sd in seeds:
                    cells.append(one_cell(kind, d, L, sd))
        p18._CACHE.clear()
        gc.collect()
        q = [c["quadrature_over_measured"] for c in cells if c["d"] == d]
        cos = [c["cos_mean"] for c in cells if c["d"] == d]
        print(f"  d={d:4d}: {len(q):2d} cells   quad/meas {span(q)[0]:.3f}-{span(q)[1]:.3f}"
              f" (median {med(q):.3f})   mean|cos| {span(cos)[0]:.3f}-{span(cos)[1]:.3f}", flush=True)

    R["a_cells"] = cells
    by_d = {}
    print(f"\n  {'d':>5} {'n':>3} {'quad/meas':>10} {'median':>8} {'mean|cos|':>10} "
          f"{'max|cos|':>9} {'cast share':>10} {'C_total':>8}")
    for d in dims:
        q = [c["quadrature_over_measured"] for c in cells if c["d"] == d]
        cm = [c["cos_mean"] for c in cells if c["d"] == d]
        cx = [c["cos_max"] for c in cells if c["d"] == d]
        cs = [c["cast_share"] for c in cells if c["d"] == d]
        ct = [c["C_total"] for c in cells if c["d"] == d]
        by_d[d] = dict(n=len(q), quad_lo=span(q)[0], quad_hi=span(q)[1], quad_median=med(q),
                       cos_mean_lo=span(cm)[0], cos_mean_hi=span(cm)[1],
                       cos_max_lo=span(cx)[0], cos_max_hi=span(cx)[1],
                       cos_max_median=med(cx), cast_share_median=med(cs),
                       C_total_median=med(ct), C_total_lo=span(ct)[0], C_total_hi=span(ct)[1])
        print(f"  {d:5d} {len(q):3d} {span(q)[0]:5.3f}-{span(q)[1]:5.3f} {med(q):8.3f} "
              f"{med(cm):10.3f} {med(cx):9.3f} {med(cs):10.3f} {med(ct):8.3f}")
    q = [c["quadrature_over_measured"] for c in cells]
    big = [c["quadrature_over_measured"] for c in cells if c["d"] >= 256]
    R["a_summary"] = dict(
        n_cells=len(cells), n_cells_large=len(big),
        quad_lo=span(q)[0], quad_hi=span(q)[1], quad_median=med(q),
        quad_lo_large=span(big)[0], quad_hi_large=span(big)[1], quad_median_large=med(big),
        frac10=float(np.mean([abs(c["quadrature_over_measured"] - 1) <= 0.10 for c in cells])),
        cos_mean=med([c["cos_mean"] for c in cells]),
        cos_max=span([c["cos_max"] for c in cells])[1],
        by_d=by_d)
    print(f"\n  all {len(cells)} cells: quad/meas {span(q)[0]:.3f}-{span(q)[1]:.3f}"
          f" (median {med(q):.3f}); within 10%: "
          f"{100 * R['a_summary']['frac10']:.0f}%;  d>=256 ({len(big)} cells):"
          f" {span(big)[0]:.3f}-{span(big)[1]:.3f} (median {med(big):.3f})")
    print(f"  mean |cos| between the 13 responses: median {med([c['cos_mean'] for c in cells]):.3f},"
          f" largest {span([c['cos_max'] for c in cells])[1]:.3f}")
    # d trend of the quadrature ratio and of C_total at fixed (family, ell)
    tr = {}
    for kind in fams:
        for L in ells:
            v = [med([c["quadrature_over_measured"] for c in cells
                      if c["spectrum"] == kind and c["ell"] == L and c["d"] == d]) for d in dims]
            ct = [med([c["C_total"] for c in cells
                       if c["spectrum"] == kind and c["ell"] == L and c["d"] == d]) for d in dims]
            tr[f"{kind}_L{L:g}"] = dict(quad=v, C_total=ct)
    R["a_trend"] = tr
    save()


# ================================================================== [B] W13
def sectionB():
    print("\n" + "=" * 122)
    print("[B] - the per-injection coefficient, as quartiles")
    print("    kappa_inj / (2 sqrt2 g(state)): 1.0 means the one-step law of the state the")
    print("    rounding enters; the spread is why the multiplicity is counted, not modelled")
    print("=" * 122)
    with open(P22, encoding="utf-8") as f:
        p22 = json.load(f)
    recs = []
    for key, cell in p22["c_cells"].items():
        for r in cell["injections"]:
            if r.get("kappa_over_pred") is not None:
                recs.append(dict(key=key, d=cell["d"], spectrum=cell["spectrum"], ell=cell["ell"],
                                 kind=r["kind"], at=r["at"], v=r["kappa_over_pred"],
                                 kappa=r.get("kappa"), resp=r["resp"]))
    out = {"source": "pilot22_results.json:c_cells[*].injections[*].kappa_over_pred",
           "n": len(recs)}
    out["all"] = quant([r["v"] for r in recs])
    out["by_channel"] = {k: quant([r["v"] for r in recs if r["kind"] == k])
                         for k in ("cast", "B", "X")}
    out["by_family"] = {k: quant([r["v"] for r in recs if r["spectrum"] == k])
                        for k in ("logspace", "plateau", "gap", "power")}
    out["by_d"] = {str(d): quant([r["v"] for r in recs if r["d"] == d]) for d in (64, 96)}
    out["by_ell"] = {f"{L:g}": quant([r["v"] for r in recs if abs(r["ell"] - L) < 1e-12])
                     for L in (1e-1, 1e-2)}
    out["frac_in_half2"] = float(np.mean([0.5 <= r["v"] <= 2 for r in recs]))
    out["frac_in_quartile_band"] = float(np.mean([
        out["all"]["q1"] <= r["v"] <= out["all"]["q3"] for r in recs]))
    print(f"  {out['n']} injections from pilot 22")
    for k in ("all",):
        a = out[k]
        print(f"  {k:>14}: min {a['min']:.4f}  Q1 {a['q1']:.3f}  median {a['median']:.3f}"
              f"  Q3 {a['q3']:.3f}  max {a['max']:.2f}  IQR {a['iqr']:.3f}")
    for grp in ("by_channel", "by_family", "by_d", "by_ell"):
        for k, a in out[grp].items():
            print(f"  {grp[3:]:>9} {k:>9}: min {a['min']:.4f}  Q1 {a['q1']:.3f}"
                  f"  median {a['median']:.3f}  Q3 {a['q3']:.3f}  max {a['max']:.2f}")
    print(f"  inside [0.5, 2]: {100 * out['frac_in_half2']:.0f}%  "
          f"inside the IQR band: {100 * out['frac_in_quartile_band']:.0f}%")

    # the same, pooled over the 96 large-d cells of [A]; here the coefficient is the raw
    # kappa (response / (rms(E) scale_t)), because g is not recomputed on that grid --
    # a flat spectrum gives 2 sqrt 2 = 2.83, and the point of the row is that it does not
    # drift with d
    vals = [v for c in R["a_cells"] for v in c["kappa"]]
    out["large_d_kappa_raw"] = quant(vals)
    out["large_d_by_d"] = {str(d): quant([v for c in R["a_cells"] if c["d"] == d
                                          for v in c["kappa"]]) for d in (64, 128, 256, 512)}
    ch = {}
    for c in R["a_cells"]:
        for r in c["injections"]:
            ch.setdefault(r["kind"], []).append(r["kappa"])
    out["large_d_by_channel"] = {k: quant(v) for k, v in ch.items()}
    a = out["large_d_kappa_raw"]
    print(f"\n  pooled over the 96 cells of [A], raw kappa ({a['n']} injections): min {a['min']:.4f}  "
          f"Q1 {a['q1']:.3f}  median {a['median']:.3f}  Q3 {a['q3']:.3f}  max {a['max']:.2f}")
    for k, v in out["large_d_by_d"].items():
        print(f"    d={k:>4}: median {v['median']:.3f}  Q1 {v['q1']:.3f}  Q3 {v['q3']:.3f}"
              f"  max {v['max']:.2f}")
    R["b_w13"] = out
    save()


# ================================================================== [C] A4
def wscale_from(sv):
    s_ = np.asarray(sv, float)
    s_ = s_ / s_[0]
    D = 1.0 / (s_[:, None] + s_[None, :])
    return float(np.linalg.norm(D, "fro") / np.sqrt(s_.size))


def orbit_env(co, grid):
    """D_l = max over sigma of |p_l'(x_{l-1}(sigma))| and the envelope product."""
    tr = orbit(co, grid)
    D = [float(np.max(np.abs(deriv(a, b, c, tr[l])))) for l, (a, b, c) in enumerate(co)]
    return D, tr


def envelope_max(co, D):
    """max over injection points j of prod_{l>j} D_l (the bound of (c1)/(c2))."""
    K = len(co)
    best, arg = 0.0, None
    for j in range(-1, K):
        p = 1.0
        for l in range(j + 1, K):
            p *= D[l]
        if p > best or arg is None:
            best, arg = p, j
    return best, arg


def pair_ratchet(co, grid):
    """prefix ratchet of the off-diagonal gain over the pair orbit of the grid."""
    K = len(co)
    x = np.repeat(grid, grid.size)
    y = np.tile(grid, grid.size)
    tr = [(x.copy(), y.copy())]
    for a, b, c in co:
        x, y = a * x + b * x ** 3 + c * x ** 5, a * y + b * y ** 3 + c * y ** 5
        tr.append((x.copy(), y.copy()))
    best, where = 1.0, None
    gmin = []
    for t in range(1, K + 1):
        for j in range(t):
            amp = np.ones_like(tr[0][0])
            for l in range(j, t):
                a, b, c = co[l]
                amp = amp * np.abs(gain(a, b, c, tr[l][0], tr[l][1]))
            m = float(np.nanmax(amp))
            if m > best:
                best, where = m, (j, t)
    for l, (a, b, c) in enumerate(co):
        gmin.append(float(np.nanmin(gain(a, b, c, tr[l][0], tr[l][1]))))
    return best, where, gmin, tr


def gain_domination(co, trp, nsl=161):
    """max over pairs and steps of gamma / (max |p'| on the pair's interval)."""
    worst, where = 0.0, None
    frac_ok, n = 0, 0
    for l, (a, b, c) in enumerate(co):
        X, Y = trp[l]
        lo, hi = np.minimum(X, Y), np.maximum(X, Y)
        g = np.abs(gain(a, b, c, X, Y))
        u = np.linspace(0.0, 1.0, nsl)[None, :] * (hi - lo)[:, None] + lo[:, None]
        dmax = np.abs(deriv(a, b, c, u)).max(axis=1)
        r = g / np.maximum(dmax, 1e-300)
        frac_ok += int(np.sum(r <= 1.0))
        n += r.size
        if r.max() > worst:
            worst, where = float(r.max()), l
    return worst, where, frac_ok / n


def sectionC():
    print("\n" + "=" * 122)
    print("[C] - the ratchet: the bound it gives, and the condition it would need")
    print("=" * 122)
    # ---------------- part 1: design-level sensitivity for K = 5, 6, 7
    grid = np.logspace(-4, 0, 401)
    design = {}
    for L in (1e-1, 1e-2, 1e-3):
        for K in (5, 6, 7):
            co, win = pe_coeffs(L, K, CUSH, SAFE)
            g_ = np.logspace(np.log10(L), 0, 401)
            pc = prefix_curve(co, g_)
            S_pert = float(max(pc))
            D, tr = orbit_env(co, g_)
            E, argE = envelope_max(co, D)
            gridp = np.unique(np.concatenate([np.logspace(np.log10(L), 0, 200),
                                              np.linspace(max(L, 0.98), 1.0, 9)]))
            G, whereG, gmin, trp = pair_ratchet(co, gridp)
            worst, whereW, frac_ok = gain_domination(co, trp)
            key = f"L{L:g}_K{K}"
            design[key] = dict(ell=L, K=K, prefix_curve=pc, S_pert=S_pert, S_terminal=float(pc[-1]),
                               D=D, envelope=E, envelope_arg=int(argE),
                               G_pair=G, G_pair_where=whereG, gmin_step=gmin,
                               gain_over_interval_deriv=worst, gain_worst_step=whereW,
                               gain_domination_frac=frac_ok,
                               ratio_G_over_Spert=G / S_pert, ratio_env_over_Spert=E / S_pert,
                               ratio_Sterm_over_Spert=pc[-1] / S_pert)
            print(f"  ell={L:g} K={K}: S_pert {S_pert:8.2f}  S(K) {pc[-1]:8.3f}"
                  f"  env {E:9.2f} ({E / S_pert:5.2f}x)  pair-ratchet {G:8.2f}"
                  f" ({G / S_pert:5.3f}x)  gamma/interval-p' max {worst:9.2e} (step {whereW})")
            p18._CACHE.clear()
    R["c_design"] = design

    # ---------------- part 2: the certified bound against measured cells
    print("\n  certified bound (2K+1) u S_pert against the measured floor, d = 64, p = 8")
    print(f"  {'family':>9} {'ell':>6} {'K':>2} {'measured':>9} {'err/u':>8} {'S_pert':>8}"
          f" {'(2K+1)S':>9} {'ok':>3} {'G_pair':>8} {'(2K+1)G':>9} {'ok':>3} {'c_raw':>6} {'c_pair':>6}")
    cells = []
    for kind in ("logspace", "plateau", "gap", "power"):
        for L in (1e-1, 1e-2, 1e-3):
            for K in (6, 7):
                d, p = 64, 8
                M, ref, sv = spectra(kind, d, L, 0)
                co, win = pe_coeffs(L, K, CUSH, SAFE)
                X0 = M / n2(M)
                X, Y, inj = kernel_record(co, X0, p)
                e = sub_err(X, ref)
                u = 2.0 ** -p
                g_ = np.logspace(np.log10(L), 0, 401)
                pc = prefix_curve(co, g_)
                S_pert = float(max(pc))
                gridp = np.unique(np.concatenate([np.logspace(np.log10(L), 0, 200),
                                                  np.linspace(max(L, 0.98), 1.0, 9)]))
                G, whereG, gmin, trp = pair_ratchet(co, gridp)
                ninj = len(inj)
                cert_s = (2 * K + 1) * u * S_pert
                cert_g = (2 * K + 1) * u * G
                cells.append(dict(d=d, spectrum=kind, ell=L, K=K, p=p, measured=e,
                                  saturated=bool(np.isfinite(e) and e > 0.5 * CAP),
                                  err_over_u=e / u if np.isfinite(e) else None, S_pert=S_pert,
                                  S_terminal=float(pc[-1]), cert_ratchet=cert_s,
                                  cert_ok=bool(np.isfinite(e) and e <= cert_s),
                                  cert_slack=cert_s / e if (e and np.isfinite(e)) else None,
                                  G_pair=G, cert_pair=cert_g,
                                  cert_pair_ok=bool(np.isfinite(e) and e <= cert_g),
                                  cert_pair_slack=cert_g / e if (e and np.isfinite(e)) else None,
                                  c_raw=(e / (u * S_pert)) if np.isfinite(e) else None,
                                  c_pair=(e / (u * G)) if np.isfinite(e) else None,
                                  ratio_G_over_Spert=G / S_pert, n_inj=ninj))
                c = cells[-1]
                print(f"  {kind:>9} {L:6g} {K:2d} {e:9.3e} {c['err_over_u']:8.2f}"
                      f" {S_pert:8.2f} {cert_s:9.2e} {'Y' if c['cert_ok'] else 'N':>3}"
                      f" {G:8.2f} {cert_g:9.2e} {'Y' if c['cert_pair_ok'] else 'N':>3}"
                      f" {c['c_raw']:6.3f} {c['c_pair']:6.3f}")
                p18._CACHE.clear()
                gc.collect()
    R["c_cells"] = cells
    sat = [c for c in cells if c["saturated"]]
    clean = [c for c in cells if not c["saturated"]]
    ok1 = [c["cert_ok"] for c in cells]
    ok2 = [c["cert_pair_ok"] for c in cells]
    R["c_summary"] = dict(
        n=len(cells), certified_ratchet_holds=int(sum(ok1)),
        certified_pair_holds=int(sum(ok2)),
        slack_ratchet_median=med([c["cert_slack"] for c in cells]),
        slack_ratchet_lo=span([c["cert_slack"] for c in cells])[0],
        slack_ratchet_hi=span([c["cert_slack"] for c in cells])[1],
        slack_pair_median=med([c["cert_pair_slack"] for c in cells]),
        slack_pair_lo=span([c["cert_pair_slack"] for c in cells])[0],
        slack_pair_hi=span([c["cert_pair_slack"] for c in cells])[1],
        c_raw_median=med([c["c_raw"] for c in cells]),
        c_raw_lo=span([c["c_raw"] for c in cells])[0],
        c_raw_hi=span([c["c_raw"] for c in cells])[1],
        c_pair_median=med([c["c_pair"] for c in cells]),
        c_pair_lo=span([c["c_pair"] for c in cells])[0],
        c_pair_hi=span([c["c_pair"] for c in cells])[1],
        n_saturated=len(sat),
        c_raw_median_clean=med([c["c_raw"] for c in clean]),
        c_raw_lo_clean=span([c["c_raw"] for c in clean])[0],
        c_raw_hi_clean=span([c["c_raw"] for c in clean])[1],
        c_pair_median_clean=med([c["c_pair"] for c in clean]),
        c_pair_lo_clean=span([c["c_pair"] for c in clean])[0],
        c_pair_hi_clean=span([c["c_pair"] for c in clean])[1],
        cert_ok_clean=int(sum(c["cert_ok"] for c in clean)),
        G_le_Spert=int(sum(c["ratio_G_over_Spert"] <= 1.0 for c in cells)),
        G_over_Spert_lo=span([c["ratio_G_over_Spert"] for c in cells])[0],
        G_over_Spert_hi=span([c["ratio_G_over_Spert"] for c in cells])[1],
        gain_domination_cells=int(sum(design[k]["gain_domination_frac"] >= 0.999
                                     for k in design)),
        n_design=len(design))
    s = R["c_summary"]
    print(f"\n  certified (2K+1)uS_pert holds in {s['certified_ratchet_holds']}/{s['n']} cells"
          f" (slack {s['slack_ratchet_lo']:.1f}-{s['slack_ratchet_hi']:.1f}x,"
          f" median {s['slack_ratchet_median']:.1f}x)")
    print(f"  certified (2K+1)uG_pair holds in {s['certified_pair_holds']}/{s['n']} cells"
          f" (slack {s['slack_pair_lo']:.1f}-{s['slack_pair_hi']:.1f}x)")
    print(f"  c_raw {s['c_raw_lo']:.3f}-{s['c_raw_hi']:.3f} (median {s['c_raw_median']:.3f})"
          f"   c_pair {s['c_pair_lo']:.3f}-{s['c_pair_hi']:.3f} (median {s['c_pair_median']:.3f})")
    print(f"  G_pair <= S_pert in {s['G_le_Spert']}/{s['n']} measured cells;"
          f" G/S_pert {s['G_over_Spert_lo']:.3f}-{s['G_over_Spert_hi']:.3f}")
    print(f"  {s['n_saturated']} saturated cells (measured > 1.0, at the cap); excluding them:"
          f" c_raw {s['c_raw_lo_clean']:.3f}-{s['c_raw_hi_clean']:.3f}"
          f" (median {s['c_raw_median_clean']:.3f}), c_pair {s['c_pair_lo_clean']:.3f}"
          f"-{s['c_pair_hi_clean']:.3f} (median {s['c_pair_median_clean']:.3f})")
    print(f"  gain domination gamma <= max|p'| on the pair's interval: holds pointwise in"
          f" {s['gain_domination_cells']}/{s['n_design']} designs")

    # ---------------- part 3: the Newton counterexample, in closed form
    a, b, c = 1.5, -0.5, 0.0
    xs = np.linspace(0.5, 1.0, 6)
    newt = dict(a=a, b=b, c=c, p_prime_at_1=float(deriv(a, b, c, 1.0)),
                gamma_at_1=float(gain(a, b, c, 1.0, 1.0)),
                p_prime_table=[[float(x), float(deriv(a, b, c, x)),
                                float(gain(a, b, c, x, x))] for x in xs])
    co, win = pe_coeffs(1e-3, 7, CUSH, SAFE)
    g_ = np.logspace(-3, 0, 401)
    pc = prefix_curve(co, g_)
    anewt = co[-1]
    shipped = dict(ell=1e-3, K=7, last_step=[float(v) for v in anewt],
                   last_p_prime_at_1=float(deriv(*anewt, 1.0)),
                   last_gamma_at_1=float(gain(*anewt, 1.0, 1.0)),
                   S_pert=float(max(pc)), S_terminal=float(pc[-1]),
                   ratio=float(max(pc) / max(pc[-1], 1e-300)))
    R["c_newton"] = dict(step=newt, shipped=shipped)
    print(f"\n  Newton step: p'(1) = {newt['p_prime_at_1']:.3f}, gamma(1,1) = {newt['gamma_at_1']:.3f}"
          f"  -> the diagonal derivative and the off-diagonal gain are different objects")
    print(f"  shipped ell=1e-3 K=7 last step {['%.3f' % v for v in anewt]}:"
          f" p'(1) = {shipped['last_p_prime_at_1']:.3f}, gamma(1,1) = {shipped['last_gamma_at_1']:.3f},"
          f" S_pert/S(K) = {shipped['ratio']:.0f}")
    save()


def main():
    sectionA()
    sectionB()
    sectionC()
    print("\nwrote", RES)


if __name__ == "__main__":
    main()
