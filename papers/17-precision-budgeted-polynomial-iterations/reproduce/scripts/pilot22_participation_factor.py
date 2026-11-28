#!/usr/bin/env python3
"""
Pilot 22 - what the participation factor 1.46 is, and whether the spectrum closes it.

Pilot 20 left the response factor 4.13 split as 2 sqrt(2) (derived: the Hilbert-Schmidt
norm of the first-order operator on an exchangeable perturbation, Wigner 2 x skew
sqrt(2)) times 1.46 (measured: "the participation of the real profile").  Two things
were not settled: what 1.46 is made of, and why it appeared to grow with d.

 [A] Is the one-step profile factor a function of the spectrum alone, and does it move
     with d at fixed (family, ell)?  For a state with spectrum s we measure
        g(s) = || (F - F^T) o D ||_2 / (2 sqrt(2) * ||D||_F / sqrt(d)),
     D_ij = 1/(s_i + s_j), F iid Gaussian with unit entry rms (the convention of pilot
     20 A3, where a flat spectrum returns 2 sqrt(2)).  The pilot-20 reading of a d-drift
     compared cells of different ell, so the same (family, ell) is followed across d.

 [B] Candidate closures of g, each computable from the spectrum and with no fitted
     constant:
       (b1) exchangeable: g = 1 (the 2 sqrt(2) prediction itself).
       (b2) participation ratio of the weight square: g = sqrt(d / PR),
            PR = (sum D^2)^2 / sum D^4.
       (b3) coarse-grained variance-profile operator.  For a Wigner matrix with variance
            profile V the edge is not 2 sqrt(d * rms(V)); on a partition into K blocks it
            is governed by the K x K operator B_kl = mean(V_kl) sqrt(m_k m_l), edge =
            2 sqrt(lambda_max B).  It returns 2 sqrt(2) rms exactly for a flat profile,
            gives the exchangeable value at K = 1 and the corner-localised one at K = d.
       (b4) sqrt(1 + cv^2) of the row sums of V.
     A candidate closes g if its ratio to the measurement is flat across families and d.

 [C] The rest of 1.46 after g: the loop.  The kernel injects 1 cast and 2 errors per
     step, each at a state with its own spectrum.  Every injection's first-order
     coefficient is measured in its own units
        kappa_inj = response / (rms(E_inj) * scale_t),     scale_t = ||D_t||_F / sqrt(d)
     and compared with 2 sqrt(2) g(s_t) for the spectrum the state has at that moment;
     the quadrature sum is checked against the measured floor.  This decides whether the
     1 in 1.46 needs any constant beyond g.

Kernel model as in pilots 1-21: state and intermediates stored in p significant bits,
products accumulated in float64, exact matrix products in between.  Writes
pilot22_results.json.
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
from pilot7_pert_aware_design import pe_coeffs  # noqa: E402
from pilot17_theory_refinement import kernel_record, propagate, skew_resp  # noqa: E402

RES = os.path.join(HERE, "pilot22_results.json")
R = {}
K_DEF, CUSH, SAFE = 6, 0.02, 1.01
SQ2 = float(np.sqrt(2.0))
TWO_SQ2 = 2.0 * SQ2


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


# ---------------------------------------------------------------- the profile
def weights(sv):
    s = np.asarray(sv, float)
    s = s / s.max()
    D = 1.0 / (s[:, None] + s[None, :])
    nWf = float(np.linalg.norm(D, "fro"))
    return s, D, nWf, nWf / np.sqrt(s.size)


def gauss_factor(D, scale, seeds=(0, 1, 2, 3, 4, 5)):
    d = D.shape[0]
    vals = []
    for sd in seeds:
        rs = np.random.RandomState(1000 + sd)
        F = rs.randn(d, d)
        F /= np.sqrt((F ** 2).mean())
        vals.append(float(np.linalg.norm((F - F.T) * D, 2)) / (TWO_SQ2 * scale))
    return med(vals), span(vals)[0], span(vals)[1]


def gauss_factor_fast(D, scale, seeds=(0, 1, 2)):
    """same, with an explicit eigendecomposition of the Hadamard product."""
    d = D.shape[0]
    vals = []
    for sd in seeds:
        rs = np.random.RandomState(1000 + sd)
        A = rs.randn(d, d)
        A -= A.T
        A *= D
        vals.append(float(np.linalg.norm(A, 2)) / (TWO_SQ2 * scale))
    return med(vals)


# ---------------------------------------------------------------- candidate closures
def pr_gauss(D):
    d = D.shape[0]
    D2 = D ** 2
    pr = float(D2.sum() ** 2 / (D2 ** 2).sum())
    return float(np.sqrt(d / pr)), pr


def block_operator(D, K, V=None, scale=None):
    """g from the K-block variance-profile Wigner edge (K blocks of equal index count)."""
    d = D.shape[0]
    if V is None:
        V = 2.0 * D ** 2
    if scale is None:
        scale = float(np.linalg.norm(D, "fro")) / np.sqrt(d)
    k = int(min(K, d))
    edges = np.linspace(0, d, k + 1).astype(int)
    m = np.diff(edges).astype(float)
    P = np.zeros((d, k))
    for a in range(k):
        P[edges[a]:edges[a + 1], a] = 1.0
    S = P.T @ V @ P                       # block sums of the variance profile
    B = S / np.sqrt(np.outer(m, m))
    lam = float(np.linalg.eigvalsh(B)[-1])
    return float(2.0 * np.sqrt(max(lam, 0.0)) / (TWO_SQ2 * scale)), lam


def rowcv_gauss(D):
    V = 2.0 * D ** 2
    r = V.sum(1)
    return float(np.sqrt(1.0 + (r.std() / r.mean()) ** 2))


# ---------------------------------------------------------------- the cell pass
def cell_values(kind, L, d, seed, seeds_g=(0, 1, 2), full=False):
    M, ref, sv = spectra(kind, d, L, seed)
    s, D, nWf, scale = weights(sv)
    g = gauss_factor_fast(D, scale, seeds_g)
    out = dict(d=d, spectrum=kind, ell=L, seed=seed, g=g, nWf=nWf, scale=scale)
    if full:
        V = 2.0 * D ** 2
        out["participation"], out["PR"] = pr_gauss(D)
        out["block1"], _ = block_operator(D, 1, V, scale)
        for K in (2, 4, 8, 16, 32):
            out[f"block{K}"], _ = block_operator(D, K, V, scale)
        if d <= 256:
            out["block_full"], out["lam_full"] = block_operator(D, d, V, scale)
        out["rowcv"] = rowcv_gauss(D)
    return out


def sectionAB():
    print("=" * 118)
    print("[A] the one-step profile factor g(s) = ||(F-F^T) o D||_2 / (2sqrt2 ||D||_F/sqrt d)")
    print("    g = 1 is the exchangeable 2sqrt2;  d trend at fixed (family, ell) is the question")
    print("=" * 118)
    cells = []
    plan = [(64, 128, 256, 512, 1024), (1e-1, 1e-2, 1e-3), (0, 1, 2)]
    for kind in ("logspace", "plateau", "gap", "power"):
        for d in plan[0]:
            for L in plan[1]:
                for seed in plan[2]:
                    cells.append(cell_values(kind, L, d, seed))
        p18._CACHE.clear()
        gc.collect()
        print(f"  {kind}: {len(cells)} cells")
    R["a_cells"] = cells
    print(f"\n  {len(cells)} cells;  g by (family, ell) across d")
    print(f"  {'family':>9} {'ell':>6} " + " ".join(f"{'d=%d' % d:>7}" for d in plan[0]))
    for kind in ("logspace", "plateau", "gap", "power"):
        for L in plan[1]:
            row = [med([c["g"] for c in cells if c["spectrum"] == kind and c["ell"] == L
                        and c["d"] == d]) for d in plan[0]]
            print(f"  {kind:>9} {L:6g} " + " ".join(f"{v:7.3f}" for v in row))
    gs = np.array([c["g"] for c in cells])
    print(f"\n  all cells: median {med(gs):.3f}  [{gs.min():.3f}, {gs.max():.3f}]")
    trend, ratio = {}, []
    print("\n  ratio to d=64 at fixed (family, ell):")
    for kind in ("logspace", "plateau", "gap", "power"):
        for L in plan[1]:
            v = [med([c["g"] for c in cells if c["spectrum"] == kind and c["ell"] == L
                      and c["d"] == d]) for d in plan[0]]
            trend[f"{kind}_L{L:g}"] = [float(x / v[0]) for x in v]
            ratio.append(trend[f"{kind}_L{L:g}"][-1])
            print(f"  {kind:>9} {L:6g}: " + " ".join(f"{x / v[0]:7.3f}" for x in v))
    print(f"\n  d=1024 / d=64 over the twelve pairs: {span(ratio)[0]:.3f} .. {span(ratio)[1]:.3f}"
          f" (median {med(ratio):.3f})")
    R["a_trend"] = trend
    R["a_summary"] = dict(median_g=med(gs), g_lo=float(gs.min()), g_hi=float(gs.max()),
                          n=int(gs.size), trend_lo=span(ratio)[0], trend_hi=span(ratio)[1],
                          trend_median=med(ratio))
    save()

    # ---------------- [B] candidates on the same cells (d <= 256 for the full operator)
    print("\n" + "=" * 118)
    print("[B] candidate closures of g from the spectrum, no fitted constant")
    print("=" * 118)
    rows = []
    for kind in ("logspace", "plateau", "gap", "power"):
        for d in (64, 128, 256, 512, 1024):
            for L in (1e-1, 1e-2, 1e-3):
                for seed in (0, 1):
                    rows.append(cell_values(kind, L, d, seed, seeds_g=(0, 1, 2), full=True))
        p18._CACHE.clear()
        gc.collect()
    R["b_cells"] = rows
    cands = ["participation", "block1", "block2", "block4", "block8", "block16",
             "block32", "block_full", "rowcv"]
    print(f"  {len(rows)} cells;  candidate / measured  (block_full only for d <= 256)")
    print(f"  {'candidate':>14} {'median':>8} {'span':>16} {'d=64':>7} {'d=1024':>7}"
          f" {'logsp3 d64':>12} {'logsp3 d1024':>13}")
    summ = {"exchangeable": dict(median=1.0, lo=1.0, hi=1.0, spread=1.0, n=len(rows),
                                 frac10=1.0, frac20=1.0)}
    for k in cands:
        r = [b[k] / b["g"] for b in rows if k in b]
        summ[k] = dict(
            n=len(r), median=med(r), lo=span(r)[0], hi=span(r)[1],
            spread=span(r)[1] / span(r)[0],
            d64=med([b[k] / b["g"] for b in rows if k in b and b["d"] == 64]),
            d1024=med([b[k] / b["g"] for b in rows if k in b and b["d"] == 1024]),
            logsp3_d64=med([b[k] / b["g"] for b in rows if k in b
                            and b["spectrum"] == "logspace" and b["ell"] == 1e-3
                            and b["d"] == 64]),
            logsp3_d1024=med([b[k] / b["g"] for b in rows if k in b
                              and b["spectrum"] == "logspace" and b["ell"] == 1e-3
                              and b["d"] == 1024]),
            frac10=float(np.mean([abs(x - 1) <= 0.10 for x in r])),
            frac20=float(np.mean([abs(x - 1) <= 0.20 for x in r])))
        sv_ = summ[k]

        def fx(v, w=7):
            return f"{v:{w}.3f}" if v is not None else " " * (w - 3) + "n/a"
        print(f"  {k:>14} {sv_['median']:8.3f} {sv_['lo']:7.3f}-{sv_['hi']:8.3f} "
              f"{fx(sv_['d64'])} {fx(sv_['d1024'])} {fx(sv_['logsp3_d64'], 12)} "
              f"{fx(sv_['logsp3_d1024'], 13)}   +/-10% {sv_['frac10'] * 100:3.0f}%"
              f"  +/-20% {sv_['frac20'] * 100:3.0f}%")
    R["b_summary"] = summ
    print(f"\n  a closure needs candidate/measured = 1 with a flat span: "
          f"the best span here is {min(v['spread'] for v in summ.values()):.2f}x")
    save()


# ================================================================== [C]
def sectionC(dims=(64, 96)):
    print("\n" + "=" * 118)
    print("[C] the loop half of 1.46: per-injection coefficients in their own units")
    print("    kappa_inj = response / (rms(E_inj) scale_t),  against 2sqrt2 g(s_t)")
    print("=" * 118)
    out = {}
    for d in dims:
        for kind in ("logspace", "plateau", "gap", "power"):
            for L in (1e-1, 1e-2):
                p = 8
                M, ref, sv = spectra(kind, d, L, 0)
                co, win = pe_coeffs(L, K_DEF, CUSH, SAFE)
                X0 = M / n2(M)
                s0 = sv / sv.max()
                X, Y, inj = kernel_record(co, X0, p)
                e_meas = sub_err(X, ref)
                s = s0.copy()
                s_orbit = [s.copy()]
                for (a, b, c) in co:
                    s = a * s + b * s ** 3 + c * s ** 5
                    s_orbit.append(s.copy())
                prof = [weights(s_t) for s_t in s_orbit]
                scale0 = prof[0][3]
                u = 2.0 ** -p
                rec, quad = [], 0.0
                for k_inj, at, E in inj:
                    dX = propagate(co, Y, E if k_inj != "B" else E @ Y[at - 1], at)
                    resp = skew_resp(dX, ref)
                    rho = rms(E)
                    scale_t = prof[at][3]
                    kappa = resp / (rho * scale_t) if rho > 0 and scale_t > 0 else None
                    g_t = gauss_factor_fast(prof[at][1], scale_t, (0, 1, 2))
                    rec.append(dict(kind=k_inj, at=at, resp=resp, rho=rho, scale_t=scale_t,
                                    kappa=kappa, g_state=g_t, pred=TWO_SQ2 * g_t,
                                    kappa_over_pred=(kappa / (TWO_SQ2 * g_t)) if kappa else None))
                    quad += resp ** 2
                quad = float(np.sqrt(quad))
                C_tot = e_meas / (u * rms(X0) * scale0)
                C_quad = quad / (u * rms(X0) * scale0)
                cast = [r for r in rec if r["kind"] == "cast"][0]
                loop = [r for r in rec if r["kind"] != "cast"]
                C_cast = cast["resp"] / (u * rms(X0) * scale0)
                C_loop = float(np.sqrt(sum(r["resp"] ** 2 for r in loop))) / (u * rms(X0) * scale0)
                C_pred = float(np.sqrt(sum((TWO_SQ2 * r["g_state"] * r["rho"] * r["scale_t"]) ** 2
                                           for r in rec))) / (u * rms(X0) * scale0)
                key = f"{kind}_d{d}_L{L:g}"
                out[key] = dict(d=d, spectrum=kind, ell=L, p=p, C_total=C_tot, C_cast=C_cast,
                                C_loop=C_loop, C_quad=C_quad, C_pred=C_pred, measured=e_meas,
                                quadrature=quad,
                                quadrature_over_measured=quad / e_meas if e_meas else None,
                                cast_share=C_cast / C_tot, g0=gauss_factor_fast(prof[0][1], scale0, (0, 1, 2)),
                                injections=rec)
                print(f"  {kind:>9} d={d:4d} L={L:6g}: C_tot {C_tot:6.3f} = cast {C_cast:6.3f}"
                      f" + loop {C_loop:5.3f} (quad {C_quad:6.3f});  law prediction {C_pred:6.3f}"
                      f" = {C_pred / C_tot:5.3f}x;  quad/meas {quad / e_meas:5.3f}")
        p18._CACHE.clear()
        gc.collect()
    R["c_cells"] = out
    ov = [r["kappa_over_pred"] for v in out.values() for r in v["injections"]
          if r["kappa_over_pred"]]
    rp = [v["C_pred"] / v["C_total"] for v in out.values() if v["C_total"]]
    qm = [v["quadrature_over_measured"] for v in out.values() if v["quadrature_over_measured"]]
    print(f"\n  per-injection kappa / (2sqrt2 g(state)): {span(ov)[0]:.3f} .. {span(ov)[1]:.3f}"
          f" (median {med(ov):.3f}) over {len(ov)} injections")
    print(f"  predicted total / measured total: {span(rp)[0]:.3f} .. {span(rp)[1]:.3f}"
          f" (median {med(rp):.3f})")
    print(f"  quadrature / measured: {span(qm)[0]:.3f} .. {span(qm)[1]:.3f} (median {med(qm):.3f})")
    R["c_summary"] = dict(n_inj=len(ov), kappa_over_pred_median=med(ov),
                          kappa_over_pred_lo=span(ov)[0], kappa_over_pred_hi=span(ov)[1],
                          C_pred_over_total_median=med(rp), C_pred_over_total_lo=span(rp)[0],
                          C_pred_over_total_hi=span(rp)[1],
                          quad_over_measured_median=med(qm),
                          quad_over_measured_lo=span(qm)[0],
                          quad_over_measured_hi=span(qm)[1])
    save()


# ================================================================== [D]
def sectionD():
    print("\n" + "=" * 118)
    print("[D] assembly: 4.13 = 2sqrt2 x g x (injection quadrature)?")
    print("=" * 118)
    cells = R["c_cells"]
    tot = [v["C_total"] for v in cells.values()]
    cast = med([v["C_cast"] / v["C_total"] for v in cells.values()])
    g_in = R["a_summary"]["median_g"]
    wg = []
    for v in cells.values():
        num = sum((r["rho"] * r["scale_t"] * r["g_state"]) ** 2 for r in v["injections"])
        den = sum((r["rho"] * r["scale_t"]) ** 2 for r in v["injections"])
        wg.append(float(np.sqrt(num / den)))
    print(f"  measured C_total in u rms(X0) scale units: median {med(tot):.3f}"
          f" [{span(tot)[0]:.3f}, {span(tot)[1]:.3f}] over {len(tot)} cells")
    print(f"  C_total / 2sqrt2 = {med(tot) / TWO_SQ2:.3f}   (the 4.13 of pilot 19-20)")
    print(f"  cast share of the total: median {cast:.3f}")
    print(f"  injection-weighted g: median {med(wg):.3f} [{span(wg)[0]:.3f}, {span(wg)[1]:.3f}]")
    print(f"  one-step g on the input spectrum (180 cells): {g_in:.3f}")
    print(f"  ratio (injection-weighted g)/(one-step g) = {med(wg) / g_in:.3f}")
    R["d_assembly"] = dict(C_total_median=med(tot), C_total_lo=span(tot)[0],
                           C_total_hi=span(tot)[1], C_total_over_2sqrt2=med(tot) / TWO_SQ2,
                           cast_share_median=cast, g_weighted_median=med(wg),
                           g_weighted_lo=span(wg)[0], g_weighted_hi=span(wg)[1],
                           g_input_median=g_in, weighted_over_input=med(wg) / g_in)
    save()


def main():
    sectionAB()
    sectionC()
    sectionD()
    print("\nwrote", RES)


if __name__ == "__main__":
    main()
