#!/usr/bin/env python3
"""
Pilot 20 - the analytic content of the floor constant, its value at d = 1024, and what a
practitioner should change.

Pilot 19 split the 240-cell constant 6.073 into a derived mantissa factor
sqrt(E[m^-2]) = sqrt(3/(2 ln 2)) = 1.471 and a residual response factor 4.13, and split
4.13 again into an input-cast channel 3.77 and an in-loop channel 2.33 (quadrature 4.43).
This script asks how much of the 4.13 is derived.

 [A] The one-step operator of the first-order formula.  For  X = U diag(s) V^T  and a
     perturbation E, theorem 3's first term is  U[(F - F^T) o D] V^T  with F = U^T E V and
     D_ij = 1/(s_i + s_j).  Three consequences, each measured separately:
       (A1) on the matrix units the operator is exactly T(e_ij) = D_ij (e_ij - e_ji), so
            its Hilbert-Schmidt norm is exactly sqrt(2) ||W||_F, W = D.  Verified against
            finite differences of the polar factor over all d^2 units of a d = 32 cell.
       (A2) if the perturbation's d^2 entries are exchangeable, ||A||_2 = 2 sqrt(d) rms(A)
            gives the response 2 sqrt(2) x (rho ||W||_F / sqrt d): a factor 2 from the
            Wigner law for a d x d incoherent matrix and a factor sqrt(2) from the skew
            pair.  On a flat spectrum (D = 1/2 exactly) the prediction is exact in the
            limit, and the measured ratio approaches 2 sqrt(2) with d.
       (A3) on a real spectrum D is not flat (1/(2 l) at the bottom against 1/2 at the
            top), so the exchangeability step is where the derivation stops.  The
            measured one-step constant over 4 families x 2 ell x 4 d x 4 seeds, and two
            candidate spectral functionals of the profile, one of which fails by an order
            of magnitude.
       (A4) the loop is neutral to the cast channel: the one-step formula and the
            full K-step linearised propagation of the same cast error agree, so the
            residual of (A3) is not loop amplification.
       (A5) assembly of 4.13 from the derived and the measured part.

 [B] d = 1024 spot extrapolation: the same cells as pilot 19 at d = 768 and 1024, the
     band, the one-sided constant, and the trend fitted on d <= 512.

 [C] The practitioner's three rules, priced on the 100 real Muon momenta of pilot 11 and
     on a 12-cell dynamic-range ladder: bits per element-slot, bytes per update for a
     square layer, and the floor each rule lands on.

Kernel model as in pilots 1-19: state and intermediates stored in p significant bits,
products accumulated in float64, exact matrix products in between.  Writes
pilot20_results.json.
"""
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pilot18_c_scaling_and_e5m2_fix import cell as p18_cell, spectra, n2, kernel_steps, sub_err
from pilot17_theory_refinement import kernel_record, propagate, skew_resp
from pilot7_pert_aware_design import q, pe_coeffs, stats

R = {}
K_DEF, CUSH, SAFE = 6, 0.02, 1.01
C18 = 6.073
L_MANT = np.sqrt(3.0 / (2.0 * np.log(2.0)))       # 1.4715
C_TOT = C18 / L_MANT                              # 4.128, the constant in the kernel's rms


def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else None


def span(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return (float(min(xs)), float(max(xs))) if xs else (None, None)


def fit_slope(xs, ys):
    x = np.log(np.asarray(xs, float))
    y = np.log(np.asarray(ys, float))
    keep = np.isfinite(y) & (y > 0)
    if keep.sum() < 2:
        return None
    A = np.vstack([x[keep], np.ones(int(keep.sum()))]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][0])


def polar(X):
    U, _, Vt = np.linalg.svd(X, full_matrices=False)
    return U @ Vt


def pair_weight(sv):
    s = sv / sv.max()
    return s, 1.0 / (s[:, None] + s[None, :]), float(np.sqrt((s ** 2).sum())), \
        float(np.linalg.norm(1.0 / (s[:, None] + s[None, :]), "fro"))


# ================================================================== [A] the constant
def sectionA():
    print("=" * 112)
    print("[A1] the one-step operator on matrix units: T(e_ij) = D_ij (e_ij - e_ji);")
    print("     Hilbert-Schmidt norm sqrt(2) ||W||_F;  checked against finite differences")
    print("=" * 112)
    d, L, kind = 32, 1e-1, "logspace"
    M, ref, sv = spectra(kind, d, L, 0)
    s, D, fro_s, nWf = pair_weight(sv)
    eps = 1e-6
    Q0 = polar(M)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    hs, fd2, fdF, ratios, diag_max = 0.0, 0.0, 0.0, [], 0.0
    I = np.eye(d)
    for i in range(d):
        for j in range(d):
            E = U @ np.outer(I[i], I[j]) @ Vt          # a unit in the (U,V) frame: F = e_ij
            if i == j:                       # a diagonal unit does not rotate the factor
                diag_max = max(diag_max, float(np.linalg.norm(polar(M + eps * E) - Q0, 2)) / eps)
                continue
            dQ = (polar(M + eps * E) - Q0) / eps
            pred = U @ (D[i, j] * (np.outer(I[i], I[j]) - np.outer(I[j], I[i]))) @ Vt
            n_pred = float(np.linalg.norm(pred, 2))
            fd2 += float(np.linalg.norm(dQ - pred, 2)) ** 2
            fdF += float(np.linalg.norm(dQ, "fro")) ** 2
            hs += float(np.linalg.norm(pred, "fro")) ** 2
            ratios.append(float(np.linalg.norm(dQ, 2)) / n_pred)
    rel = float(np.sqrt(fd2 / fdF))
    hs_exact = 2 * (nWf ** 2 - float((np.diag(D) ** 2).sum()))
    print(f"  d={d}, logspace ell={L:g}: sum_(i!=j) ||dQ_fd||_F^2 = {hs:.6f}  against the"
          f" operator's exact 2(||W||_F^2 - sum_i D_ii^2) = {hs_exact:.6f}"
          f"  (ratio {hs / hs_exact:.6f});  the d diagonal units give {diag_max:.1e}")
    print(f"  finite difference (eps={eps:g}) against the formula: relative residual "
          f"{rel:.2e}, mean ratio of the 2-norms {med(ratios):.4f} "
          f"[{span(ratios)[0]:.4f}, {span(ratios)[1]:.4f}]")
    R["a1_operator"] = dict(d=d, hs_measured=hs, hs_exact=hs_exact,
                            hs_exact_over_2Ws=hs_exact / (2 * nWf ** 2),
                            hs_ratio=hs / hs_exact, fd_residual=rel,
                            fd_ratio_median=med(ratios), diag_unit=diag_max)

    print("\n" + "=" * 112)
    print("[A2] flat spectrum (sigma = 1, D = 1/2): the incoherent response is 2 sqrt(2) x")
    print("     rho ||W||_F / sqrt(d);  the two factors are the Wigner 2 and the skew sqrt(2)")
    print("=" * 112)
    print(f"  {'d':>5} {'||skew(G)||_2/(sqrt d rms)':>27} {'response / (rho ||W||_F/sqrt d)':>32}"
          f" {'x 2sqrt2':>9}")
    flat = {}
    rs = np.random.RandomState(11)
    for d in (64, 128, 256, 512, 1024):
        U, _ = np.linalg.qr(rs.randn(d, d))
        V, _ = np.linalg.qr(rs.randn(d, d))
        Q = U @ V.T
        W = np.ones((d, d)) / 2.0
        scale = np.linalg.norm(W, "fro") / np.sqrt(d)          # = sqrt(d)/2
        wf, rr = [], []
        for t in range(3):
            F = rs.randn(d, d)
            F /= np.sqrt((F ** 2).mean())                       # entry rms 1
            sk = (F - F.T) / 2.0
            wf.append(float(np.linalg.norm(sk, 2)) / (np.sqrt(d) * np.sqrt((sk ** 2).mean())))
            rr.append(float(np.linalg.norm(sk, 2)) / scale)
        # the same with the actual cast rounding of Q at p = 8
        E = q(Q, 8) - Q
        rho = float(np.sqrt((E ** 2).mean()))
        rnd = float(np.linalg.norm((E - E.T) / 2.0, 2)) / (rho * scale)
        flat[d] = dict(wigner=med(wf), ratio=med(rr), ratio_per_2sqrt2=med(rr) / (2 * np.sqrt(2)),
                       rounding=rnd, rho=rho, scale=float(scale))
        print(f"  {d:5d} {med(wf):27.3f} {med(rr):32.3f} {med(rr) / (2 * np.sqrt(2)):9.3f}"
              f"   (rounding {rnd:.3f})")
    R["a2_flat"] = flat
    print("  the Wigner factor converges to 2 and the response to 2 sqrt(2) = 2.828; the two")
    print("  are the same statement, D = 1/2 making the skew pair carry sqrt(2) per unit.")

    print("\n" + "=" * 112)
    print("[A3] a real spectrum is not flat: the measured one-step constant over")
    print("     4 families x 2 ell x 4 d x 4 seeds, in units of rho ||W||_F / sqrt d")
    print("=" * 112)
    rows, cells = [], {}
    for kind in ("logspace", "plateau", "gap", "power"):
        for L in (1e-1, 1e-2):
            for d in (64, 128, 256, 512):
                M, ref, sv = spectra(kind, d, L, 0)
                s, D, fro_s, nWf = pair_weight(sv)
                scale = nWf / np.sqrt(d)
                rs = np.random.RandomState(77)
                vals = []
                for t in range(4):
                    F = rs.randn(d, d)
                    F /= np.sqrt((F ** 2).mean())
                    vals.append(float(np.linalg.norm((F - F.T) * D, 2)) / scale)
                cells[f"{kind}_L{L:g}_d{d}"] = dict(median=med(vals), lo=span(vals)[0],
                                                    hi=span(vals)[1],
                                                    over_2sqrt2=med(vals) / (2 * np.sqrt(2)))
                rows.append((kind, L, d, med(vals)))
    for kind in ("logspace", "plateau", "gap", "power"):
        for L in (1e-1, 1e-2):
            row = [f"{c / (2 * np.sqrt(2)):5.2f}" for (k, l, d, c) in rows if k == kind and l == L]
            print(f"  {kind:>9} ell={L:6g}: " + " ".join(row) + "   (x 2sqrt2, d=64,128,256,512)")
    allc = [c for _, _, _, c in rows]
    print(f"\n  one-step constant over the 32 cells: median {med(allc):.3f} "
          f"[{span(allc)[0]:.3f}, {span(allc)[1]:.3f}], "
          f"= {med(allc) / (2 * np.sqrt(2)):.3f} x 2 sqrt(2), "
          f"+/-25% of 2.83: {np.mean(np.abs(np.array(allc) / (2 * np.sqrt(2)) - 1) <= 0.25) * 100:.0f}%")
    fam = {k: med([c for (kk, _, _, c) in rows if kk == k]) for k in
           ("logspace", "plateau", "gap", "power")}
    per_d = {d: med([c for (_, _, dd, c) in rows if dd == d]) for d in (64, 128, 256, 512)}
    print("  family medians: " + " ".join(f"{k} {v:.2f}" for k, v in fam.items()))
    print("  d medians:      " + " ".join(f"{d} {v:.2f}" for d, v in per_d.items()))

    print("\n  candidate spectral functionals of the profile 2 D^2, as predictors of the")
    print("  same constant (a positive one would close the derivation):")
    print(f"  {'cell':>22} {'measured':>9} {'2sqrt(d mean)':>14} {'2sqrt(d lmax)':>14} "
          f"{'2sqrt(d)maxrow':>15}")
    cand = {}
    for key in ("logspace_L0.1_d128", "logspace_L0.01_d256", "plateau_L0.01_d256",
                "power_L0.1_d256", "gap_L0.01_d512"):
        kind, Ls, ds = key.rsplit("_", 2)
        L = float(Ls[1:])
        d = int(ds[1:])
        M, ref, sv = spectra(kind, d, L, 0)
        s, D, fro_s, nWf = pair_weight(sv)
        G2 = 2 * D ** 2
        lmax = float(np.linalg.eigvalsh(G2)[-1])
        mean = float(G2.mean())
        rowmax = max(float(np.sqrt((G2[i] ** 2).sum())) for i in range(d))     # row degree
        scale = nWf / np.sqrt(d)
        p_mean = 2 * np.sqrt(d * mean) / scale
        p_lmax = 2 * np.sqrt(d * lmax) / scale
        p_row = 2 * np.sqrt(d) * rowmax / scale
        cand[key] = dict(measured=med([cells[key]["median"]]), p_mean=p_mean, p_lmax=p_lmax,
                         p_row=p_row)
        print(f"  {key:>22} {cand[key]['measured']:9.3f} {p_mean:14.3f} {p_lmax:14.3f} "
              f"{p_row:15.3f}")
    lr = [v["p_lmax"] / v["measured"] for v in cand.values()]
    rr = [v["p_row"] / v["measured"] for v in cand.values()]
    print(f"\n  lmax functional: {min(lr):.1f}-{max(lr):.1f}x the measurement (fails);")
    print(f"  largest-row functional: {min(rr):.1f}-{max(rr):.1f}x (fails).")
    print("  so the profile's participation is not any single spectral functional of W")
    print("  tried here: the 2-norm of an incoherent matrix sits between the Frobenius")
    print("  average (2 sqrt(2)) and the corner scale, and only the measurement locates it.")
    R["a3_profile"] = dict(cells=cells, median=med(allc), lo=span(allc)[0], hi=span(allc)[1],
                           over_2sqrt2=med(allc) / (2 * np.sqrt(2)),
                           frac_25_of_2sqrt2=float(np.mean(np.abs(np.array(allc)
                                                                   / (2 * np.sqrt(2)) - 1) <= 0.25)),
                           family=fam, per_d=per_d, candidates=cand,
                           lmax_over_measured=[min(lr), max(lr)],
                           row_over_measured=[min(rr), max(rr)])

    print("\n" + "=" * 112)
    print("[A4] the loop is neutral to the cast channel: the one-step formula against the")
    print("     full K-step linearised propagation of the same cast error")
    print("=" * 112)
    print(f"  {'family':>9} {'ell':>6} {'d':>4} {'formula':>9} {'propagated':>11} {'ratio':>7}")
    neu = []
    for kind in ("logspace", "plateau"):
        for L in (1e-1, 1e-2):
            for d in (128, 256):
                M, ref, sv = spectra(kind, d, L, 0)
                co, _ = pe_coeffs(L, K_DEF, CUSH, SAFE)
                X0 = M / n2(M)
                X, Y, inj = kernel_record(co, X0, 8)
                s, D, fro_s, nWf = pair_weight(sv)
                scale = nWf / np.sqrt(d)
                U, _, Vt = np.linalg.svd(X0, full_matrices=False)
                rs = np.random.RandomState(31)
                fo, pr = [], []
                for t in range(3):
                    G = rs.randn(d, d)
                    G /= np.sqrt((G ** 2).mean())
                    F = U.T @ G @ Vt
                    fo.append(float(np.linalg.norm((F - F.T) * D, 2)) / scale)
                    pr.append(skew_resp(propagate(co, Y, G, 0), ref) / scale)
                neu.append(dict(kind=kind, ell=L, d=d, formula=med(fo), propagated=med(pr),
                                ratio=med(pr) / med(fo)))
                print(f"  {kind:>9} {L:6g} {d:4d} {med(fo):9.3f} {med(pr):11.3f} "
                      f"{med(pr) / med(fo):7.3f}")
    rt = [n["ratio"] for n in neu]
    print(f"\n  propagated / formula: {span(rt)[0]:.3f} .. {span(rt)[1]:.3f} "
          f"(median {med(rt):.3f}) - the design maps the window onto the window, so the")
    print("  cast channel's response is a one-step object; what grows with K is the number")
    print("  of in-loop injections, not the amplification of any one of them.")
    R["a4_neutral"] = dict(cells=neu, ratio_median=med(rt), ratio_lo=span(rt)[0],
                           ratio_hi=span(rt)[1])

    print("\n" + "=" * 112)
    print("[A5] assembly of the 240-cell constant 4.13 in the kernel's own rms units")
    print("=" * 112)
    rp = R['a3_profile']
    quad = float(np.sqrt((2 * np.sqrt(2) * rp['over_2sqrt2']) ** 2 + 2.3306 ** 2))
    print(f"  derived: 2 sqrt(2)                                    = {2 * np.sqrt(2):6.3f}")
    print(f"  measured: profile participation at one step             = {rp['over_2sqrt2']:6.3f}")
    print(f"            -> one-step cast channel                     = "
          f"{2 * np.sqrt(2) * rp['over_2sqrt2']:6.3f}  (pilot 19, 16 cells: 3.771, "
          f"ratio {3.7715 / (2 * np.sqrt(2) * rp['over_2sqrt2']):.2f})")
    print(f"  measured (pilot 19, kernel rms): in-loop 2.331, quadrature 4.433, "
          f"first-order total 4.540")
    print(f"  cast one-step + in-loop in quadrature                   = {quad:6.3f}"
          f"  (vs first-order 4.540, {quad / 4.5396 - 1:+.1%})")
    print(f"  the 240-cell fit: 4.128 = 6.073 / {L_MANT:.4f}")
    print("  of 4.13: 2.83 is derived, 1.46 is the profile's participation (1.26 in the")
    print("  cast channel's one step, the rest from the twelve in-loop injections); the")
    print("  first-order bookkeeping sits 2-3% above the fit.  The 1.46 is what stays a")
    print("  measurement, and A3's two negatives say why no spectral functional of the")
    print("  profile replaces it.")
    R["a5_assembly"] = dict(derived_2sqrt2=float(2 * np.sqrt(2)),
                            participation=rp["over_2sqrt2"],
                            one_step_cast=float(2 * np.sqrt(2) * rp["over_2sqrt2"]),
                            pilot19_cast=3.7715, pilot19_loop=2.3306,
                            pilot19_quad=4.4335, pilot19_total=4.5396,
                            quadrature_cast_loop=quad, quadrature_over_total=quad / 4.5396 - 1,
                            c_total_kernel=C_TOT, mant=L_MANT)


# ================================================================== [B] d = 1024
def sectionB():
    print("\n" + "=" * 112)
    print("[B1] the constant at d = 768 and 1024 (pilot 19's cells, two seeds each)")
    print("=" * 112)
    cells = []
    dims = (768, 1024)
    for kind in ("logspace", "plateau"):
        for L in (1e-1, 1e-2):
            for p in (8, 12):
                for seed in (0, 1):
                    for d in dims:
                        cells.append(p18_cell(d, kind, L, p, seed))
    R["b1_cells"] = cells
    print(f"  {len(cells)} cells = 2 families x 2 ell x 2 p x 2 seeds x 2 dims")
    per_d, per_cell = {}, {}
    for d in dims:
        for kind in ("logspace", "plateau"):
            for L in (1e-1, 1e-2):
                for p in (8, 12):
                    v = [c["over_pred"]["Fd"] for c in cells
                         if c["d"] == d and c["spectrum"] == kind and c["ell"] == L
                         and c["p"] == p and "Fd" in c["over_pred"]]
                    per_cell[f"{kind}_L{L:g}_p{p}_d{d}"] = med(v)
    print(f"  {'family':>9} {'ell':>6} " + " ".join(f"{'p=%d d=%d' % (p, d):>12}"
                                                  for d in dims for p in (8, 12)))
    for kind in ("logspace", "plateau"):
        for L in (1e-1, 1e-2):
            vals = [per_cell[f"{kind}_L{L:g}_p{p}_d{d}"] for d in dims for p in (8, 12)]
            print(f"  {kind:>9} {L:6g} " + " ".join(f"{v:12.3f}" for v in vals))
    for d in dims:
        cs = np.array([c["over_pred"]["Fd"] for c in cells if c["d"] == d and "Fd" in c["over_pred"]])
        per_d[d] = dict(n=int(cs.size), median=float(np.median(cs)), lo=float(cs.min()),
                        hi=float(cs.max()), ratio_to_fit=float(np.median(cs) / C18),
                        frac_25=float(np.mean(np.abs(cs / C18 - 1) <= 0.25)),
                        frac_40=float(np.mean(np.abs(cs / C18 - 1) <= 0.40)),
                        covered_8p9=float(np.mean(cs <= 8.918)),
                        covered_9p3=float(np.mean(cs <= 9.3)),
                        covered_one_sided=float(np.mean(cs <= float(cs.max()))))
        print(f"  d={d}: median {per_d[d]['median']:6.3f} = {per_d[d]['ratio_to_fit']:.3f}x the "
              f"fit, band [{per_d[d]['lo']:.3f}, {per_d[d]['hi']:.3f}], "
              f"+-25% {per_d[d]['frac_25'] * 100:.0f}%, +-40% {per_d[d]['frac_40'] * 100:.0f}%, "
              f"<= 8.918 {per_d[d]['covered_8p9'] * 100:.0f}%, <= 9.3 "
              f"{per_d[d]['covered_9p3'] * 100:.0f}%")
    R["b1_per_d"] = per_d
    R["b1_per_cell"] = per_cell
    allcs = np.array([c["over_pred"]["Fd"] for c in cells if "Fd" in c["over_pred"]])
    lo, hi = float(allcs.min()), float(allcs.max())
    print(f"\n  all {allcs.size} cells at d in {{768,1024}}: median {np.median(allcs):.3f} "
          f"[{lo:.3f}, {hi:.3f}] = 2.20x band, one-sided {hi:.3f}")
    # the same two families pooled at d = 512, so the comparison has one cell mix
    H = load_pilot19()["a1_cells"]
    ref512 = np.array([c["over_pred"]["Fd"] for c in H
                       if c["d"] == 512 and c["spectrum"] in ("logspace", "plateau")
                       and "Fd" in c["over_pred"]])
    print(f"  the same two families pooled at d = 512 (pilot 19): median "
          f"{np.median(ref512):.3f} [{ref512.min():.3f}, {ref512.max():.3f}];  "
          f"768/1024 over 512 = {np.median(allcs) / np.median(ref512):.3f}x")
    fam1024 = {}
    for kind in ("logspace", "plateau"):
        for d in dims:
            v = [c["over_pred"]["Fd"] for c in cells if c["d"] == d
                 and c["spectrum"] == kind and "Fd" in c["over_pred"]]
            fam1024[f"{kind}_d{d}"] = med(v)
    print("  family medians: " + "  ".join(f"{k} {v:.3f}" for k, v in fam1024.items()))
    R["b1_all"] = dict(n=int(allcs.size), median=float(np.median(allcs)), lo=lo, hi=hi,
                       ratio_to_fit=float(np.median(allcs) / C18), span=hi / lo,
                       one_sided=hi, ref512_median=float(np.median(ref512)),
                       ratio_over_512=float(np.median(allcs) / np.median(ref512)),
                       family=fam1024,
                       frac_25=float(np.mean(np.abs(allcs / C18 - 1) <= 0.25)),
                       frac_40=float(np.mean(np.abs(allcs / C18 - 1) <= 0.40)))

    print("\n" + "=" * 112)
    print("[B2] does the trend fitted on d <= 512 predict d = 768, 1024?")
    print("=" * 112)
    print(f"  {'cell':>24} {'slope':>7} {'meas 768':>9} {'ratio':>6} {'meas 1024':>10} {'ratio':>6}")
    extr = {}
    for kind in ("logspace", "plateau"):
        for L in (1e-1, 1e-2):
            for p in (8, 12):
                fitd = (32, 64, 128, 256, 384, 512)
                pts = [per_cell_hist(kind, L, p, d) for d in fitd]
                sl = fit_slope(fitd, pts)
                anchor = pts[-1] * 512 ** (-sl) if sl is not None else None
                pred = {d: anchor * d ** sl for d in dims} if anchor else {}
                act = {d: per_cell[f"{kind}_L{L:g}_p{p}_d{d}"] for d in dims}
                rat = {d: act[d] / pred[d] for d in pred}
                extr[f"{kind}_L{L:g}_p{p}"] = dict(slope=sl, pred=pred, measured=act, ratio=rat)
                print(f"  {kind[:5]}-{L:g}-p{p:<2}: {sl:7.3f} " +
                      " ".join(f"{act[d]:9.3f} {rat[d]:6.3f}" for d in dims))
    ra = [v for e in extr.values() for v in e["ratio"].values()]
    R["b2_extrapolation"] = extr
    print(f"\n  measured / extrapolated over the 8 curves x 2 dims: {min(ra):.3f}-{max(ra):.3f} "
          f"(median {med(ra):.3f})")

    print("\n" + "=" * 112)
    print("[B3] the profile participation at d = 1024 (one-step, iid Gaussian, 4 seeds)")
    print("=" * 112)
    rs = np.random.RandomState(91)
    part = {}
    for kind in ("logspace", "plateau"):
        for L in (1e-1, 1e-2):
            M, ref, sv = spectra(kind, 1024, L, 0)
            s, D, fro_s, nWf = pair_weight(sv)
            scale = nWf / np.sqrt(1024)
            vals = []
            for t in range(4):
                F = rs.randn(1024, 1024)
                F /= np.sqrt((F ** 2).mean())
                vals.append(float(np.linalg.norm((F - F.T) * D, 2)) / scale)
            part[f"{kind}_L{L:g}"] = dict(median=med(vals), over_2sqrt2=med(vals) / (2 * np.sqrt(2)))
            print(f"  {kind:>9} ell={L:6g} d=1024: {med(vals):.3f} = "
                  f"{med(vals) / (2 * np.sqrt(2)):.3f} x 2sqrt2")
    R["b3_participation_1024"] = part


def per_cell_hist(kind, L, p, d):
    """the d <= 512 curve of pilot 19 (results JSON), so the fit uses the same cells."""
    H = R.get("_hist")
    if H is None:
        H = load_pilot19()["a1_cells"]
        R["_hist"] = H
    vals = [c["over_pred"]["Fd"] for c in H if c["spectrum"] == kind and c["ell"] == L
            and c["p"] == p and c["d"] == d and "Fd" in c["over_pred"]]
    return med(vals)


def load_pilot19():
    """pilot 19's JSON: next to this script when it has just run, otherwise the archived
    copy in ../results/.  B1 and B2 extend pilot 19's own grid, so they read its cells."""
    for cand in (os.path.join(HERE, "pilot19_results.json"),
                 os.path.join(HERE, "..", "results", "pilot19_results.json")):
        if os.path.exists(cand):
            print(f"  (pilot 19's grid read from {os.path.relpath(cand, HERE)})")
            with open(cand) as f:
                return json.load(f)
    raise FileNotFoundError("pilot19_results.json not found next to this script or in ../results")


# ================================================================== [C] the rules
RULES = [
    ("bf16", dict(p=8, safety=1.01)),
    ("E4M3", dict(p=5, safety=1.01)),
    ("E5M2 s1.01", dict(p=4, safety=1.01)),
    ("E5M2 s1.035", dict(p=4, safety=1.035)),
    ("cast8", dict(p=4, safety=1.01, p_cast=8)),
    ("cast8+1", dict(p=4, safety=1.01, p_cast=8, precs=[8, 4, 4, 4, 4, 4])),
    ("cast8+2", dict(p=4, safety=1.01, p_cast=8, precs=[8, 8, 4, 4, 4, 4])),
    ("E4M3 cast8+2", dict(p=5, safety=1.01, p_cast=8, precs=[8, 8, 5, 5, 5, 5])),
]


def rule_bits(rule, m, n, K=K_DEF):
    """element-slots and mean bits per slot for a state m x n: one cast of mn, and per
    step a state write (mn) and the intermediate B (m x m)."""
    p_cast = rule.get("p_cast", rule["p"])
    precs = rule.get("precs", [rule["p"]] * K)
    slots = float(m * n)
    bits = float(m * n) * p_cast
    for t in range(K):
        slots += m * n + m * m
        bits += (m * n + m * m) * precs[t]
    return slots, bits / slots


def run_rule(co, X0, ref, rule):
    p_cast = rule.get("p_cast", rule["p"])
    precs = rule.get("precs", [rule["p"]] * K_DEF)
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            X = kernel_steps(co, X0, p_cast, precs)
        st = n2(X - ref)
        if not np.isfinite(st):
            return float("inf"), float("inf")
        sub = sub_err(X, ref)
        return st, (sub if np.isfinite(sub) else float("inf"))
    except Exception:
        return float("inf"), float("inf")


def sectionC():
    print("\n" + "=" * 112)
    print("[C0] bit accounting: element-slots of one Muon update of an m x n state")
    print("     (cast mn, then per step a state write mn and the intermediate B of m x m)")
    print("=" * 112)
    print(f"  {'rule':>14} {'mean bits 128x32':>17} {'mean bits 4096^2':>17} "
          f"{'MB per update 4096^2':>21}")
    cost = {}
    for name, rule in RULES:
        s1, b1 = rule_bits(rule, 128, 32)
        s2, b2 = rule_bits(rule, 4096, 4096)
        cost[name] = dict(bits_rect=b1, bits_sq=b2, slots_sq=s2, mb_sq=s2 * b2 / 8 / 1e6)
        print(f"  {name:>14} {b1:17.2f} {b2:17.2f} {s2 * b2 / 8 / 1e6:21.1f}")
    R["c0_cost"] = cost
    mb8 = cost["bf16"]["mb_sq"]
    print(f"  bf16 through one 4096^2 update: {mb8:.0f} MB of stores; E4M3 "
          f"{cost['E4M3']['mb_sq']:.0f} MB ({cost['E4M3']['mb_sq'] / mb8 - 1:+.0%}), "
          f"cast8+2 {cost['cast8+2']['mb_sq']:.0f} MB ({cost['cast8+2']['mb_sq'] / mb8 - 1:+.0%}).")
    p_hyb, p_base = cost["cast8+2"]["bits_sq"], cost["E4M3"]["bits_sq"]
    fleet = {}
    for f in (0.0, 0.1, 0.25, 0.5, 1.0):
        b = (1 - f) * p_base + f * p_hyb
        fleet[f] = b
        print(f"  a fleet with {f * 100:3.0f}% ill-conditioned layers (hybrid there, E4M3 "
              f"elsewhere): {b:.2f} bits per slot, {b / 8 - 1:+.0%} against bf16")
    cost["fleet"] = fleet

    print("\n" + "=" * 112)
    print("[C1] the same rules on the 100 real Muon momenta (128 x 32, every fifth update)")
    print("=" * 112)
    npz = os.path.join(HERE, "pilot11_momentum.npz")
    if not os.path.exists(npz):                      # the archive keeps it at its root
        npz = os.path.join(HERE, "..", "pilot11_momentum.npz")
    z = np.load(npz)
    upds = [u.astype(float) for u in z["upds"][::5][:100]]
    ells, svs, refs = [], [], []
    for M in upds:
        sv = np.linalg.svd(M, compute_uv=False)
        ells.append(float(sv[-1] / sv[0]))
        refs.append(polar(M))
    print(f"  {len(upds)} momenta; dynamic range (sigma_min/sigma_max) median "
          f"{med(ells):.4f}, 10th percentile {np.percentile(ells, 10):.4f}, min {min(ells):.4f}")
    real = {}
    for name, rule in RULES:
        subs, states = [], []
        for M, ell, ref in zip(upds, ells, refs):
            co, _ = pe_coeffs(ell, K_DEF, CUSH, rule["safety"])
            if not np.all(np.isfinite(np.array(co, float))):
                continue
            X0 = M / n2(M)
            st, sub = run_rule(co, X0, ref, rule)
            subs.append(sub)
            states.append(st)
        real[name] = dict(median_floor=med(subs), max_floor=span(subs)[1],
                          n_cap=sum(1 for s in subs if not np.isfinite(s) or s >= 1.8),
                          n_div=sum(1 for s in states if not np.isfinite(s) or s > 5.0))
        print(f"  {name:>14}: floor median {med(subs):.4e}  max {span(subs)[1]:.2e}  "
              f"cap {real[name]['n_cap']}/100  blow-up {real[name]['n_div']}/100")
    base = real["bf16"]["median_floor"]
    for name in real:
        real[name]["over_bf16"] = real[name]["median_floor"] / base
    print(f"  floor relative to bf16 on the same 100 momenta: "
          + ", ".join(f"{k} {v['over_bf16']:.1f}x" for k, v in real.items()))
    print("  on this benign range the rule only decides how much of bf16's accuracy is"
          " given back, not whether it blows up:")
    print(f"    E4M3 {real['E4M3']['over_bf16']:.1f}x at {cost['E4M3']['bits_rect']:.2f} bits, "
          f"cast8+2 {real['cast8+2']['over_bf16']:.1f}x at {cost['cast8+2']['bits_rect']:.2f}, "
          f"E4M3 cast8+2 {real['E4M3 cast8+2']['over_bf16']:.1f}x at "
          f"{cost['E4M3 cast8+2']['bits_rect']:.2f}")
    R["c1_real"] = dict(n=len(upds), rules=real, ell_median=med(ells),
                        ell_p10=float(np.percentile(ells, 10)), ell_min=float(min(ells)))

    print("\n" + "=" * 112)
    print("[C2] the ill-conditioned tail: d = 64, four seeds, logspace at three dynamic")
    print("     ranges - where the rules differ")
    print("=" * 112)
    print(f"  {'rule':>14} {'floor 1e-1':>12} {'floor 1e-2':>12} {'floor 3e-3':>12} "
          f"{'cap 3e-3':>9} {'bits':>6}")
    tail = {}
    for name, rule in RULES:
        res = {}
        for L in (1e-1, 1e-2, 3e-3):
            subs, caps = [], 0
            for seed in range(4):
                M, ref, sv = spectra("logspace", 64, L, seed)
                co, _ = pe_coeffs(L, K_DEF, CUSH, rule["safety"])
                if not np.all(np.isfinite(np.array(co, float))):
                    raise ValueError(f"design coefficients are not finite at ell={L:g}")
                X0 = M / n2(M)
                st, sub = run_rule(co, X0, ref, rule)
                subs.append(sub)
                if not np.isfinite(sub) or sub >= 1.8:
                    caps += 1
            res[L] = dict(floor=med(subs), cap=caps)
        tail[name] = res
        _, bits = rule_bits(rule, 64, 64)
        print(f"  {name:>14} {res[1e-1]['floor']:12.3e} {res[1e-2]['floor']:12.3e} "
              f"{res[3e-3]['floor']:12.3e} {res[3e-3]['cap']:9d} {bits:6.2f}")
    R["c2_tail"] = {k: {str(L): v for L, v in d.items()} for k, d in tail.items()}

    print("\n" + "=" * 112)
    print("[C3] rule 1 - the safety factor at four bits")
    print("=" * 112)
    print("  measured (pilot 18, 12-spectrum census at ell = 1e-3, p = 4):")
    print("    s = 1.01   -> 12/12 blow-up (state distance up to 6.1e34), floor at the cap")
    print("    s = 1.035  ->  0/12 blow-up, floor still at the cap")
    print("  cost of the damping at the benign ranges (same census):")
    print("    ell = 1e-1: 0.2166 -> 0.2208 (+1.9%),  ell = 1e-2: 0.877 -> 0.841 (-4.1%)")
    t = R["c2_tail"]
    print(f"  on this run's four-seed d = 64 ladder the same move reads "
          f"{t['E5M2 s1.01']['0.01']['floor']:.3f} -> {t['E5M2 s1.035']['0.01']['floor']:.3f} "
          f"at ell = 1e-2 (where the undamped rule saturates) and "
          f"{t['E5M2 s1.01']['0.1']['floor']:.3f} -> {t['E5M2 s1.035']['0.1']['floor']:.3f} "
          f"at ell = 1e-1.")
    print("  so the damping is free; it removes the blow-up and not the floor, and both")
    print("  statements are what a rule has to carry.")
    R["c3_rule1"] = dict(s101_div="12/12", s1035_div="0/12",
                         cost_ell1e_1=[0.2166, 0.2208], cost_ell1e_2=[0.877, 0.841])

    print("\n" + "=" * 112)
    print("[C4] rule 2 - sigma_min as a one-sided input (pilot 13's 252-decision sweep)")
    print("=" * 112)
    print("  designing at ell_hat = alpha x ell, error ratio to the correct design:")
    print("    alpha 0.1: median 1.000, max 1.007, up to 3 extra steps")
    print("    alpha 0.5: median 1.000, max 1.071, up to 1 extra step")
    print("    alpha 2.0: median 2.64, max 58.8;  alpha 10: median 5.97, max 244")
    print("  under-estimating costs steps, over-estimating costs error silently.")
    R["c4_rule2"] = dict(alpha_0p1=dict(err_max=1.007, dK_max=3),
                         alpha_0p5=dict(err_max=1.071, dK_max=1),
                         alpha_2=dict(err_median=2.64, err_max=58.8),
                         alpha_10=dict(err_median=5.97, err_max=244.4))


def main():
    R["config"] = dict(K=K_DEF, cushion=CUSH, safety=SAFE, C18=C18,
                       mant=np.sqrt(3.0 / (2.0 * np.log(2.0))), numpy=np.__version__)
    sectionA()
    sectionB()
    sectionC()
    R.pop("_hist", None)
    out = {k: v for k, v in R.items() if k != "_hist"}
    path = os.path.join(HERE, "pilot20_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
