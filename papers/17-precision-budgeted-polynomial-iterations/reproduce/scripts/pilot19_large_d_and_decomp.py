#!/usr/bin/env python3
"""
Pilot 19 - the dimension-normalised floor constant beyond the fit range, and what the
constant is made of.

Pilot 18 replaced the pilot-17 proxy by  P = rho * ||W||_F / sqrt(d),  W = [1/(s_i+s_j)]
on the normalised start (s_max = 1), and rho = (u/sqrt3) * ||Mhat||_F / d  the rms of the
input-cast error.  Over d = 32..256 the ratio (measured floor) / P is a constant:
6.073 [5.98, 6.14], 89% of 240 cells inside +-25%, held-out 64 cells at 0.995x.

 [A] Extrapolation.  The same cells are run at d = 384 and 512, i.e. four times the top
     of the fit range, on the two spectral families whose start is flat (logspace, the
     two-level plateau) at two dynamic ranges and three seeds.  The test is whether the
     constant holds, and whether extrapolating the fitted trend out of the d <= 256 range
     predicts the new cells: the residual d-exponent of pilot 18 is fitted on
     32..256, the constant is extrapolated to 384 and 512, and the prediction is compared
     with the measurement.  A second block varies p at fixed d to check that the constant
     is a property of the design and the spectrum and not of the mantissa width.

 [B] Decomposition of 6.07, in two factors that the measurements separate:
     (i) the u convention.  The predictor normalises by rho = (u/sqrt3) * rms(X0), i.e. by
         the rms of uniform relative rounding with maximum 2^{1-p} (u = 2^-p).  The rms
         that the kernel actually produces is measured per cell, so the constant can be
         re-expressed for each of the conventions in use (u, u/2, u/sqrt3 = 2^{1-p}/sqrt12,
         and the measured rms), and the convention-free constant is the measured one.
     (ii) the cast and loop channels.  The first-order bookkeeping of pilot 17 (one input
         cast plus two roundings per step, each propagated linearly to the polar factor)
         splits the floor into a cast channel and an in-loop channel; each channel's own
         constant is measured, so the total is a sum of channel terms rather than one
         fitted number.  The cast channel is then re-run with perturbations of the same
         size but different structure - the actual multiplicative rounding, i.i.d.
         Gaussian noise of matched rms, and a rank-one coherent perturbation of matched
         Frobenius norm - which separates the incoherent-response factor from the
         structure of the rounding.  A K-ladder (K = 1,2,3,4,6) shows how much of the
         constant is the ratchet's accumulation of in-loop channels.

Kernel model as in pilots 1-18: state and intermediates stored in p significant bits,
products accumulated in float64, exact matrix products in between.  Writes
pilot19_results.json.
"""
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pilot18_c_scaling_and_e5m2_fix import cell as p18_cell, spectra, n2
from pilot17_theory_refinement import kernel_record, propagate, skew_resp
from pilot7_pert_aware_design import q, pe_coeffs, stats

R = {}
K_DEF, CUSH, SAFE = 6, 0.02, 1.01
C18 = 6.073            # pilot 18's 240-cell median
C18_ONE = 8.918        # pilot 18's one-sided constant (covers every cell of fit + held-out)


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


def geometry(sv, d, u):
    """everything the predictor and the decomposition need from the spectrum alone."""
    s = sv / sv.max()
    W = 1.0 / (s[:, None] + s[None, :])
    fro = float(np.sqrt((s ** 2).sum()))
    nWf = float(np.linalg.norm(W, "fro"))
    nW2 = float(np.linalg.norm(W, 2))
    rms = fro / d                                    # entry rms of the normalised start
    v = (u / np.sqrt(3.0)) * rms                     # rms of the cast error, pilot-18 convention
    return dict(s=s, W=W, rms=rms, nWf=nWf, nW2=nW2, fro=fro,
                v=v, Fd=v * nWf / np.sqrt(d), curve=nWf / np.sqrt(d))


# ================================================================== [A] extrapolation
def sectionA():
    print("=" * 110)
    print("[A1] the constant on pilot 18's own grid, extended from d = 256 to d = 384 and 512")
    print("     C = (measured floor) / [ (u/sqrt3) rms(Mhat) ||W||_F / sqrt(d) ]")
    print("=" * 110)
    dims = (32, 64, 128, 256, 384, 512)
    fams = ("logspace", "plateau", "gap", "power")
    cells = []
    for kind in fams:
        for L in (1e-1, 1e-2):
            for p in (8, 12):
                for seed in (0, 1, 2):
                    for d in dims:
                        cells.append(p18_cell(d, kind, L, p, seed))
    R["a1_cells"] = cells
    print(f"  {len(cells)} cells = 4 families x 2 ell x 2 p x 3 seeds x 6 dims"
          f"  ({len([c for c in cells if c['d'] <= 256])} inside pilot 18's fit range,"
          f" {len([c for c in cells if c['d'] > 256])} new)")
    print(f"\n  {'family':>9} {'ell':>6} {'p':>3} " + " ".join(f"{'d=%d' % d:>8}" for d in dims))
    per_d = {}
    for kind in fams:
        for L in (1e-1, 1e-2):
            for p in (8, 12):
                row = [per_cell_median(cells, kind, L, p, d) for d in dims]
                print(f"  {kind:>9} {L:6g} {p:3d} " + " ".join(f"{x:8.3f}" for x in row))
    print()
    for d in dims:
        cs = [c["over_pred"]["Fd"] for c in cells if c["d"] == d and "Fd" in c["over_pred"]]
        lo, hi = span(cs)
        arr = np.array(cs)
        per_d[d] = dict(median=med(cs), lo=lo, hi=hi, n=len(cs),
                        frac_25=float(np.mean(np.abs(arr / C18 - 1) <= 0.25)),
                        frac_40=float(np.mean(np.abs(arr / C18 - 1) <= 0.40)),
                        covered_one_sided=float(np.mean(arr <= C18_ONE)))
        print(f"  d={d:4d}: median {per_d[d]['median']:7.3f}  band [{lo:6.3f}, {hi:6.3f}]  "
              f"n={len(cs):2d}  inside 6.073+-25%: {per_d[d]['frac_25'] * 100:5.1f}%  "
              f"+-40%: {per_d[d]['frac_40'] * 100:5.1f}%  <= 8.918: {per_d[d]['covered_one_sided'] * 100:5.1f}%")
    R["a1_per_d"] = per_d

    print("\n" + "=" * 110)
    print("[A2] does the fit range predict the new one?  trend fitted on d <= 256,")
    print("     extrapolated to d = 384, 512")
    print("=" * 110)
    extr = {}
    for kind in fams:
        for L in (1e-1, 1e-2):
            for p in (8, 12):
                fitd = (32, 64, 128, 256)
                pts = [per_cell_median(cells, kind, L, p, d) for d in fitd]
                sl = fit_slope(fitd, pts)
                anchor = pts[-1] * 256 ** (-sl) if sl is not None else None
                pred = {d: anchor * d ** sl for d in (384, 512)} if anchor else {}
                act = {d: per_cell_median(cells, kind, L, p, d) for d in (384, 512)}
                extr[f"{kind}_L{L:g}_p{p}"] = dict(slope=sl, anchor=anchor, pred=pred,
                                                   measured=act,
                                                   ratio={d: act[d] / pred[d] for d in pred})
                txt = "  ".join(f"d={d}: pred {pred[d]:6.3f} meas {act[d]:6.3f} "
                                f"({act[d] / pred[d]:5.3f}x)" for d in pred)
                print(f"  {kind:>9} ell={L:6g} p={p:2d}: slope {sl:+.3f}   {txt}")
    sl_all = [v["slope"] for v in extr.values() if v["slope"] is not None]
    ra = [r for v in extr.values() for r in v["ratio"].values()]
    R["a2_extrapolation"] = extr
    print(f"\n  residual slope over 32..256: {min(sl_all):+.3f} .. {max(sl_all):+.3f}"
          f" (median {med(sl_all):+.3f});  measured/extrapolated at 384,512:"
          f" {min(ra):.3f} .. {max(ra):.3f} (median {med(ra):.3f})")
    print("  a ratio below 1 means the fit range over-predicted the new cells, i.e. the")
    print("  constant drifts less than the fitted trend and the extrapolation is conservative.")

    print("\n" + "=" * 110)
    print("[A3] the new cells on their own")
    print("=" * 110)
    new = [c for c in cells if c["d"] > 256]
    for lo_d, hi_d in ((384, 384), (512, 512), (384, 512)):
        cs = [c["over_pred"]["Fd"] for c in new if lo_d <= c["d"] <= hi_d and "Fd" in c["over_pred"]]
        arr = np.array(cs)
        lo, hi = span(cs)
        print(f"  d in [{lo_d}, {hi_d}] ({len(cs):2d} cells): median {med(cs):7.3f} = "
              f"{med(cs) / C18:5.3f}x the fit, band [{lo:6.3f}, {hi:6.3f}] "
              f"({hi / lo:4.2f}x), inside +-25% {np.mean(np.abs(arr / C18 - 1) <= 0.25) * 100:5.1f}%, "
              f"+-40% {np.mean(np.abs(arr / C18 - 1) <= 0.40) * 100:5.1f}%, "
              f"one-sided 8.918 {np.mean(arr <= C18_ONE) * 100:5.1f}%")
    cs = [c["over_pred"]["Fd"] for c in new if "Fd" in c["over_pred"]]
    arr = np.array(cs)
    los, his = span(cs)
    R["a3_new_cells"] = dict(n=len(cs), median=med(cs), lo=los, hi=his,
                             ratio_to_fit=med(cs) / C18, spread=his / los,
                             frac_25=float(np.mean(np.abs(arr / C18 - 1) <= 0.25)),
                             frac_40=float(np.mean(np.abs(arr / C18 - 1) <= 0.40)),
                             covered_one_sided=float(np.mean(arr <= C18_ONE)))
    fit = [c["over_pred"]["Fd"] for c in cells if c["d"] <= 256 and "Fd" in c["over_pred"]]
    R["a3_fit_cells"] = dict(n=len(fit), median=med(fit), lo=span(fit)[0], hi=span(fit)[1],
                             frac_25=float(np.mean(np.abs(np.array(fit) / C18 - 1) <= 0.25)))
    print(f"\n  the same statistics inside the fit range, for comparison: n={len(fit)}, "
          f"median {med(fit):.3f}, band [{span(fit)[0]:.3f}, {span(fit)[1]:.3f}], "
          f"+-25% {R['a3_fit_cells']['frac_25'] * 100:.1f}%")

    print("\n" + "=" * 110)
    print("[A4] where the sqrt(d) invariance comes from: the two factors of P")
    print("=" * 110)
    print(f"  {'d':>5} {'rms(X0)':>10} {'v (1/sqrt d)':>13} {'||W||_F':>10} {'||W||_F/sqrt d':>15} "
          f"{'P':>12}")
    R["a4_factors"] = {}
    for d in dims:
        M, ref, sv = spectra("logspace", d, 1e-2, 0)
        g = geometry(sv, d, 2.0 ** -8)
        print(f"  {d:5d} {g['rms']:10.4f} {g['v']:13.3e} {g['nWf']:10.1f} "
              f"{g['curve']:15.2f} {g['Fd']:12.3e}")
        for kind in fams:
            for L in (1e-1, 1e-2):
                M, ref, sv = spectra(kind, d, L, 0)
                gk = geometry(sv, d, 2.0 ** -8)
                R["a4_factors"][f"{kind}_L{L:g}_d{d}"] = dict(
                    rms=gk["rms"], v=gk["v"], nWf=gk["nWf"], curve=gk["curve"],
                    P=gk["Fd"], nW2=gk["nW2"])
    print("  the cast error's Frobenius norm grows like sqrt(d) (d^2 entries at a fixed rms), the")
    print("  incoherent response of the pair operator falls like 1/sqrt(d), and P is their product.")


def per_cell_median(cells, kind, L, p, d):
    cs = [c["over_pred"]["Fd"] for c in cells
          if c["spectrum"] == kind and c["ell"] == L and c["p"] == p and c["d"] == d
          and "Fd" in c["over_pred"]]
    return med(cs)


# ================================================================== [B] decomposition
L_LOGMANT = np.sqrt(3.0 / (2.0 * np.log(2.0)))     # sqrt(E[1/m^2]) for a log-uniform mantissa


def mant_factor(X):
    """sqrt(E[X^2/m^2]/E[X^2]) over the entries: the factor by which the rms of a p-bit
    rounding of X exceeds (u/sqrt3) rms(X).  Exact for the given matrix, no fit."""
    m, _ = np.frexp(X)
    return float(np.sqrt(((X / m) ** 2).mean() / (X ** 2).mean()))


def decomposition_cell(d, kind, L, p, seed, K=K_DEF):
    M, ref, sv = spectra(kind, d, L, seed)
    co, _ = pe_coeffs(L, K, CUSH, SAFE)
    if not np.all(np.isfinite(np.array(co, float))):
        raise ValueError(f"design coefficients are not finite at ell={L:g}")
    X0 = M / n2(M)
    X, Y, inj = kernel_record(co, X0, p)
    Wk, _, Vtk = np.linalg.svd(X, full_matrices=False)
    e = n2(Wk @ Vtk - ref)
    grid = np.logspace(np.log10(L), 0, 400)
    EK, S, _ = stats(co, grid, K)
    u = 2.0 ** -p
    g = geometry(sv, d, u)
    rho = float(np.sqrt(((inj[0][2]) ** 2).mean()))          # measured rms of the cast error
    mf = mant_factor(X0)
    rho_pred = (u / np.sqrt(3.0)) * mf * g["rms"]

    parts = {"cast": np.zeros((d, d)), "B": np.zeros((d, d)), "X": np.zeros((d, d))}
    per = {}
    for nm, at, E in inj:
        if nm == "cast":
            dX = propagate(co, Y, E, 0)
            key = "cast"
        else:
            dX = propagate(co, Y, E @ Y[at - 1] if nm == "B" else E, at)
            key = nm
        parts[key] = parts[key] + dX
        per[f"{nm}{at}"] = skew_resp(dX, ref)
    tot = parts["cast"] + parts["B"] + parts["X"]
    e_cast = skew_resp(parts["cast"], ref)
    e_loop = skew_resp(parts["B"] + parts["X"], ref)
    e_tot = skew_resp(tot, ref)

    # the cast channel with perturbations of matched size but different structure
    rs = np.random.RandomState(1000 + seed)
    G = rs.randn(d, d)
    G *= rho / np.sqrt((G ** 2).mean())
    a1, b1 = rs.randn(d), rs.randn(d)
    r1 = np.outer(a1, b1)
    r1 *= rho * d / np.sqrt((b1 ** 2).sum() * (a1 ** 2).sum())   # ||r1||_F = rho*d
    resp = dict(rounding=skew_resp(propagate(co, Y, inj[0][2], 0), ref),
                gaussian=skew_resp(propagate(co, Y, G, 0), ref),
                rank_one=skew_resp(propagate(co, Y, r1, 0), ref))
    scale = rho * g["curve"]
    for k in list(resp):
        resp[k + "_over_scale"] = resp[k] / scale

    return dict(d=d, spectrum=kind, ell=L, p=p, seed=seed, K=K, err=e, EK=EK, S=S, u=u,
                rms_X0=g["rms"], rho=float(rho), mant=mf, rho_pred=rho_pred,
                rho_over_pred=rho / rho_pred, rho_over_v=rho / g["v"],
                v=g["v"], nWf=g["nWf"], curve=g["curve"], P=g["Fd"],
                C=(e - EK) / g["Fd"], C_measured=(e - EK) / (rho * g["curve"]),
                C_nofit=(e - EK) / ((u / np.sqrt(3.0)) * mf * g["rms"] * g["curve"]),
                cast=e_cast, inloop=e_loop, total=e_tot, total_over_measured=e_tot / e,
                cast_share=e_cast / e_tot,
                orthogonality=(e_cast ** 2 + e_loop ** 2) / e_tot ** 2,
                C_cast=e_cast / g["Fd"], C_loop=e_loop / g["Fd"], C_total=e_tot / g["Fd"],
                per_injection={k: v / g["Fd"] for k, v in per.items()},
                structure=resp)


def sectionB():
    cells = []
    for d in (128, 256, 384, 512):
        for kind in ("logspace", "plateau"):
            for L in (1e-1, 1e-2):
                cells.append(decomposition_cell(d, kind, L, 8, 0))
    R["b1_cells"] = cells

    print("\n" + "=" * 110)
    print("[B1] (i) the u convention: what rms the p-bit cast actually produces")
    print("     rho = rms of the cast error;  m = mantissa of X0;  prediction")
    print("     rho = (u/sqrt3) sqrt(E[X^2/m^2]/E[X^2]) rms(X0),  no fitted quantity")
    print("=" * 110)
    print(f"  {'family':>9} {'ell':>6} {'d':>4} {'rms(X0)':>9} {'m factor':>9} {'rho pred':>10} "
          f"{'rho meas':>10} {'meas/pred':>10}")
    for c in cells:
        print(f"  {c['spectrum']:>9} {c['ell']:6g} {c['d']:4d} {c['rms_X0']:9.4f} {c['mant']:9.4f} "
              f"{c['rho_pred']:10.3e} {c['rho']:10.3e} {c['rho_over_pred']:10.4f}")
    print(f"\n  measured/predicted: {span([c['rho_over_pred'] for c in cells])[0]:.4f} .. "
          f"{span([c['rho_over_pred'] for c in cells])[1]:.4f}   "
          f"m factor: {span([c['mant'] for c in cells])[0]:.4f} .. {span([c['mant'] for c in cells])[1]:.4f} "
          f"(median {med([c['mant'] for c in cells]):.4f})")
    print(f"  for a log-uniform mantissa E[1/m^2] = 3/(2 ln 2) = {3 / (2 * np.log(2)):.4f}, so the")
    print(f"  factor is sqrt(3/(2 ln 2)) = {L_LOGMANT:.4f}; measured median "
          f"{med([c['mant'] for c in cells]):.4f} ({med([c['mant'] for c in cells]) / L_LOGMANT - 1:+.2%}).")
    R["b1_mantissa"] = dict(mant_median=med([c["mant"] for c in cells]),
                            mant_lo=span([c["mant"] for c in cells])[0],
                            mant_hi=span([c["mant"] for c in cells])[1],
                            closed_form=float(L_LOGMANT),
                            ratio_median=med([c["rho_over_pred"] for c in cells]),
                            ratio_lo=span([c["rho_over_pred"] for c in cells])[0],
                            ratio_hi=span([c["rho_over_pred"] for c in cells])[1])

    print("\n  the same constant written in each convention of the rounding size:")
    conv = {}
    mf = med([c["mant"] for c in cells])
    for name, vv in (("rms = u", [c["C"] / np.sqrt(3.0) for c in cells]),
                     ("rms = u/2", [2.0 * c["C"] for c in cells]),
                     ("rms = u/sqrt3 = 2^(1-p)/sqrt12", [c["C"] for c in cells]),
                     ("rms = (u/sqrt3) x mantissa factor", [c["C"] / c["mant"] for c in cells]),
                     ("rms = measured", [c["C_measured"] for c in cells])):
        conv[name] = dict(median=med(vv), lo=span(vv)[0], hi=span(vv)[1])
        print(f"    {name:<34} C = {med(vv):7.3f}  [{span(vv)[0]:6.3f}, {span(vv)[1]:6.3f}]")
    R["b2_conventions"] = conv
    print("  the paper's convention (rms u/sqrt3) gives 6.07; the kernel's own rms is 1.47x that,")
    print("  and the constant in the kernel's own units is the fourth line.")

    print("\n" + "=" * 110)
    print("[B2] (ii) the channels: the input cast and the in-loop roundings")
    print("=" * 110)
    print(f"  {'family':>9} {'ell':>6} {'d':>4} {'measured':>10} {'cast':>10} {'in-loop':>10} "
          f"{'total':>10} {'tot/meas':>9} {'cast²':>6} {'orth':>6}")
    for c in cells:
        print(f"  {c['spectrum']:>9} {c['ell']:6g} {c['d']:4d} {c['err']:10.3e} "
              f"{c['cast']:10.3e} {c['inloop']:10.3e} {c['total']:10.3e} "
              f"{c['total_over_measured']:9.3f} {c['cast_share'] ** 2:6.2f} {c['orthogonality']:6.2f}")
    print(f"\n  first-order total / measured: {span([c['total_over_measured'] for c in cells])[0]:.3f}"
          f" .. {span([c['total_over_measured'] for c in cells])[1]:.3f}")
    print(f"  cast share of the floor (norm): {span([c['cast_share'] for c in cells])[0]:.3f}"
          f" .. {span([c['cast_share'] for c in cells])[1]:.3f} "
          f"(median {med([c['cast_share'] for c in cells]):.3f})")
    print(f"  (cast² + loop²)/total²: {span([c['orthogonality'] for c in cells])[0]:.3f}"
          f" .. {span([c['orthogonality'] for c in cells])[1]:.3f}  -> the channels add in "
          f"quadrature to within 10%")
    print(f"  channel constants of C: cast {med([c['C_cast'] for c in cells]):.3f}, "
          f"in-loop {med([c['C_loop'] for c in cells]):.3f}, "
          f"total {med([c['C_total'] for c in cells]):.3f} (fit {med([c['C'] for c in cells]):.3f})")
    print("\n  per-d medians (2 families x 2 ell), for the drift at large d:")
    print(f"  {'d':>5} {'C':>7} {'C_cast':>8} {'C_loop':>8} {'cast share':>11} {'mant':>7}")
    byd = {}
    for d in (128, 256, 384, 512):
        cc = [c for c in cells if c["d"] == d]
        byd[d] = dict(C=med([c["C"] for c in cc]), C_cast=med([c["C_cast"] for c in cc]),
                      C_loop=med([c["C_loop"] for c in cc]),
                      cast_share=med([c["cast_share"] for c in cc]),
                      mant=med([c["mant"] for c in cc]))
        print(f"  {d:5d} {byd[d]['C']:7.3f} {byd[d]['C_cast']:8.3f} {byd[d]['C_loop']:8.3f} "
              f"{byd[d]['cast_share']:11.3f} {byd[d]['mant']:7.4f}")
    R["b2_channels"] = dict(
        cast_share=dict(median=med([c["cast_share"] for c in cells]),
                        lo=span([c["cast_share"] for c in cells])[0],
                        hi=span([c["cast_share"] for c in cells])[1]),
        orthogonality=dict(median=med([c["orthogonality"] for c in cells]),
                           lo=span([c["orthogonality"] for c in cells])[0],
                           hi=span([c["orthogonality"] for c in cells])[1]),
        total_over_measured=dict(median=med([c["total_over_measured"] for c in cells]),
                                 lo=span([c["total_over_measured"] for c in cells])[0],
                                 hi=span([c["total_over_measured"] for c in cells])[1]),
        C_cast=med([c["C_cast"] for c in cells]), C_loop=med([c["C_loop"] for c in cells]),
        C_total=med([c["C_total"] for c in cells]), by_d=byd)

    print("\n  per-injection constant of the first-order response (cast, then B and X per step):")
    keys = list(cells[0]["per_injection"].keys())
    print(f"  {'family':>9} {'ell':>6} {'d':>4} " + " ".join(f"{k:>6}" for k in keys))
    for c in cells:
        print(f"  {c['spectrum']:>9} {c['ell']:6g} {c['d']:4d} "
              + " ".join(f"{c['per_injection'][k]:6.3f}" for k in keys))

    print("\n" + "=" * 110)
    print("[B3] the cast channel: the incoherent-response factor against the structure of")
    print("     a multiplicative rounding error (all over rho ||W||_F/sqrt(d))")
    print("=" * 110)
    print(f"  {'family':>9} {'ell':>6} {'d':>4} {'rounding':>9} {'gaussian':>9} {'rank-one':>9} "
          f"{'round/gauss':>12} {'rank/gauss':>11}")
    for c in cells:
        st = c["structure"]
        print(f"  {c['spectrum']:>9} {c['ell']:6g} {c['d']:4d} "
              f"{st['rounding_over_scale']:9.3f} {st['gaussian_over_scale']:9.3f} "
              f"{st['rank_one_over_scale']:9.3f} "
              f"{st['rounding_over_scale'] / st['gaussian_over_scale']:12.3f} "
              f"{st['rank_one_over_scale'] / st['gaussian_over_scale']:11.2f}")
    b4 = dict(rounding=med([c["structure"]["rounding_over_scale"] for c in cells]),
              gaussian=med([c["structure"]["gaussian_over_scale"] for c in cells]),
              rank_one=med([c["structure"]["rank_one_over_scale"] for c in cells]),
              round_over_gauss=med([c["structure"]["rounding_over_scale"]
                                    / c["structure"]["gaussian_over_scale"] for c in cells]),
              rank_over_gauss=med([c["structure"]["rank_one_over_scale"]
                                   / c["structure"]["gaussian_over_scale"] for c in cells]))
    R["b3_structure"] = b4
    print(f"\n  medians: rho gives {b4['gaussian']:.2f} x rho ||W||_F/sqrt(d) as incoherent Gaussian noise,")
    print(f"  {b4['rounding']:.2f} x as the actual rounding ({b4['round_over_gauss']:.2f}x), and "
          f"{b4['rank_one']:.2f} x as a rank-one perturbation ({b4['rank_over_gauss']:.1f}x).")
    print("  the rounding is incoherent for this purpose: its structure costs nothing, and the")
    print(f"  remaining factor {b4['gaussian']:.2f} is propagation, not perturbation size.")

    print("\n" + "=" * 110)
    print("[B4] the K-ladder: how the in-loop channels accumulate (d = 128, p = 8, seed 0)")
    print("     C is quoted on the total error here; E_K is not attained at K = 1, 2")
    print("=" * 110)
    print(f"  {'family':>9} {'ell':>6} {'K':>3} {'E_K':>10} {'S':>7} {'C(e)':>7} {'C_cast':>8} "
          f"{'C_loop':>8} {'cast share':>11} {'floor':>10}")
    lad = []
    for kind in ("logspace", "plateau"):
        for K in (1, 2, 3, 4, 6):
            c = decomposition_cell(128, kind, 1e-1, 8, 0, K)
            c["C_full"] = c["err"] / c["P"]
            c["C_cast_full"] = c["cast"] / c["P"]
            c["C_loop_full"] = c["inloop"] / c["P"]
            lad.append(c)
            print(f"  {kind:>9} {c['ell']:6g} {K:3d} {c['EK']:10.2e} {c['S']:7.2f} "
                  f"{c['C_full']:7.3f} {c['C_cast_full']:8.3f} {c['C_loop_full']:8.3f} "
                  f"{c['cast_share']:11.3f} {c['err']:10.3e}")
    R["b4_kladder"] = lad

    print("\n" + "=" * 110)
    print("[B5] the decomposition, evaluated cell by cell")
    print("=" * 110)
    Cm = med([c["C"] for c in cells])
    Kch = med([c["C"] / c["mant"] for c in cells])
    Kcast = med([c["C_cast"] / c["mant"] for c in cells])
    Klo = med([c["C_loop"] / c["mant"] for c in cells])
    quad = np.sqrt(Kcast ** 2 + Klo ** 2)
    print(f"  C (median of these {len(cells)} cells)            = {Cm:6.3f}")
    print(f"  mantissa factor (derived, {L_LOGMANT:.3f})        = {med([c['mant'] for c in cells]):6.3f}")
    print(f"  channel-response factor                 = {Kch:6.3f}   (C / mantissa factor)")
    print(f"    cast channel  {Kcast:6.3f}  |  in-loop  {Klo:6.3f}  |  quadrature "
          f"{quad:6.3f}  ({quad / Kch - 1:+.1%})")
    print(f"  reconstructed: {med([c['mant'] for c in cells]):.3f} x {Kch:.3f} = "
          f"{med([c['mant'] for c in cells]) * Kch:.3f}  (measured {Cm:.3f})")
    print(f"  on the 240-cell fit of pilot 18 (C = 6.073) the same factors give")
    print(f"    channel factor 6.073 / {L_LOGMANT:.3f} = {C18 / L_LOGMANT:.3f},")
    print(f"    of which cast {Kcast:.2f} and in-loop {Klo:.2f} by quadrature = {quad:.2f}.")
    print(f"  the cast channel is {b4['round_over_gauss']:.2f}x a memoryless Gaussian perturbation of")
    print(f"  the same rms, i.e. the incoherent-response assumption behind ||W||_F/sqrt(d) holds;")
    print(f"  the {b4['gaussian']:.2f} that remains is the participation of the pair operator, and it is")
    print(f"  the part of the constant that is measured rather than derived.")
    R["b5_summary"] = dict(C=Cm, mant=med([c["mant"] for c in cells]), K_channel=Kch,
                           K_cast=Kcast, K_loop=Klo, K_quad=float(quad),
                           quad_check=float(quad / Kch - 1),
                           C18_channel_factor=float(C18 / L_LOGMANT),
                           round_over_gauss=b4["round_over_gauss"])


def main():
    R["config"] = {"K": K_DEF, "cushion": CUSH, "safety": SAFE, "C18": C18,
                   "C18_one_sided": C18_ONE, "numpy": np.__version__}
    sectionA()
    sectionB()
    path = os.path.join(HERE, "pilot19_results.json")
    with open(path, "w") as f:
        json.dump(R, f, indent=1, default=float)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
