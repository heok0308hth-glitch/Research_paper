#!/usr/bin/env python3
"""
Pilot 15 - the exponent range of fp8.

Pilots 1-14 model storage as p significant bits with an unlimited exponent, so the
rounding is a relative error u = 2^-p wherever a value sits.  fp8 formats have a finite
exponent window, and every decision the paper makes is measured under the assumption
that this does not matter.  This script asks whether it does.

Formats (OCP FP8, round-to-nearest-even, subnormals, saturate at the largest finite):
  e4m3  1+4+3, bias 7 :  normals [2^-6, 448],     subnormal step 2^-9
  e5m2  1+5+2, bias 15:  normals [2^-14, 57344],  subnormal step 2^-16
Matching the paper's parameterisation: in the normal range the e4m3 grid is exactly the
grid of the paper's q(.,5) and the e5m2 grid is q(.,4) -- verified elementwise in [A],
because the paper's q rounds the frexp mantissa in [0.5,1) to 2^-(p-1).  So the matched
ideal references are ideal_p5 (e4m3, u = 2^-5) and ideal_p4 (e5m2, u = 2^-4), and the
explicit mantissa counts are 3 and 2.

 [A] format definitions against an exhaustive enumeration of the codes; the grid of each
     format against the paper's q at the matched p, over the design window.
 [B] the law: subspace floor and certificate ratio for ideal_p5 / e4m3 / e4m3 with
     per-tensor power-of-two scales / ideal_p4 / e5m2 / e5m2 scaled / ideal_p3, at
     ell in {0.15, 0.1, 3.6e-2, 1.3e-2}, five seeds.
 [C] the window: state scale s = 2^k swept at fixed ell (subnormal cliff, saturation
     cliff, and how wide the flat window is).
 [D] which tensor leaves the window first, and what an unnormalised input does.
 [E] is the fp8 floor flat in K, and does it scale as u between the two formats?
 [F] the boundary in ell over five decades, with the count of subnormal and flushed
     stored singular directions -- the regime boundary as a computable predicate.
 [G] the Muon shape 128x32 at three dynamic ranges.

Kernel model as in pilot 12: state and intermediate quantised, products accumulated in
float64, no accumulation order.  Prediction: c = 0.142, S_pert from the coefficients,
u = 2^-p of the matched p.
"""
import json
import os
import numpy as np

from pilot7_pert_aware_design import q, pe_coeffs, stats
from pilot10_subspace_theory import polar_safe

HERE = os.path.dirname(os.path.abspath(__file__))
R = {}
C_CERT = 0.142
CUSH, SAFE = 0.02, 1.01
TARGET = 0.8          # per-tensor scale puts the largest entry at 0.8 * max finite value

FMT = {
    "e4m3": dict(eb=4, mb=3, bias=7, maxv=448.0, minn=2.0 ** -6, mins=2.0 ** -9, p=5),
    "e5m2": dict(eb=5, mb=2, bias=15, maxv=57344.0, minn=2.0 ** -14, mins=2.0 ** -16, p=4),
}


# ---------------------------------------------------------------- format
def qfmt(x, name):
    """Quantise to an fp8 format: RTNE, subnormals, saturate at the largest finite."""
    f = FMT[name]
    mb, bias, maxv, mins = f["mb"], f["bias"], f["maxv"], f["mins"]
    x = np.asarray(x, float)
    sgn = np.sign(x)
    ax = np.abs(x)
    r = np.zeros_like(ax)
    n = ax > 0
    if n.any():
        a = ax[n]
        m, e = np.frexp(a)                 # a = m 2^e, m in [0.5, 1)
        E = e - 1                          # a in [2^E, 2^(E+1))
        normal = E >= (1 - bias)
        out = np.empty_like(a)
        if normal.any():
            st = np.ldexp(1.0, E[normal] - mb)
            out[normal] = np.rint(a[normal] / st) * st
        if (~normal).any():
            out[~normal] = np.rint(a[~normal] / mins) * mins
        r[n] = out
    return sgn * np.minimum(r, maxv)


def all_codes(name):
    f = FMT[name]
    vals = set()
    for E in range(1 - f["bias"], f["bias"] + 1):
        for mm in range(2 ** f["mb"]):
            vals.add((1.0 + mm * 2.0 ** (-f["mb"])) * 2.0 ** E)
    vals = {v for v in vals if v <= f["maxv"]}
    for mm in range(1, 2 ** f["mb"]):
        vals.add(mm * f["mins"])
    return np.array(sorted(vals))


def fmt_check(name):
    f = FMT[name]
    c = all_codes(name)
    rt = qfmt(c, name)
    return {"n_finite_codes": int(c.size),
            "max_rel_deviation_on_codes": float(np.max(np.abs(rt - c) / c)),
            "spacing_above_one": float(qfmt(np.array([1 + 2.0 ** (-f["mb"])]), name)[0] - 1.0),
            "subnormal_step": f["mins"],
            "three_times_max": float(qfmt(np.array([3 * f["maxv"]]), name)[0]),
            "half_min_subnormal": float(qfmt(np.array([f["mins"] / 2]), name)[0]),
            "max_finite": f["maxv"], "min_normal": f["minn"], "min_subnormal": f["mins"],
            "matched_p": f["p"], "explicit_mantissa_bits": f["mb"]}


# ---------------------------------------------------------------- quantiser builders
def build(nm, sx=None, sB=None):
    if nm.startswith("ideal"):
        p = int(nm[-1])
        f = lambda v: q(v, p)
        return f, f, 2.0 ** -p, 1e300, 0.0, {}
    f8 = nm
    cap, minn, u = FMT[f8]["maxv"], FMT[f8]["minn"], 2.0 ** -FMT[f8]["p"]

    def mk(s):
        if s is None:
            return lambda v: qfmt(v, f8)
        return lambda v: qfmt(np.asarray(v, float) * s, f8) / s

    return mk(sx), mk(sB), u, cap, minn, {"sx": sx, "sB": sB}


def auto_scales(co, X0, nm, cap):
    """Per-tensor power-of-two scales (a delayed-scaling picture)."""
    sx = 2.0 ** int(np.floor(np.log2(TARGET * cap / max(float(np.max(np.abs(X0))), 1e-300))))
    amaxB = 0.0
    X = np.asarray(X0, float)
    for a, b, c in co:
        A = X @ X.T
        Braw = b * A + c * (A @ A)
        amaxB = max(amaxB, float(np.max(np.abs(Braw))))
        X = a * X + Braw @ X
    sB = 2.0 ** int(np.floor(np.log2(TARGET * cap / max(amaxB, 1e-300))))
    return sx, sB, amaxB


# ---------------------------------------------------------------- kernel
def mat(d, L, seed=1, m=None):
    rs = np.random.RandomState(seed)
    m = m or d
    U, _ = np.linalg.qr(rs.randn(m, d))
    V, _ = np.linalg.qr(rs.randn(d, d))
    sv = np.logspace(0, np.log10(L), d)
    return (U * sv) @ V.T, U @ V.T


def n2(X):
    return float(np.linalg.norm(X, 2))


def orbit(co, X0, qs, qi, cap, keep_x=False):
    st = {"maxB": 0.0, "sat": 0, "flush": int(np.sum((X0 != 0) & (qs(X0) == 0))),
          "maxX": float(np.max(np.abs(qs(X0))))}
    X = qs(X0)
    for a, b, c in co:
        A = X @ X.T
        Braw = b * A + c * (A @ A)
        st["maxB"] = max(st["maxB"], float(np.max(np.abs(Braw))))
        st["sat"] += int(np.sum(np.abs(Braw) > cap))
        B = qi(Braw)
        Xn = a * X + B @ X
        st["sat"] += int(np.sum(np.abs(Xn) > cap))
        st["maxX"] = max(st["maxX"], float(np.max(np.abs(Xn))))
        X = qs(Xn)
    if keep_x:
        st["_X"] = X
    return X, st


def measure(co, M, ref, nm, sx=None, sB=None, X0_override=None):
    if nm.startswith("ideal") or (sx is None and sB is None):
        qs, qi, u, cap, minn, sc = build(nm)
    else:
        qs, qi, u, cap, minn, sc = build(nm, sx, sB)
    X0 = M / n2(M) if X0_override is None else X0_override
    sv0 = np.linalg.svd(X0, compute_uv=False)
    grid = np.logspace(np.log10(max(float(sv0[-1] / sv0[0]), 1e-12)), 0, 400)
    EK, S, _ = stats(co, grid, len(co))
    pred = EK + C_CERT * u * S
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        X, st = orbit(co, X0, qs, qi, cap)
        pol = polar_safe(X) if np.all(np.isfinite(X)) else None
        e = n2(pol - ref) if pol is not None else None
        ex = n2(X - ref) if pol is not None else None
    svq = np.linalg.svd(qs(X0), compute_uv=False)
    svq = svq[sv0 > 0]
    return {"err": e, "errX": ex, "pred": float(pred), "EK": float(EK), "S_pert": float(S),
            "ratio": (float(pred) / e) if e else None, "u": u,
            "diverged": e is None,
            "min_sv_stored": float(svq.min()) if svq.size else 0.0,
            "n_zero_sv": int(np.sum(svq == 0)),
            "n_subnormal_sv": int(np.sum((svq < minn) & (svq > 0))) if minn else 0,
            "scales": sc, **st}


def fmt_err(v, w=10):
    return " " * w if v is None else f"{v:{w}.3e}"


def main():
    R["config"] = {"c": C_CERT, "cushion": CUSH, "safety": SAFE, "target_scale": TARGET,
                   "formats": FMT, "seeds_b": [1, 2, 3, 4, 5]}

    # ---------------------------------------------------------------- [A]
    print("=" * 100)
    print("[A] format definitions and the grid of each format")
    print("=" * 100)
    R["formats"] = {}
    for nm in FMT:
        d = fmt_check(nm)
        R["formats"][nm] = d
        print(f"  {nm}: {d['n_finite_codes']} finite codes, max|q(v)-v|/v on codes "
              f"{d['max_rel_deviation_on_codes']:.1e}; spacing just above 1 = "
              f"{d['spacing_above_one']:.4g} (2^-{FMT[nm]['mb']}); subnormal step "
              f"{d['subnormal_step']:.3g}; q(3*max) = {d['three_times_max']:.6g}; "
              f"q(min_sub/2) = {d['half_min_subnormal']:.3g}")
    ident = []
    print("\n  the format's grid against the paper's q at the matched p, on [ell, 1]:")
    for L in (0.15, 1e-2):
        x = np.logspace(np.log10(L), 0, 4000)
        for nm in FMT:
            p = FMT[nm]["p"]
            d = np.abs(qfmt(x, nm) - q(x, p))
            same = int(np.sum(d == 0))
            ident.append({"ell": L, "format": nm, "matched_p": p, "n_points": int(x.size),
                          "n_identical": same, "max_abs_diff": float(d.max()),
                          "max_abs_diff_above_min_normal":
                              float(np.max(np.abs(qfmt(x, nm) - q(x, p))[x >= FMT[nm]["minn"]]))})
            print(f"    ell={L:g} {nm} vs q(.,{p}): identical {same}/{x.size}, "
                  f"max|diff| {d.max():.2e}, above min normal "
                  f"{ident[-1]['max_abs_diff_above_min_normal']:.2e}")
    R["grid_identity"] = ident

    # ---------------------------------------------------------------- [B]
    print("\n" + "=" * 100)
    print("[B] the certificate under fp8 (d=64, K=6, five seeds, median)")
    print("=" * 100)
    R["law"] = {}
    ELL = (0.15, 0.1, 3.6e-2, 1.3e-2)
    print(f"  {'format':<10} {'ell':>7} {'median err':>11} {'range':>24} {'pred':>10} {'ratio':>6} "
          f"{'max|B|':>9} {'sat':>5} {'#sub':>5} {'#0sv':>5}")
    for nm in ("ideal_p5", "e4m3", "e4m3_x", "ideal_p4", "e5m2", "e5m2_x", "ideal_p3"):
        for L in ELL:
            cells = []
            for seed in (1, 2, 3, 4, 5):
                M, ref = mat(64, L, seed)
                co, _ = pe_coeffs(L, 6, CUSH, SAFE)
                if nm.endswith("_x"):
                    base = nm[:-2]
                    sx, sB, amaxB = auto_scales(co, M / n2(M), base, FMT[base]["maxv"])
                    cell = measure(co, M, ref, base, sx, sB)
                    cell["amax_B"] = amaxB
                else:
                    cell = measure(co, M, ref, nm)
                cell.update({"ell": L, "seed": seed, "format": nm})
                cells.append(cell)
                R["law"][f"{nm}_L{L:g}_s{seed}"] = cell
            e = np.array([c["err"] if c["err"] is not None else np.nan for c in cells])
            med = float(np.nanmedian(e))
            rec = {"ell": L, "format": nm, "median": med, "min": float(np.nanmin(e)),
                   "max": float(np.nanmax(e)), "pred": cells[0]["pred"],
                   "ratio_median": (cells[0]["pred"] / med) if med else None,
                   "maxB": max(c["maxB"] for c in cells),
                   "sat": max(c["sat"] for c in cells),
                   "n_sub": max(c["n_subnormal_sv"] for c in cells),
                   "n_zero": max(c["n_zero_sv"] for c in cells),
                   "n_diverged": int(np.sum(np.isnan(e)))}
            R.setdefault("law_summary", {})[f"{nm}_L{L:g}"] = rec
            print(f"  {nm:<10} {L:7g} {med:11.3e} [{np.nanmin(e):9.3e}, {np.nanmax(e):9.3e}] "
                  f"{cells[0]['pred']:10.3e} {(cells[0]['pred'] / med) if med else float('nan'):6.2f} "
                  f"{rec['maxB']:9.3g} {rec['sat']:5d} {rec['n_sub']:5d} {rec['n_zero']:5d}"
                  + (f"  ({rec['n_diverged']} saturate)" if rec["n_diverged"] else ""))

    # ---------------------------------------------------------------- [C]
    print("\n" + "=" * 100)
    print("[C] the exponent window: state scale s=2^k, e4m3, d=64, K=6, seed 2")
    print("=" * 100)
    R["window"] = {}
    for L in (0.1, 1e-2):
        M, ref = mat(64, L, 2)
        co, _ = pe_coeffs(L, 6, CUSH, SAFE)
        X0 = M / n2(M)
        sx0, sB, amaxB = auto_scales(co, X0, "e4m3", FMT["e4m3"]["maxv"])
        best = measure(co, M, ref, "e4m3", sx0, sB)
        row = {"ell": L, "amax_B": amaxB, "sB": sB, "e4m3_auto_err": best["err"], "sweep": {}}
        print(f"  ell={L:g}: max|B| exact {amaxB:.3g} -> intermediate scale 2^{np.log2(sB):.0f}; "
              f"e4m3 with per-tensor scales {best['err']:.3e}")
        for k in range(-5, 13):
            sx = 2.0 ** k
            cell = measure(co, M, ref, "e4m3", sx, sB)
            row["sweep"][f"k{k}"] = {"sx": sx, **{kk: cell[kk] for kk in
                                    ("err", "flush", "sat", "maxB", "n_zero_sv", "n_subnormal_sv")}}
            print(f"    s=2^{k:<3d} err {fmt_err(cell['err'])} flush {cell['flush']:5d} "
                  f"sat {cell['sat']:3d} sub {cell['n_subnormal_sv']:3d} 0sv {cell['n_zero_sv']:3d}  "
                  f"window [{sx*L:8.3g}, {sx:8.3g}]")
        R["window"][f"L{L:g}"] = row

    # ---------------------------------------------------------------- [D]
    print("\n" + "=" * 100)
    print("[D1] which tensor leaves the window first?  e4m3, d=64, K=6, seed 2")
    print("=" * 100)
    R["operand"] = {}
    print(f"  {'ell':>7} {'scales':>18} {'err':>10} {'pred':>10} {'ratio':>6} {'max|B|':>9} "
          f"{'sat':>5} {'flush':>6}")
    for L in (0.1, 1e-2):
        M, ref = mat(64, L, 2)
        co, _ = pe_coeffs(L, 6, CUSH, SAFE)
        X0 = M / n2(M)
        sx, sB, amaxB = auto_scales(co, X0, "e4m3", FMT["e4m3"]["maxv"])
        for mn, (a_, b_) in {"none": (None, None), "state only": (sx, None),
                             "intermediate only": (None, sB), "both": (sx, sB)}.items():
            cell = measure(co, M, ref, "e4m3", a_, b_)
            R["operand"][f"L{L:g}_{mn}"] = cell
            print(f"  {L:7g} {mn:>18} {fmt_err(cell['err'])} {cell['pred']:10.3e} "
                  f"{cell['ratio'] if cell['ratio'] else float('nan'):6.2f} {cell['maxB']:9.3g} "
                  f"{cell['sat']:5d} {cell['flush']:6d}")
        print(f"          (state scale 2^{np.log2(sx):.0f}, intermediate scale 2^{np.log2(sB):.0f}, "
              f"Intermediate max|B| exact {amaxB:.3g})")

    print("\n" + "=" * 100)
    print("[D2] a caller who forgets the normalisation: input of spectral norm 1e2, 1e3 (ell=1e-2)")
    print("=" * 100)
    R["input"] = {}
    M0, ref = mat(64, 1e-2, 2)
    co, _ = pe_coeffs(1e-2, 6, CUSH, SAFE)
    for mag in (1.0, 1e2, 1e3):
        cell = measure(co, M0 * mag, ref, "e4m3", X0_override=M0 * mag)
        R["input"][f"mag{mag:g}"] = cell
        print(f"  sigma_max(M) {mag:8g}: err {fmt_err(cell['err'])} pred {cell['pred']:10.3e} "
              f"ratio {cell['ratio'] if cell['ratio'] else float('nan'):5.2f} "
              f"max|B| {cell['maxB']:9.3g} sat {cell['sat']:6d}")

    # ---------------------------------------------------------------- [E]
    print("\n" + "=" * 100)
    print("[E] is the fp8 floor flat in K, and does it scale with the significand?  (d=64, seed 2)")
    print("=" * 100)
    R["kflat"] = {}
    for L in (0.1, 3.6e-2):
        M, ref = mat(64, L, 2)
        for nm in ("ideal_p5", "e4m3", "e4m3_x", "ideal_p4", "e5m2"):
            xs = []
            for K in (4, 6, 8, 10):
                co, _ = pe_coeffs(L, K, CUSH, SAFE)
                if nm.endswith("_x"):
                    base = nm[:-2]
                    sx, sB, _ = auto_scales(co, M / n2(M), base, FMT[base]["maxv"])
                    cell = measure(co, M, ref, base, sx, sB)
                else:
                    cell = measure(co, M, ref, nm)
                R["kflat"][f"{nm}_L{L:g}_K{K}"] = cell
                xs.append(cell["err"])
            print(f"  ell={L:<6g} {nm:<10}: " + "  ".join(
                f"K={K}:{fmt_err(x)}" for K, x in zip((4, 6, 8, 10), xs)))

    # ---------------------------------------------------------------- [F]
    print("\n" + "=" * 100)
    print("[F] the boundary in ell (d=64, K=6, seed 3)")
    print("=" * 100)
    R["boundary"] = {}
    print(f"  {'ell':>9} {'log2':>7} {'ideal_p5':>10} {'e4m3':>10} {'e4m3/ideal':>10} "
          f"{'e4m3_x':>10} {'x/ideal':>8} {'min sv':>9} {'#sub':>5} {'#zero':>6}")
    for L in np.logspace(-1, np.log10(3e-5), 9):
        L = float(L)
        M, ref = mat(64, L, 3)
        co, _ = pe_coeffs(L, 6, CUSH, SAFE)
        X0 = M / n2(M)
        sv = np.linalg.svd(X0, compute_uv=False)
        ci = measure(co, M, ref, "ideal_p5")
        cr = measure(co, M, ref, "e4m3")
        sxa, sBa, _ = auto_scales(co, X0, "e4m3", FMT["e4m3"]["maxv"])
        cx = measure(co, M, ref, "e4m3", sxa, sBa)
        svq = qfmt(sv, "e4m3")
        nsub = int(np.sum((svq < FMT["e4m3"]["minn"]) & (svq > 0)))
        nzero = int(np.sum(svq == 0))
        R["boundary"][f"L{L:.2e}"] = {"ell": L, "ideal_p5": ci["err"], "e4m3": cr["err"],
                                      "e4m3_x": cx["err"], "pred": ci["pred"],
                                      "min_sv": float(sv.min()), "n_subnormal": nsub,
                                      "n_zero": nzero, "scales_x": {"sx": sxa, "sB": sBa}}
        print(f"  {L:9.2e} {np.log2(L):7.2f} {fmt_err(ci['err'])} {fmt_err(cr['err'])} "
              f"{(cr['err'] / ci['err']) if ci['err'] else float('nan'):10.2f} {fmt_err(cx['err'])} "
              f"{(cx['err'] / ci['err']) if ci['err'] else float('nan'):8.2f} {sv.min():9.3g} "
              f"{nsub:5d} {nzero:6d}")

    # ---------------------------------------------------------------- [G]
    print("\n" + "=" * 100)
    print("[G] the Muon shape 128x32, K=6, seed 1")
    print("=" * 100)
    R["rect"] = {}
    for L in (0.15, 1e-2, 1e-3):
        M, ref = mat(32, L, 1, m=128)
        co, _ = pe_coeffs(L, 6, CUSH, SAFE)
        for nm in ("ideal_p5", "e4m3", "e4m3_x", "ideal_p4", "e5m2", "e5m2_x"):
            if nm.endswith("_x"):
                base = nm[:-2]
                sx, sB, amaxB = auto_scales(co, M / n2(M), base, FMT[base]["maxv"])
                cell = measure(co, M, ref, base, sx, sB)
                cell["amax_B"] = amaxB
            else:
                cell = measure(co, M, ref, nm)
            R["rect"][f"{nm}_L{L:g}"] = cell
            print(f"  ell={L:<6g} {nm:<10} err {fmt_err(cell['err'])} pred {cell['pred']:10.3e} "
                  f"ratio {cell['ratio'] if cell['ratio'] else float('nan'):5.2f} "
                  f"max|B| {cell['maxB']:9.3g} sat {cell['sat']:4d} 0sv {cell['n_zero_sv']:3d}")

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 100)
    print("summary")
    print("=" * 100)
    S = {}
    for nm in ("ideal_p5", "e4m3", "e4m3_x", "ideal_p4", "e5m2", "e5m2_x", "ideal_p3"):
        rr, by = [], {}
        for L in ELL:
            r = R["law_summary"][f"{nm}_L{L:g}"]
            if r["ratio_median"]:
                rr.append(r["ratio_median"])
            by[f"{L:g}"] = r["ratio_median"]
        S[nm] = {"ratio_by_ell": by, "ratio_median": float(np.median(rr)) if rr else None,
                 "ratio_min": float(min(rr)) if rr else None,
                 "ratio_max": float(max(rr)) if rr else None}
        print(f"  {nm:<9} predicted/measured by ell: " +
              ", ".join(f"{L}: {v:.2f}" for L, v in S[nm]["ratio_by_ell"].items() if v is not None))
    R["summary"] = S
    out = os.path.join(HERE, "pilot15_results.json")
    with open(out, "w") as f:
        json.dump(R, f, indent=1, default=float)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
