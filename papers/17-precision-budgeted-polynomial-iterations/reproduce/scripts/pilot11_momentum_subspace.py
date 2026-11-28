#!/usr/bin/env python3
"""
Pilot 11 - the first-order polar formula on the *real* momentum spectra that a
Muon update actually produces, and the rectangular term that they expose.

Pilot 10 verified, on square matrices with a logspace spectrum,

    polar(Mhat + E) - polar(Mhat) = U [ (F_ij - F_ji)/(s_i + s_j) ] V^T + O(||E||^2),
    F = U^T E V,   Mhat = U diag(s) V^T.

A Muon update is not square: for a hidden layer of shape H x D_in the momentum is
128 x 32.  For a rectangular Mhat (m >= n, full column rank) the same expansion
carries a second first-order term, because the column space itself rotates out of
range(U):

    polar(Mhat + E) - polar(Mhat)
        = U [ (F_ij - F_ji)/(s_i + s_j) ] V^T          (in-subspace rotation)
        + (I_m - U U^T) E (Mhat^T Mhat)^{-1/2}          (column-space rotation)
        + O(||E||^2).

The second term vanishes identically for square Mhat, which is why 54 synthetic
square cells in pilot 10 could not see it.  This script

  [A] rebuilds the exact-arithmetic Muon trajectory of pilot8 (2-layer MLP,
      hidden 128, D_in 32, 500 steps, momentum 0.95, Nesterov, exact polar) and
      checks the replay against pilot8's frozen sequence;
  [B] for all 500 momentum matrices and p in {8,11,16} compares three predictions
      with the measured subspace error: the square-only formula, the two-term
      formula, and (as a control) the same formulas on square matrices;
  [C] checks the residual of the two-term formula against u (the O(u^2) term);
  [D] decomposes C*ell -- the participation that makes the measured exponent
      -0.73 a chord rather than a law -- into a mass-weighted pair damping, a
      spectral concentration and the size of the skew / projected perturbations,
      and tests a closed form built from the spectrum alone;
  [E] per-slot gains of the subspace channel on the real spectra.

Kernel model as in pilots 1-10: X and B are stored with p significant bits,
products are accumulated in float64.  Exact arithmetic uses p >= 30.
"""
import json
import os
import sys
import time
import numpy as np

from pilot7_pert_aware_design import q, pe_coeffs, make_M
from pilot8_muon_mlp import run_coupled, make_task

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "pilot11_results.json")
NPZ = os.path.join(HERE, "pilot11_momentum.npz")
EXACT = 60
CUSH, SAFE = 0.02, 1.01
ELL_MED, K_FROZEN = 0.14994553643082875, 8      # pilot8 frozen replay settings


def n2(X):
    return float(np.linalg.norm(X, 2))


def polar_svd(X):
    w, s, vt = np.linalg.svd(X, full_matrices=False)
    return w @ vt, w, s, vt


def polar_safe(X):
    if not np.all(np.isfinite(X)):
        return None
    try:
        return polar_svd(X)[0]
    except np.linalg.LinAlgError:
        return None


def orbit(co, X0, p0, ps):
    """p0 = precision of the input cast, ps[t] = precision of step t (>=30 exact)."""
    X = q(X0, p0)
    traj = [X]
    for (a, b, c), p in zip(co, ps):
        A = X @ X.T
        B = q(b * A + c * (A @ A), p)
        X = q(a * X + B @ X, p)
        traj.append(X)
    return X, traj


def first_order(X0, E):
    """Both first-order terms, their norms, and the factors of C*ell."""
    m, n = X0.shape
    U, s, Vt = np.linalg.svd(X0, full_matrices=False)
    V = Vt.T
    F = U.T @ E @ Vt.T
    Ft = F - F.T
    G = Ft / (s[:, None] + s[None, :])
    P1 = U @ G @ Vt                                    # in-subspace rotation (Vt = V^T)
    if m >= n:
        H = V @ np.diag(1.0 / s) @ V.T
        P2 = (np.eye(m) - U @ U.T) @ E @ H              # column-space rotation
    else:
        K = U @ np.diag(1.0 / s) @ U.T
        P2 = K @ E @ (np.eye(n) - V @ V.T)              # row-space rotation
    ell = float(s[-1])
    a = 2.0 * ell / (s[:, None] + s[None, :])          # in (0,1], 1 at the trailing pair
    mass = float((Ft ** 2).sum())
    Pi = float(((Ft ** 2) * a ** 2).sum() / mass) if mass > 0 else 0.0
    p1 = n2(P1)
    r = float((G ** 2).sum()) / p1 ** 2 if p1 > 0 else float("nan")
    return {"P1": P1, "P2": P2, "pred1": p1, "pred2": n2(P2), "pred": n2(P1 + P2),
            "ell": ell, "Pi": Pi, "r": r, "Pi_unif": float((a ** 2).sum() / (n * (n - 1))),
            "skew_fro": float(np.sqrt(mass)), "F_fro": float(np.linalg.norm(F, "fro")),
            "E_fro": float(np.linalg.norm(E, "fro")), "X0_fro": float(np.linalg.norm(X0, "fro")),
            "s": s, "U": U, "Vt": Vt}


def direrr(D_pred, D_true):
    return float(np.linalg.norm(D_pred - D_true) / n2(D_true))


# ------------------------------------------------------------------ [A]
def part_A(R):
    print("=" * 104)
    print("[A] exact Muon trajectory (pilot8 replay): 500 momentum matrices of the hidden layer")
    print("=" * 104)
    Xtr, ytr, _, _ = make_task()
    t0 = time.time()
    exact, _ = run_coupled("fixed8", None, 0, L=0.15, collect=True, Xtr=Xtr, ytr=ytr)
    upds = [u_.astype(np.float32).astype(float) for u_ in exact["upds"]]
    meta = {"n": len(upds), "shape": list(upds[0].shape),
            "exact_best_val": exact["best_val"], "exact_acc": exact["acc"],
            "exact_final_val": exact["final_val"], "seconds": time.time() - t0}
    refs = []
    for M in upds:
        pol, U, s, Vt = polar_svd(M)
        refs.append((pol, s / s[0]))
    lls = np.array([sv[-1] for _, sv in refs])
    meta.update({"ell_med": float(np.median(lls)), "ell_p10": float(np.percentile(lls, 10)),
                 "ell_p90": float(np.percentile(lls, 90)),
                 "ell_min": float(lls.min()), "ell_max": float(lls.max())})
    pilot8 = {"n": 500, "ell_med": 0.14994553643082875, "ell_p10": 0.05821005469486956,
              "ell_p90": 0.17551465387045193, "exact_best_val": 0.6534074707634926,
              "exact_acc": 0.762}
    meta["pilot8_meta"] = pilot8
    meta["replay_matches_pilot8"] = {
        "n": meta["n"] == pilot8["n"],
        "ell_med": abs(meta["ell_med"] - pilot8["ell_med"]) < 1e-9,
        "exact_best_val": abs(meta["exact_best_val"] - pilot8["exact_best_val"]) < 1e-9,
        "exact_acc": abs(meta["exact_acc"] - pilot8["exact_acc"]) < 1e-9}
    np.savez_compressed(NPZ, upds=np.stack([u_.astype(np.float32) for u_ in upds]))
    R["trajectory"] = meta
    print(f"  {meta['n']} steps of {meta['shape']}  ({meta['seconds']:.1f}s)")
    print(f"  ell(sigma_min/sigma_max): median {meta['ell_med']:.4f}, "
          f"p10 {meta['ell_p10']:.4f}, p90 {meta['ell_p90']:.4f}, "
          f"range {meta['ell_min']:.4f}-{meta['ell_max']:.4f}")
    print(f"  exact run: best val {meta['exact_best_val']:.4f}, acc {meta['exact_acc']:.3f}")
    print(f"  replay identical to pilot8 frozen sequence: {meta['replay_matches_pilot8']}")
    return upds, refs


# ------------------------------------------------------------------ [B]
def part_B(R, upds, refs):
    print("\n" + "=" * 104)
    print("[B] first-order formula on the 500 real momentum spectra (128x32, tall)")
    print("=" * 104)
    co, _ = pe_coeffs(ELL_MED, K_FROZEN, CUSH, SAFE)
    R["real_spectra"] = {}
    for p in (8, 11, 16):
        u = 2.0 ** -p
        rows = []
        t0 = time.time()
        for (ref, sv), M in zip(refs, upds):
            X0 = M / n2(M)
            E = q(X0, p) - X0
            fo = first_order(X0, E)
            ell = fo["ell"]
            Xi = q(X0, p)
            dtrue = polar_svd(Xi)[0] - ref
            meas_in = n2(dtrue)
            Xk, _ = orbit(co, X0, p, [p] * K_FROZEN)
            polk = polar_safe(Xk)
            meas_tot = n2(polk - ref) if polk is not None else float("nan")
            rows.append({"ell": ell, "u": u,
                         "pred": fo["pred"], "pred1": fo["pred1"], "pred2": fo["pred2"],
                         "meas_in": meas_in, "meas_tot": meas_tot,
                         "ratio": fo["pred"] / meas_in,
                         "ratio1": fo["pred1"] / meas_in,
                         "ratio_tot": fo["pred"] / meas_tot if meas_tot else None,
                         "in_over_tot": meas_in / meas_tot if meas_tot else None,
                         "dir_rel": direrr(fo["P1"] + fo["P2"], dtrue),
                         "dir_rel1": direrr(fo["P1"], dtrue),
                         "C": meas_in / u, "C_pred": fo["pred"] / u,
                         "C_ell": meas_in * ell / u,
                         "C_ell_pred": fo["pred"] * ell / u,
                         "C_ell_pred1": fo["pred1"] * ell / u,
                         "term_ratio": fo["pred2"] / fo["pred1"],
                         "E2_u": n2(E) / u, "E_fro_u": fo["E_fro"] / u,
                         "Pi": fo["Pi"], "r": fo["r"], "Pi_unif": fo["Pi_unif"],
                         "skew_fro_u": fo["skew_fro"] / u, "X0_fro": fo["X0_fro"]})
        a = {k: np.array([r[k] for r in rows], float) for k in rows[0]}
        ok = np.isfinite(a["ratio"])
        qq = lambda k, pc: float(np.percentile(a[k][np.isfinite(a[k])], pc))
        s = {"rows": rows, "seconds": time.time() - t0,
             "ratio_median": qq("ratio", 50), "ratio_p10": qq("ratio", 10),
             "ratio_p90": qq("ratio", 90), "ratio_min": qq("ratio", 0), "ratio_max": qq("ratio", 100),
             "ratio1_median": qq("ratio1", 50), "ratio1_p10": qq("ratio1", 10),
             "ratio1_p90": qq("ratio1", 90),
             "frac_10pct": float(np.mean(np.abs(np.log(a["ratio"][ok])) / np.log(10) <= 0.0414)),
             "frac_2x": float(np.mean(np.abs(np.log2(a["ratio"][ok])) <= 1.0)),
             "ratio_tot_median": qq("ratio_tot", 50),
             "ratio_tot_p10": qq("ratio_tot", 10), "ratio_tot_p90": qq("ratio_tot", 90),
             "in_over_tot_median": qq("in_over_tot", 50),
             "in_over_tot_min": qq("in_over_tot", 0), "in_over_tot_max": qq("in_over_tot", 100),
             "dir_rel_median": qq("dir_rel", 50), "dir_rel_p90": qq("dir_rel", 90),
             "dir_rel1_median": qq("dir_rel1", 50),
             "term_ratio_median": qq("term_ratio", 50),
             "term_ratio_min": qq("term_ratio", 0), "term_ratio_max": qq("term_ratio", 100),
             "C_median": qq("C", 50), "C_ell_median": qq("C_ell", 50),
             "C_ell_pred_median": qq("C_ell_pred", 50),
             "C_ell_pred1_median": qq("C_ell_pred1", 50),
             "E2_u_median": qq("E2_u", 50)}
        R["real_spectra"][str(p)] = s
        print(f"\n  p={p:2d}  u={u:.2e}  ({s['seconds']:.1f}s)")
        print(f"    two-term   pred/meas median {s['ratio_median']:.3f} "
              f"[p10 {s['ratio_p10']:.3f}, p90 {s['ratio_p90']:.3f}, "
              f"range {s['ratio_min']:.3f}-{s['ratio_max']:.3f}]  "
              f"within 10%: {100 * s['frac_10pct']:.0f}%, within 2x: {100 * s['frac_2x']:.0f}%")
        print(f"    square-only term1 pred/meas median {s['ratio1_median']:.3f} "
              f"[{s['ratio1_p10']:.3f}, {s['ratio1_p90']:.3f}]")
        print(f"    ||term2||/||term1|| median {s['term_ratio_median']:.2f} "
              f"[{s['term_ratio_min']:.2f}, {s['term_ratio_max']:.2f}]")
        print(f"    direction error: two-term {s['dir_rel_median']:.3f} "
              f"(p90 {s['dir_rel_p90']:.3f}), term1 only {s['dir_rel1_median']:.3f}")
        print(f"    full orbit (K={K_FROZEN}): pred/meas median {s['ratio_tot_median']:.3f} "
              f"[{s['ratio_tot_p10']:.3f}, {s['ratio_tot_p90']:.3f}]; "
              f"input cast / full orbit {s['in_over_tot_median']:.3f} "
              f"[{s['in_over_tot_min']:.3f}, {s['in_over_tot_max']:.3f}]")
        print(f"    C=sub/u median {s['C_median']:.2f}, C*ell median {s['C_ell_median']:.4f} "
              f"(pred {s['C_ell_pred_median']:.4f}, square-only {s['C_ell_pred1_median']:.4f})")

    # control: the same two formulas on synthetic matrices of three shapes
    print("\n  control - the same two first-order terms on three shapes (p=8, logspace spectrum)")
    R["shape_control"] = {}
    for (m, n) in ((128, 32), (64, 64), (32, 128)):
        for L in (1e-1, 1e-2, 1e-3):
            rs = np.random.RandomState(1)
            U, _ = np.linalg.qr(rs.randn(m, min(m, n)))
            V, _ = np.linalg.qr(rs.randn(n, min(m, n)))
            sv = np.logspace(0, np.log10(L), min(m, n))
            M = (U * sv) @ V.T
            ref = U @ V.T
            X0 = M / n2(M)
            E = q(X0, 8) - X0
            fo = first_order(X0, E)
            dtrue = polar_svd(q(X0, 8))[0] - ref
            meas = n2(dtrue)
            R["shape_control"][f"{m}x{n}_L{L:g}"] = {
                "meas": meas, "pred": fo["pred"], "pred1": fo["pred1"], "pred2": fo["pred2"],
                "ratio": fo["pred"] / meas, "ratio1": fo["pred1"] / meas,
                "dir_rel": direrr(fo["P1"] + fo["P2"], dtrue),
                "dir_rel1": direrr(fo["P1"], dtrue)}
            c = R["shape_control"][f"{m}x{n}_L{L:g}"]
            print(f"    {m:3d}x{n:<3d} ell={L:<7g} meas {meas:.3e} two-term/meas "
                  f"{c['ratio']:.3f} term1/meas {c['ratio1']:.3f} "
                  f"|term2|/|term1| {(c['pred2'] / c['pred1'] if c['pred1'] else 0):.2f} "
                  f"direrr {c['dir_rel']:.3f} vs {c['dir_rel1']:.3f}")


# ------------------------------------------------------------------ [C]
def part_C(R, upds, refs):
    print("\n" + "=" * 104)
    print("[C] residual of the two-term formula against u (the O(u^2) term)")
    print("=" * 104)
    idx = list(range(0, len(upds), max(1, len(upds) // 60)))[:60]
    per_p = {}
    for p in (6, 8, 11, 14, 16):
        u = 2.0 ** -p
        res, rat, sc = [], [], []
        for i in idx:
            ref, sv = refs[i]
            M = upds[i]
            X0 = M / n2(M)
            E = q(X0, p) - X0
            fo = first_order(X0, E)
            meas = n2(polar_svd(q(X0, p))[0] - ref)
            res.append(abs(meas - fo["pred"]))
            rat.append(fo["pred"] / meas)
            sc.append(n2(E) ** 2 / fo["ell"])
        res = np.array(res)
        per_p[str(p)] = {"u": u, "resid_median": float(np.median(res)),
                         "resid_over_u2": float(np.median(res) / u ** 2),
                         "resid_over_scale": float(np.median(res / np.array(sc))),
                         "ratio_median": float(np.median(rat)),
                         "ratio_min": float(np.min(rat)), "ratio_max": float(np.max(rat))}
        c = per_p[str(p)]
        print(f"  p={p:2d} u={u:.1e}: |pred-meas| median {c['resid_median']:.2e} "
              f"= {c['resid_over_u2']:.3g} u^2 = {c['resid_over_scale']:.3f} * ||E||^2/ell  "
              f"pred/meas median {c['ratio_median']:.3f} [{c['ratio_min']:.3f}, {c['ratio_max']:.3f}]")
    ps = np.array([6, 8, 11, 14, 16], float)
    lg = np.array([per_p[str(int(p))]["resid_median"] for p in ps])
    slope = float(-np.polyfit(np.log(2.0 ** -ps), np.log(lg), 1)[0])
    R["residual"] = {"per_p": per_p, "loglog_slope_vs_u": slope}
    print(f"  log-log slope of |pred-meas| against u: {slope:.2f}  (2 = second order)")


# ------------------------------------------------------------------ [D]
def part_D(R, upds, refs):
    print("\n" + "=" * 104)
    print("[D] C*ell: participation decomposition (synthetic square and real rectangular)")
    print("=" * 104)
    R["participation"] = {"synthetic": {}, "real": {}}
    for p in (8, 12):
        u = 2.0 ** -p
        for L in (1e-2, 1e-3, 1e-4):
            rows = []
            for seed in (1, 2, 3, 4):
                M, ref = make_M(64, L, seed)
                X0 = M / n2(M)
                E = q(X0, p) - X0
                fo = first_order(X0, E)
                meas = n2(polar_svd(q(X0, p))[0] - ref)
                rows.append({"C_ell": meas * fo["ell"] / u, "C_ell_pred": fo["pred"] * fo["ell"] / u,
                             "Pi": fo["Pi"], "r": fo["r"], "Pi_unif": fo["Pi_unif"],
                             "skew_fro_u": fo["skew_fro"] / u, "F_fro_u": fo["F_fro"] / u,
                             "X0_fro": fo["X0_fro"], "ell": fo["ell"]})
            agg = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
            agg["n_seeds"] = len(rows)
            agg["rows"] = rows
            agg["C_ell_per_seed"] = [r["C_ell"] for r in rows]
            agg["identity"] = 0.5 * agg["skew_fro_u"] * np.sqrt(agg["Pi"] / agg["r"])
            agg["model_unif"] = 0.5 * agg["skew_fro_u"] * np.sqrt(agg["Pi_unif"] / agg["r"])
            agg["model_unif_iso"] = (0.5 * np.sqrt(2.0) * agg["X0_fro"] / np.sqrt(12.0)
                                     * np.sqrt(agg["Pi_unif"] / agg["r"]))
            R["participation"]["synthetic"][f"d64_p{p}_L{L:g}"] = agg
            print(f"  square d=64 p={p:2d} ell={L:<7g} C*ell meas {agg['C_ell']:.4f} "
                  f"pred {agg['C_ell_pred']:.4f} | Pi {agg['Pi']:.4f} Pi_unif {agg['Pi_unif']:.4f} "
                  f"r {agg['r']:.2f} ||skewF||_F/u {agg['skew_fro_u']:.2f} "
                  f"| closed forms {agg['identity']:.4f} {agg['model_unif']:.4f} "
                  f"{agg['model_unif_iso']:.4f}")
    rows = R["real_spectra"]["8"]["rows"]
    out = {}
    for key in ("C_ell", "C_ell_pred", "C_ell_pred1", "Pi", "r", "Pi_unif", "E2_u", "term_ratio"):
        v = np.array([r[key] for r in rows], float)
        out[key] = {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
                    "p90": float(np.percentile(v, 90))}
    # does the projected term explain the real C*ell?  Phi2 = ell*||P2||/u ,
    # normalised by ||E||/u it is a participation in (0, sqrt(n)]
    phi2 = np.array([r["pred2"] * r["ell"] / r["u"] for r in rows])
    norm2 = np.array([r["pred2"] * r["ell"] / (r["E2_u"] * r["u"]) for r in rows])
    out["phi2"] = {"median": float(np.median(phi2)), "p10": float(np.percentile(phi2, 10)),
                   "p90": float(np.percentile(phi2, 90))}
    out["phi2_over_E2"] = {"median": float(np.median(norm2)),
                           "p10": float(np.percentile(norm2, 10)),
                           "p90": float(np.percentile(norm2, 90))}
    R["participation"]["real"] = out
    print(f"\n  real spectra (p=8, n={len(rows)}): C*ell meas {out['C_ell']['median']:.4f} "
          f"[{out['C_ell']['p10']:.4f}, {out['C_ell']['p90']:.4f}], two-term pred "
          f"{out['C_ell_pred']['median']:.4f}, square-only pred {out['C_ell_pred1']['median']:.4f}")
    print(f"    projected-term participation ell*||P2||/||E||_2: median "
          f"{out['phi2_over_E2']['median']:.3f} [{out['phi2_over_E2']['p10']:.3f}, "
          f"{out['phi2_over_E2']['p90']:.3f}]  -> C*ell is set by ||E||/u times this number")
    print(f"    ||E||_2/u median {out['E2_u']['median']:.2f}, ||term2||/||term1|| median "
          f"{out['term_ratio']['median']:.2f}")


# ------------------------------------------------------------------ [E]
def part_E(R, upds, refs):
    print("\n" + "=" * 104)
    print("[E] per-slot gains on the real spectra (which rounding makes the floor)")
    print("=" * 104)
    co, _ = pe_coeffs(ELL_MED, K_FROZEN, CUSH, SAFE)
    p = 8
    u = 2.0 ** -p
    idx = list(range(0, len(upds), max(1, len(upds) // 100)))[:100]
    slots = ["input"] + [f"step{t}" for t in range(K_FROZEN)]
    gains = {s: [] for s in slots}
    tog = {"both0": [], "all": []}
    for i in idx:
        ref, sv = refs[i]
        M = upds[i]
        X0 = M / n2(M)
        base = [p] * K_FROZEN
        for t, name in enumerate(slots):
            if t == 0:
                Xk, _ = orbit(co, X0, EXACT, base)
            else:
                ps = list(base)
                ps[t - 1] = EXACT
                Xk, _ = orbit(co, X0, p, ps)
            pol = polar_safe(Xk)
            gains[name].append(n2(pol - ref) if pol is not None else float("nan"))
        Xk, _ = orbit(co, X0, EXACT, [EXACT] + base[1:])
        tog["both0"].append(n2(polar_safe(Xk) - ref))
        Xk, _ = orbit(co, X0, p, base)
        tog["all"].append(n2(polar_safe(Xk) - ref))
    prof = {s: {"median_err": float(np.median(gains[s])),
                "gain_u": float(np.median(gains[s]) / u)} for s in slots}
    allm = np.array(tog["all"])
    for s in slots:
        g = np.array(gains[s])
        prof[s]["contribution_u"] = float(np.median(np.sqrt(np.maximum(allm ** 2 - g ** 2, 0.0))) / u)
    prof["_all_p"] = {"median_err": float(np.median(tog["all"]))}
    prof["_input_and_step0_exact"] = {"median_err": float(np.median(tog["both0"])),
                                      "gain": float(np.median(tog["all"]) / np.median(tog["both0"]))}
    R["slot_profile_real"] = {"p": p, "n": len(idx), "K": K_FROZEN, "profile": prof}
    print(f"  p={p}, n={len(idx)}, K={K_FROZEN}, design window {ELL_MED:.3f}   (units of u)")
    print("  " + " ".join(f"{s:>7s}" for s in ["all"] + slots))
    print("  " + " ".join(f"{(prof['_all_p']['median_err'] / u if s == 'all' else prof[s]['gain_u']):7.2f}"
                          for s in ["all"] + slots))
    print(f"  making input + step 0 exact: {prof['_input_and_step0_exact']['gain']:.2f}x")
    print("  per-slot contribution in quadrature (median, units of u): "
          + " ".join(f"{s} {prof[s]['contribution_u']:.2f}" for s in slots))


def main():
    stages = sys.argv[1] if len(sys.argv) > 1 else "ABCDE"
    t0 = time.time()
    if os.path.exists(OUT) and os.path.exists(NPZ):
        with open(OUT) as f:
            R = json.load(f)
        data = np.load(NPZ)
        upds = [d.astype(float) for d in data["upds"]]
        refs = []
        for M in upds:
            pol, U, s, Vt = polar_svd(M)
            refs.append((pol, s / s[0]))
        print(f"[resume] {len(upds)} momentum matrices loaded")
    else:
        R = {"config": {"ELL_MED": ELL_MED, "K": K_FROZEN, "cushion": CUSH, "safety": SAFE,
                        "kernel": "X and B stored in p significant bits, products in float64",
                        "momenta": "numpy 2-layer MLP, hidden 128 x D_in 32, Muon momentum 0.95 Nesterov"}}
        upds, refs = part_A(R)
        with open(OUT, "w") as f:
            json.dump(R, f, indent=1, default=float)
    if "B" in stages:
        part_B(R, upds, refs)
    if "C" in stages:
        part_C(R, upds, refs)
    if "D" in stages:
        part_D(R, upds, refs)
    if "E" in stages:
        part_E(R, upds, refs)
    with open(OUT, "w") as f:
        json.dump(R, f, indent=1, default=float)
    print(f"\ntotal {time.time() - t0:.1f}s   wrote {os.path.basename(OUT)}")


if __name__ == "__main__":
    main()
