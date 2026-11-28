#!/usr/bin/env python3
"""
Pilot 10 - an analytic account of the singular-subspace floor.

Pilot 8b decomposed the p-bit floor of a polar iteration into a singular-value
part and a singular-subspace part, found the latter to be 86-100% of the floor,
and measured C(ell) = sub_err/u ~ ell^-0.73 (p=8, d=64).  This script turns the
observation into theory and measures the theory's accuracy.

 [A] invariance.  The exact step p(X) = aX + (b I + c XX^T) XX^T acts on singular
     values only: for X = U diag(s) V^T, p(X) = U diag(p(s)) V^T.  Hence
         polar(P_K(X)) = polar(X)   for every full-rank X and every K,
     i.e. the exact iteration cannot change the singular subspace.  Consequence:
     the subspace part of the floor is fixed at the quantised start and cannot be
     reduced by K, by the design, or by a step schedule.  Measured two ways: the
     exact orbit's polar factor is constant, and an error injected on purpose as a
     rotated start is transported unchanged by every exact step.

 [B] first-order perturbation theory.  With X0 = q(Mhat,p) = Mhat + E and
     Mhat = U diag(s) V^T, the polar factor's derivative is
         polar(X0) - polar(Mhat) = U [ (F_ij - F_ji)/(s_i + s_j) ] V^T + O(||E||^2),
     F = U^T E V  (the symmetric part of the perturbation does not move the polar
     factor at all; the skew part is damped by 1/(s_i+s_j)).  Tested over
     ell x p x d, against the measured subspace distance.

 [C] the exponent.  Measure C(ell) densely at two precisions, and compare with
     (i) the first-order prediction and (ii) the cap sub_err <= 2 that any
     subspace distance obeys.  Decides whether -0.73 is a law or an artefact of
     fitting across the cap and across the large-ell regime.

 [D] a computable upper bound with no SVD: sub_err <= ||E||_2/sigma_min(Mhat),
     with ||E||_2 by power iteration and sigma_min taken from the design.  Reports
     validity and slack over the whole grid.

 [E] a negative result.  No function of X's singular values can certify the
     subspace part: a random orthogonal matrix has XX^T = I exactly and subspace
     distance 2 from the reference.  Measured correlation of the subspace error
     with the internal quantity ||XX^T - I|| confirms the argument on the grid.

Kernel model as in pilots 1-9: d x d matrix, spectrum logspace(0, log10 ell),
X and B stored in p significant bits, products accumulated in float64.
"""
import json
import os
import numpy as np

from pilot7_pert_aware_design import q, pe_coeffs, make_M

HERE = os.path.dirname(os.path.abspath(__file__))
R = {}
CUSH, SAFE = 0.02, 1.01
EXACT = 60


def polar(X):
    w, _, vt = np.linalg.svd(X, full_matrices=False)
    return w @ vt


def polar_safe(X):
    """None if the orbit left the finite range (bf16-style divergence)."""
    if not np.all(np.isfinite(X)):
        return None
    try:
        return polar(X)
    except np.linalg.LinAlgError:
        return None


def polarsvd(X):
    w, s, vt = np.linalg.svd(X, full_matrices=False)
    return w @ vt, w, s, vt


def n2(X):
    return float(np.linalg.norm(X, 2))


def matvec_norm2(E, iters=2000, seed=3):
    """||E||_2 using matrix-vector products only (no SVD): power iteration on E^T E."""
    rs = np.random.RandomState(seed)
    x = rs.randn(E.shape[1])
    x /= np.linalg.norm(x)
    for _ in range(iters):
        z = E.T @ (E @ x)
        nz = np.linalg.norm(z)
        if nz == 0:
            return 0.0
        x = z / nz
    return float(np.linalg.norm(E @ x))


def orbit(co, X0, ps):
    """one p-bit step per entry of ps; ps[t] >= 30 means exact."""
    X = X0
    traj = [X]
    for (a, b, c), p in zip(co, ps):
        A = X @ X.T
        B = q(b * A + c * (A @ A), p)
        X = q(a * X + B @ X, p)
        traj.append(X)
    return X, traj


def start(M, p):
    Mn = M / n2(M)
    return Mn, q(Mn, p)


# ------------------------------------------------------------------ [A]
def part_A():
    print("=" * 104)
    print("[A] invariance: the exact iteration cannot move the singular subspace")
    print("=" * 104)
    R["invariance"] = {}
    for L, K in ((1e-1, 6), (1e-3, 8), (1e-4, 7)):
        M, ref = make_M(64, L, 1)
        Mn, _ = start(M, 8)
        co, _ = pe_coeffs(L, K, CUSH, SAFE)
        X, traj = orbit(co, Mn, [EXACT] * K)          # exact arithmetic orbit
        devs = [n2(polar(Xt) - ref) for Xt in traj]
        print(f"  ell={L:<7g} K={K}: ||polar(X_t) - polar(X_0)||_2 at t=0..K : "
              + " ".join(f"{v:.1e}" for v in devs))
        print(f"            exact final E_K = {n2(traj[-1] - ref):.2e}, "
              f"max polar deviation = {max(devs):.1e}")
        R["invariance"][f"L{L:g}"] = {"polar_dev": devs, "E_K": n2(traj[-1] - ref)}

    print("\n  [A2] an injected subspace error is transported unchanged (exact arithmetic)")
    print(f"      {'ell':>8s} {'theta':>7s} " + " ".join(f"{'t='+str(t):>9s}" for t in range(5)))
    R["transport"] = {}
    for L in (1e-2, 1e-3):
        M, ref = make_M(64, L, 1)
        Mn = M / n2(M)
        U, s, Vt = np.linalg.svd(Mn, full_matrices=False)
        co, _ = pe_coeffs(L, 6, CUSH, SAFE)
        for th in (1e-2, 1e-1):
            # rotate the two trailing right singular directions of the START
            Rm = np.eye(64)
            Rm[-2:, -2:] = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            X0 = U @ np.diag(s) @ Rm.T @ Vt
            X, traj = orbit(co, X0, [EXACT] * 6)
            ds = [n2(polar(Xt) - ref) for Xt in traj]
            R["transport"][f"L{L:g}_th{th:g}"] = ds
            print(f"      {L:8g} {th:7.0e} " + " ".join(f"{v:9.2e}" for v in ds[:5]))


# ------------------------------------------------------------------ [B]
def part_B():
    print("\n" + "=" * 104)
    print("[B] first-order formula  polar(X0)-polar(M) = U[(F-F^T)_ij/(s_i+s_j)]V^T,  F = U^T E V")
    print("=" * 104)
    print(f"  {'d':>4s} {'ell':>8s} {'p':>3s} {'C=sub/u':>9s} {'C_pred':>9s} {'pred/meas':>9s} "
          f"{'tail_C':>8s} {'||E||/u':>8s} {'||E||/ell/u':>11s}")
    R["first_order"] = {}
    for d in (32, 64, 128):
        for L in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4):
            for p in (8, 11, 16):
                u = 2.0 ** -p
                M, ref = make_M(d, L, 1)
                Mn, X0 = start(M, p)
                E = X0 - Mn
                U, s, Vt = np.linalg.svd(Mn, full_matrices=False)
                V = Vt.T
                F = U.T @ E @ V
                G = (F - F.T) / (s[:, None] + s[None, :])
                pred = n2(G)                       # ||U G V^T||_2 = ||G||_2
                meas = n2(polar(X0) - ref)
                # tail contribution: run the orbit with the first step exact
                co, _ = pe_coeffs(L, 8, CUSH, SAFE)
                with np.errstate(over="ignore", invalid="ignore"):
                    Xt_, _ = orbit(co, X0, [EXACT] + [p] * 7)
                pt = polar_safe(Xt_)
                tail = n2(pt - ref) if pt is not None else float("nan")
                key = f"d{d}_L{L:g}_p{p}"
                R["first_order"][key] = {"sub": meas, "pred": float(pred),
                                         "ratio": float(pred / meas) if meas else None,
                                         "tail": tail, "E2": n2(E), "u": u,
                                         "E2_u": n2(E) / u, "E2_pow": matvec_norm2(E)}
                print(f"  {d:4d} {L:8g} {p:3d} {meas/u:9.2f} {pred/u:9.2f} {pred/meas:9.2f} "
                      f"{tail/u:8.2f} {n2(E)/u:8.2f} {n2(E)/L/u:11.1f}")

    # power iteration vs SVD for ||E||_2 (the bound of [D] must be SVD-free)
    rel = [abs(v["E2_pow"] - v["E2"]) / v["E2"] for v in R["first_order"].values()]
    print(f"\n  power-iteration ||E||_2 vs SVD: max relative deviation {max(rel):.1e}")
    R["power_iter_err"] = float(max(rel))


# ------------------------------------------------------------------ [C]
def part_C():
    print("\n" + "=" * 104)
    print("[C] is C(ell) ~ ell^-0.73 a law?  dense ell sweep, measured vs first-order vs the cap 2/u")
    print("=" * 104)
    R["C_of_ell"] = {}
    LS = (3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5)
    for d in (64, 128):
        for p in (8, 12):
            u = 2.0 ** -p
            rows = []
            print(f"\n  d={d} p={p}")
            print(f"      {'ell':>8s} {'K':>2s} {'sub':>10s} {'C':>8s} {'C_pred':>9s} {'sigma':>10s} "
                  f"{'sub/2':>6s} {'slope':>7s}")
            prev = None
            for L in LS:
                M, ref = make_M(d, L, 1)
                Mn, X0 = start(M, p)
                E = X0 - Mn
                U, s, Vt = np.linalg.svd(Mn, full_matrices=False)
                F = U.T @ E @ Vt.T
                G = (F - F.T) / (s[:, None] + s[None, :])
                pred = n2(G)
                K = 8 if L > 1e-5 else 8
                co, _ = pe_coeffs(L, K, CUSH, SAFE)
                with np.errstate(over="ignore", invalid="ignore"):
                    X, _ = orbit(co, X0, [p] * K)
                pol = polar_safe(X)
                if pol is None:
                    print(f"      {L:8g} diverged at p={p} (cell excluded)")
                    R.setdefault("diverged", []).append(f"d{d}_p{p}_L{L:g}")
                    prev = None
                    continue
                try:
                    internal = n2(X @ X.T - np.eye(d))     # singular-value information only
                except Exception:
                    internal = float("nan")
                tot = n2(X - ref)
                sig = n2(X - pol)
                sub = n2(pol - ref)
                C = sub / u
                slope = None
                if prev:
                    slope = float(np.log(C / prev[0]) / np.log(L / prev[1]))
                rows.append({"L": L, "K": K, "total": tot, "sigma": sig, "sub": sub,
                             "C": C, "C_pred": pred / u, "C_pred_raw": float(pred),
                             "internal": internal, "slope": slope})
                print(f"      {L:8g} {K:2d} {sub:10.2e} {C:8.2f} {pred/u:9.2f} {sig:10.2e} "
                      f"{sub/2:6.3f} " + (f"{slope:7.2f}" if slope else "      -"))
                prev = (C, L)
            R["C_of_ell"][f"d{d}_p{p}"] = rows
            Cs = np.array([r["C"] for r in rows])
            Ls = np.array([r["L"] for r in rows])
            sat = np.array([r["sub"] > 1.6 for r in rows])
            # fits: all points / unsaturated points / small-ell window
            def fit(mask):
                if mask.sum() < 2:
                    return None
                sl = -np.polyfit(np.log(Ls[mask]), np.log(Cs[mask]), 1)[0]
                return float(sl)
            ok = ~sat
            print(f"      fit all {len(Cs)} pts: slope {-np.polyfit(np.log(Ls), np.log(Cs), 1)[0]:+.3f}"
                  f" | unsaturated {int(ok.sum())} pts: {fit(ok)}"
                  f" | ell<=1e-2 unsaturated: {fit(ok & (Ls <= 1e-2))}")
            R["C_of_ell"][f"d{d}_p{p}_fits"] = {
                "all": float(-np.polyfit(np.log(Ls), np.log(Cs), 1)[0]),
                "unsat": fit(ok),
                "small_unsat": fit(ok & (Ls <= 1e-2)),
                "n_sat": int(sat.sum()),
                "cap": 2.0 / u}


# ------------------------------------------------------------------ [D]
def part_D():
    print("\n" + "=" * 104)
    print("[D] an SVD-free upper bound:  sub_err <= ||E||_2 / ell   (E = quantised start - start)")
    print("=" * 104)
    print(f"  {'d':>4s} {'ell':>8s} {'p':>3s} {'sub/u':>9s} {'bound1/u':>9s} {'ratio':>7s} "
          f"{'bound2/u':>9s} {'ratio':>7s}")
    R["bound"] = {}
    for d in (32, 64, 128):
        for L in (1e-1, 1e-2, 1e-3, 1e-4):
            for p in (8, 12):
                u = 2.0 ** -p
                M, ref = make_M(d, L, 1)
                Mn, X0 = start(M, p)
                E = X0 - Mn
                co, _ = pe_coeffs(L, 8, CUSH, SAFE)
                with np.errstate(over="ignore", invalid="ignore"):
                    X, _ = orbit(co, X0, [p] * 8)
                pol = polar_safe(X)
                if pol is None:
                    print(f"  {d:4d} {L:8g} {p:3d} diverged (cell excluded)")
                    R.setdefault("diverged", []).append(f"d{d}_p{p}_L{L:g}")
                    continue
                sub = n2(pol - ref)
                En = matvec_norm2(E)                       # no SVD
                # sigma_min(X0) lower-bounded by ell*(1-u/ell) in theory; use the
                # design value ell (what a caller knows) and no other information
                b1 = 2 * En / (2 * L)                      # 2||E||/(s_min(X0)+s_min(M))
                b2 = En / L                                 # first-order form, ||E||/(2 ell) x2
                b3 = float(np.linalg.norm(E, "fro")) / L    # exact O(d^2) fallback, ||E||_F/ell
                R["bound"][f"d{d}_L{L:g}_p{p}"] = {"sub": sub, "b1": b1, "b2": b2, "b3": b3,
                                                   "b1_ratio": b1 / sub, "b2_ratio": b2 / sub,
                                                   "b3_ratio": b3 / sub,
                                                   "pow_rel": abs(En - n2(E)) / n2(E)}
                print(f"  {d:4d} {L:8g} {p:3d} {sub/u:9.2f} {b1/u:9.1f} {b1/sub:7.1f} "
                      f"{b2/u:9.1f} {b2/sub:7.1f}  Frobenius slack {b3/sub:6.1f}x  "
                      f"pow. err {abs(En - n2(E)) / n2(E):5.1%}")
    ratios = [v["b1_ratio"] for v in R["bound"].values()]
    fr = [v["b3_ratio"] for v in R["bound"].values()]
    pw = [v["pow_rel"] for v in R["bound"].values()]
    print(f"\n  bound 2||E||/(2 ell): valid in {sum(r >= 1 for r in ratios)}/{len(ratios)} cells, "
          f"slack {min(ratios):.1f}x - {max(ratios):.1f}x (median {np.median(ratios):.1f}x)")
    print(f"  Frobenius fallback ||E||_F/ell: valid {sum(r >= 1 for r in fr)}/{len(fr)}, "
          f"slack {min(fr):.1f}x - {max(fr):.1f}x (median {np.median(fr):.1f}x)")
    print(f"  power iteration vs SVD: <=1% in {sum(p <= 0.01 for p in pw)}/{len(pw)}, "
          f"<=5% in {sum(p <= 0.05 for p in pw)}/{len(pw)}, max {max(pw):.1%}")
    R["bound_summary"] = {"n": len(ratios), "valid": int(sum(r >= 1 for r in ratios)),
                          "min": float(min(ratios)), "max": float(max(ratios)),
                          "median": float(np.median(ratios)),
                          "frob_min": float(min(fr)), "frob_median": float(np.median(fr)),
                          "frob_max": float(max(fr)),
                          "pow_le_1pct": int(sum(p <= 0.01 for p in pw)),
                          "pow_max": float(max(pw))}


# ------------------------------------------------------------------ [E]
def part_D2():
    print("\n" + "=" * 104)
    print("[D2] validating the polar-factor Lipschitz form used in [D]")
    print("      ||Q(A+E)-Q(A)|| <= 2||E||/(s_min(A)+s_min(A+E))")
    print("=" * 104)
    rs = np.random.RandomState(0)
    worst, ws = 0.0, None
    for _ in range(200):
        d = int(rs.choice([8, 16, 32]))
        U, _ = np.linalg.qr(rs.randn(d, d))
        V, _ = np.linalg.qr(rs.randn(d, d))
        L = 10 ** rs.uniform(-4, -0.5)
        A = (U * np.logspace(0, np.log10(L), d)) @ V.T
        E = rs.randn(d, d)
        E *= 2.0 ** -8 * rs.uniform(0.3, 3) / np.linalg.norm(E, 2)
        num = n2(polar(A) - polar(A + E))
        b = 2 * n2(E) / (np.linalg.svd(A, compute_uv=False)[-1] +
                         np.linalg.svd(A + E, compute_uv=False)[-1])
        if num / b > worst:
            worst, ws = num / b, {"d": d, "ell": L, "num": num, "bound": b}
    print(f"  200 random (A,E) pairs: max  measured/bound = {worst:.3f}  (<=1 means the bound holds)")
    print(f"  worst cell: d={ws['d']}, ell={ws['ell']:.2e}, measured={ws['num']:.2e}, bound={ws['bound']:.2e}")
    R["lipschitz_check"] = {"n": 200, "max_ratio": float(worst), "worst": ws}


def part_E():
    print("\n" + "=" * 104)
    print("[E] negative: no spectral quantity of X alone can certify the subspace part")
    print("=" * 104)
    rs = np.random.RandomState(0)
    Q, _ = np.linalg.qr(rs.randn(64, 64))
    M, ref = make_M(64, 1e-3, 1)
    print(f"  a random orthogonal Q:  ||Q Q^T - I||_2 = {n2(Q @ Q.T - np.eye(64)):.1e}, "
          f"subspace distance to UV^T = {n2(polar(Q) - ref):.3f}")
    R["spectral_negative"] = {"orth_internal": n2(Q @ Q.T - np.eye(64)),
                              "orth_sub": n2(polar(Q) - ref)}
    xs, ys = [], []
    for k, v in R["C_of_ell"].items():
        if k.endswith("_fits"):
            continue
        for r in v:
            if "internal" in r:
                xs.append(r["internal"])
                ys.append(r["sub"])
    print(f"  grid cells with recorded ||XX^T-I||: {len(xs)}")
    if len(xs) > 3:
        from scipy.stats import spearmanr, pearsonr
        rho, pv = spearmanr(xs, ys)
        pr, _ = pearsonr(np.log(xs), np.log(ys))
        R["internal_corr"] = {"n": len(xs), "spearman": float(rho), "p": float(pv),
                              "pearson_loglog": float(pr)}
        print(f"  Spearman(||XX^T-I||, sub_err) = {rho:+.2f}  (p={pv:.2g}); "
              f"Pearson on logs = {pr:+.2f}")
        print(f"  internal quantity range {min(xs):.2e}-{max(xs):.2e} vs sub_err range "
              f"{min(ys):.2e}-{max(ys):.2e}")


def main():
    part_A()
    part_B()
    part_C()
    part_D()
    part_D2()
    part_E()
    out = os.path.join(HERE, "pilot10_results.json")
    with open(out, "w") as f:
        json.dump(R, f, indent=1, default=float)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
