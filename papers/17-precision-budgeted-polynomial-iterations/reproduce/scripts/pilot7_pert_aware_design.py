#!/usr/bin/env python3
"""
Pilot 7 - Is a precision-aware design real?

The certificate of pilot6,

        err ~ E_K + c * u * S_pert(K),   c = 0.142,  u = 2^-p,

turns the design problem into the minimisation of  J = E_K + c*u*S_pert(K).  If that
objective had an interior optimum away from the shipped prescription (cushion 0.02,
safety 1.01), "precision-budgeted Remez" would be a real algorithm: fewer steps or
lower error at bf16.  This script decides the question with measurements, not with
the model:

  [A] scan J over a three-parameter design family
      (cushion c, safety s, first-step damping tau) x K, for
      L in {1e-2,1e-3,1e-4}, p in {8,11,14,16,24}; record the argmin, the shipped
      point and the predicted gain;
  [B] the literal one-parameter family of the brief: the last step solved as a
      linear program with p'(1) = lam fixed, everything else shipped.  The ratchet
      makes S_pert(K) >= S_pert(K-1) for a design and all its extensions, so this
      family is predicted to have no interior optimum; measured below;
  [C] the decisive test: run the actual p-bit matrix kernel on a 24-design grid
      (plus NS5, LSQ5, greedy) and compare measured error against the objective's
      ranking.  A model that ranks designs correctly can be used for design; one
      that does not cannot, however small its within-family residual.

Kernel model as in pilot1c/pilot5b/pilot6: X and B stored in p significant bits,
A = XX^T not stored, products accumulated in float64.
"""
import json
import os
import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
R = {"config": {}}
C_CERT = 0.142
CAP = 1e12


# ---------------------------------------------------------------- kernel model
def q(x, p):
    if p is None or p >= 30:
        return np.asarray(x, float)
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


def damp_first(co, tau):
    """design family: extra damping of step 1 only (tau = 1 is the shipped family)"""
    if tau == 1.0:
        return list(co)
    a, b, c = co[0]
    return [(a / tau, b / tau ** 3, c / tau ** 5)] + list(co[1:])


def deriv(a, b, c, x):
    return a + 3 * b * x ** 2 + 5 * c * x ** 4


def orbit(co, x0):
    x = np.asarray(x0, float)
    tr = [x.copy()]
    for a, b, c in co:
        x = a * x + b * x ** 3 + c * x ** 5
        tr.append(x.copy())
    return np.array(tr)


def prefix_curve(co, grid):
    """S(t) for every prefix t = 1..K (ratchet = max of this curve)."""
    tr = orbit(co, grid)
    cur = np.zeros((0, grid.size))
    out = []
    for t in range(1, len(co) + 1):
        a, b, c = co[t - 1]
        d = np.abs(deriv(a, b, c, tr[t - 1]))
        cur = np.vstack([cur * d[None, :], d[None, :]])
        out.append(float(np.max(cur)))
    return out


def stats(co, grid, K=None):
    K = K or len(co)
    pc = prefix_curve(co, grid)
    EK = float(np.max(np.abs(orbit(co, grid)[K] - 1)))
    return EK, max(pc), pc[-1]


def mat_run(co, M, p):
    X = q(M / np.linalg.norm(M, 2), p)
    for a, b, c in co:
        A = X @ X.T
        B = q(b * A + c * (A @ A), p)
        X = q(a * X + B @ X, p)
    return X


def measured(co, M, ref, p):
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            v = float(np.linalg.norm(mat_run(co, M, p) - ref, 2))
        return v if np.isfinite(v) else float("inf")
    except Exception:
        return float("inf")


def make_M(d, L, seed=1):
    rs = np.random.RandomState(seed)
    U, _ = np.linalg.qr(rs.randn(d, d))
    V, _ = np.linalg.qr(rs.randn(d, d))
    return (U * np.logspace(0, np.log10(L), d)) @ V.T, U @ V.T


def lsq_coeffs(l0, K, n=800):
    u, l, out = 1.0, l0, []
    for _ in range(K):
        x = np.linspace(l, u, n)
        A = np.stack([x, x**3, x**5], axis=1)
        a, b, c = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
        out.append((a, b, c))
        l = a * l + b * l**3 + c * l**5
        u = 2 - l
    return out


NS5 = (15 / 8, -10 / 8, 3 / 8)


# ---------------------------------------------------------------- [A] objective scan
CUS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.6)
SAS = (1.0, 1.01, 1.05, 1.1, 1.2, 1.35, 1.6)
TAUS = (1.0, 1.05, 1.1, 1.25, 1.5, 1.8, 2.2)


def scan_objective(L, p, grid, Krange=range(2, 13)):
    u = 2.0 ** -p
    per_K = {}
    for K in Krange:
        best = None
        for cu in CUS:
            for sa in SAS:
                base, _ = pe_coeffs(L, K, cu, sa)
                for tau in TAUS:
                    co = damp_first(base, tau)
                    EK, S, _ = stats(co, grid, K)
                    J = EK + C_CERT * u * S
                    if best is None or J < best["J"]:
                        best = {"J": J, "cushion": cu, "safety": sa, "tau": tau, "EK": EK, "S_pert": S}
        co, _ = pe_coeffs(L, K, 0.02, 1.01)
        EKs, Ss, _ = stats(co, grid, K)
        per_K[K] = {"best": best,
                    "shipped": {"J": EKs + C_CERT * u * Ss, "EK": EKs, "S_pert": Ss},
                    "gain": (EKs + C_CERT * u * Ss) / best["J"]}
    Kb = min(per_K, key=lambda k: per_K[k]["best"]["J"])
    Ks = min(per_K, key=lambda k: per_K[k]["shipped"]["J"])
    return {"per_K": per_K, "argmin_K": Kb, "best": per_K[Kb]["best"],
            "shipped_at_that_K": per_K[Kb]["shipped"], "gain_at_equal_K": per_K[Kb]["gain"],
            "shipped_argmin_K": Ks, "shipped_best_J": per_K[Ks]["shipped"]["J"],
            "gain_full": per_K[Ks]["shipped"]["J"] / per_K[Kb]["best"]["J"]}


# ---------------------------------------------------------------- [B] last-step derivative family
def last_step_lp(l, u, lam, n=400):
    """odd quintic p minimising max|p-1| on [l,u] subject to p'(1) = lam"""
    x = np.linspace(l, u, n)
    A = np.stack([x, x**3, x**5], axis=1)
    # variables a,b,c,eps
    Aub = np.vstack([np.hstack([A, -np.ones((n, 1))]),
                     np.hstack([-A, -np.ones((n, 1))])])
    bub = np.concatenate([np.ones(n), -np.ones(n)])
    Aeq = np.array([[1.0, 3.0, 5.0, 0.0]]) if lam is not None else None
    beq = np.array([lam]) if lam is not None else None
    c = np.array([0.0, 0.0, 0.0, 1.0])
    bounds = [(None, None), (None, None), (None, None), (0.0, None)]
    res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    if not res.success:
        return None
    a, b, cc, eps = res.x
    return (float(a), float(b), float(cc)), float(eps)


def lambda_family(L, K, p, grid, lams=None):
    base, win = pe_coeffs(L, K, 0.02, 1.01)
    l, u = win[-1]
    free = last_step_lp(l, u, None)          # unconstrained minimax on that window
    lam_star = free[0][0] + 3 * free[0][1] + 5 * free[0][2]
    if lams is None:
        lams = sorted({round(lam_star + d, 4) for d in
                       (-1.0, -0.5, -0.2, -0.05, -0.01, 0.0, 0.01, 0.05, 0.2, 0.5, 1.0)}
                      | {0.0})
    out = {}
    for lam in lams:
        sol = last_step_lp(l, u, lam)
        if sol is None:
            out[lam] = None
            continue
        coef, eps_lp = sol
        co = list(base[:-1]) + [coef]
        EK, S, Send = stats(co, grid, K)
        uu = 2.0 ** -p
        out[lam] = {"coef": coef, "lp_eps": eps_lp, "EK": EK, "S_pert": S, "S_terminal": Send,
                    "J": EK + C_CERT * uu * S, "p_prime_1": coef[0] + 3 * coef[1] + 5 * coef[2]}
    return {"window": [l, u], "lambda_star": lam_star, "lp_eps_free": free[1],
            "shipped_last_p1": base[-1][0] + 3 * base[-1][1] + 5 * base[-1][2], "rows": out}


# ---------------------------------------------------------------- [C] measurement grid
def design_grid(L, K):
    out = {}
    base, _ = pe_coeffs(L, K, 0.02, 1.01)
    out["shipped"] = base
    for cu in (0.0, 0.02, 0.1, 0.2):
        for sa in (1.0, 1.1, 1.35):
            for tau in (1.0, 1.5, 3.5):
                base, _ = pe_coeffs(L, K, cu, sa)
                out[f"c{cu}_s{sa}_t{tau}"] = damp_first(base, tau)
    out["NS5"] = [NS5] * K
    out["LSQ5"] = lsq_coeffs(L, K)
    return out


def spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def main():
    R["config"] = {"c_cert": C_CERT, "kernel": "X,B stored in p significant bits, A not stored, products float64",
                   "family": {"cushion": CUS, "safety": SAS, "first_step_damping_tau": TAUS},
                   "Ls": [1e-2, 1e-3, 1e-4], "bits": [8, 11, 14, 16, 24]}

    # ---------------- A
    print("=" * 108)
    print("[A] scan of J = E_K + 0.142 u S_pert over the 3-parameter design family")
    print("=" * 108)
    scan = {}
    for L in (1e-2, 1e-3, 1e-4):
        grid = np.logspace(np.log10(L), 0, 1500)
        for p in (8, 11, 14, 16, 24):
            r = scan_objective(L, p, grid)
            scan[f"L{L}_p{p}"] = r
            b, s = r["best"], r["shipped_at_that_K"]
            print(f" L={L:<7g} p={p:<3d} argmin K={r['argmin_K']:2d}  best J={b['J']:.3e} "
                  f"(cushion {b['cushion']}, safety {b['safety']}, tau {b['tau']}, E_K {b['EK']:.1e}, S_pert {b['S_pert']:.0f})")
            print(f"                       shipped at same K: J={s['J']:.3e} (E_K {s['EK']:.1e}, S_pert {s['S_pert']:.0f})"
                  f"  -> predicted gain {r['gain_at_equal_K']:.2f}x ; shipped global best K={r['shipped_argmin_K']} J={r['shipped_best_J']:.3e}"
                  f" -> {r['gain_full']:.2f}x")
    R["scan"] = scan

    print("\n  design parameters chosen by the objective (K, cushion, safety, tau) as a function of the budget")
    print(f"  {'L':>7s} " + "  ".join(f"{'p=' + str(p):>28s}" for p in (8, 11, 14, 16, 24)))
    for L in (1e-2, 1e-3, 1e-4):
        row = f"  {L:7g} "
        for p in (8, 11, 14, 16, 24):
            r = scan[f"L{L}_p{p}"]
            b = r["best"]
            row += f"  K{b_J_short(b)} ({r['gain_at_equal_K']:.2f}x)".ljust(30)
        print(row)

    # ---------------- B
    print("\n" + "=" * 108)
    print("[B] the brief's one-parameter family: last step solved by LP with p'(1) = lam fixed")
    print("    (ratchet monotonicity: for a design and every extension of it, S_pert(K') >= S_pert(K)")
    print("     because the ratchet of K' contains every prefix of K -> no design change confined to")
    print("     the appended steps can lower the certified floor; only E_K can move)")
    print("=" * 108)
    lams = None
    lam_out = {}
    for L, K in ((1e-3, 6), (1e-2, 6)):
        grid = np.logspace(np.log10(L), 0, 1500)
        for p in (8, 14):
            r = lambda_family(L, K, p, grid, lams)
            lam_out[f"L{L}_K{K}_p{p}"] = r
            print(f"\n L={L:g} K={K} p={p}  last-step window [{r['window'][0]:.6g}, {r['window'][1]:.6g}]")
            print(f"   unconstrained minimax on that window: lambda* = p'(1) = {r['lambda_star']:+.4f}"
                  f" (LP error {r['lp_eps_free']:.3e}); shipped last step p'(1) = {r['shipped_last_p1']:+.4f}")
            for lam in sorted(r["rows"]):
                row = r["rows"][lam]
                if row is None:
                    print(f"   lam={lam:7.4f}   LP infeasible")
                    continue
                print(f"   lam={lam:7.4f}  p'(1)={row['p_prime_1']:7.4f}  E_K={row['EK']:9.3e}  "
                      f"S_pert={row['S_pert']:8.1f}  S(K)={row['S_terminal']:8.1f}  J={row['J']:9.3e}")
    R["lambda_family"] = {k: {"window": v["window"], "lambda_star": v["lambda_star"],
                               "shipped_last_p1": v["shipped_last_p1"],
                               "rows": {str(l): r for l, r in v["rows"].items()}}
                          for k, v in lam_out.items()}

    # ---------------- C
    print("\n" + "=" * 108)
    print("[C] the decisive test: measured p-bit kernel error versus the objective's ranking")
    print("=" * 108)
    meas = []
    for L, K in ((1e-3, 8), (1e-2, 6)):
        grid = np.logspace(np.log10(L), 0, 1500)
        designs = design_grid(L, K)
        for d in (64, 128):
            M, ref = make_M(d, L)
            for p in (8, 11, 16):
                u = 2.0 ** -p
                for nm, co in designs.items():
                    EK, S, _ = stats(co, grid, K)
                    J = EK + C_CERT * u * S
                    err = measured(co, M, ref, p)
                    meas.append({"L": L, "K": K, "d": d, "p": p, "u": u, "design": nm, "EK": EK,
                                 "S_pert": S, "J": J, "err": err})
    R["measurement"] = meas

    for L, K in ((1e-3, 8), (1e-2, 6)):
        for p in (8, 11, 16):
            rows = [r for r in meas if r["L"] == L and r["p"] == p and r["d"] == 64
                    and np.isfinite(r["err"])]
            allrows = [r for r in rows if r["design"] not in ("NS5", "LSQ5")]
            ok = [r for r in allrows if r["err"] <= 1.0]                 # sane: no divergence
            bad = [r for r in allrows if r["err"] > 1.0]
            errs = np.array([r["err"] for r in ok])
            best = min(ok, key=lambda r: r["err"])
            ship = [r for r in ok if r["design"] == "shipped"][0]
            obj = min(ok, key=lambda r: r["J"])
            u = 2.0 ** -p
            rank = int(np.sum([r["err"] < ship["err"] for r in ok])) + 1
            print(f"\n L={L:g} K={K} p={p} d=64:  {len(ok)} sane designs (+{len(bad)} divergent), "
                  f"measured error {errs.min():.3e} .. {errs.max():.3e} (spread {errs.max() / errs.min():.2f}x)")
            print(f"    shipped (cushion .02, safety 1.01) err={ship['err']:.3e}  S_pert={ship['S_pert']:.0f}  "
                  f"J={ship['J']:.3e}  rank {rank}/{len(ok)}")
            print(f"    objective argmin                    err={obj['err']:.3e}  S_pert={obj['S_pert']:.0f}  "
                  f"J={obj['J']:.3e}  ({obj['design']})  -> {obj['err'] / ship['err']:.2f}x the shipped error")
            print(f"    measured best                       err={best['err']:.3e}  S_pert={best['S_pert']:.0f}  "
                  f"J={best['J']:.3e}  ({best['design']})  -> shipped/best = {ship['err'] / best['err']:.2f}x")
            cs = [r["err"] / (r["u"] * r["S_pert"]) for r in ok]
            print(f"    Spearman(J, err) = {spearman([r['J'] for r in ok], [r['err'] for r in ok]):+.2f}"
                  f"   Spearman(S_pert, err) = {spearman([r['S_pert'] for r in ok], [r['err'] for r in ok]):+.2f}"
                  f"   c spread {min(cs):.4f} .. {max(cs):.3f}")
            R.setdefault("cells", {})[f"L{L}_K{K}_p{p}_d64"] = {
                "n_sane": len(ok), "n_divergent": len(bad), "err_min": float(errs.min()),
                "err_max": float(errs.max()), "shipped": ship["err"], "shipped_rank": rank,
                "objective_argmin_err": obj["err"], "objective_argmin_design": obj["design"],
                "objective_over_shipped": obj["err"] / ship["err"],
                "measured_best": best["err"], "measured_best_design": best["design"],
                "shipped_over_best": ship["err"] / best["err"],
                "spearman_J_err": spearman([r["J"] for r in ok], [r["err"] for r in ok]),
                "spearman_S_err": spearman([r["S_pert"] for r in ok], [r["err"] for r in ok]),
                "c_min": float(min(cs)), "c_max": float(max(cs))}

    # steps to the floor: does any design get there with fewer steps?
    print("\n  steps to the floor (L=1e-3, d=64, p=8): measured error against K")
    L, d, p = 1e-3, 64, 8
    grid = np.logspace(np.log10(L), 0, 1500)
    M, ref = make_M(d, L)
    base, _ = pe_coeffs(L, 12, 0.02, 1.01)
    probes = {"shipped": base, "greedy (cushion 0)": list(pe_coeffs(L, 12, 0.0, 1.0)[0]),
              "damped (0.1/1.35/tau2.2)": damp_first(pe_coeffs(L, 12, 0.1, 1.35)[0], 2.2)}
    steps = {}
    print(f"   {'design':24s} " + " ".join(f"{'K=' + str(k):>12s}" for k in range(4, 10)))
    for nm, co12 in probes.items():
        vals = []
        for K in range(4, 10):
            co = co12[:K]
            EK, S, _ = stats(co, grid, K)
            e = measured(co, M, ref, p)
            vals.append(e)
            steps.setdefault(nm, {})[K] = {"err": e, "EK": EK, "S_pert": S}
        print(f"   {nm:24s} " + " ".join(f"{v:12.3e}" for v in vals))
    R["steps_to_floor"] = steps

    # divergence census
    print("\n  divergence census (non-finite or > 1 for a design whose exact E_K < 1):")
    div = {}
    for r in meas:
        if r["EK"] < 0.5 and (not np.isfinite(r["err"]) or r["err"] > 1.0):
            div.setdefault(r["design"], []).append(f"L{r['L']:g}_p{r['p']}_d{r['d']}")
    for k, v in div.items():
        print(f"   {k}: {len(v)} cells  {v[:6]}{' ...' if len(v) > 6 else ''}")
    R["divergence_census"] = div

    out = os.path.join(HERE, "pilot7_results.json")
    with open(out, "w") as f:
        json.dump(R, f, indent=1, default=float)
    print(f"\nwrote {out}")


def b_J_short(b):
    return f"{b['cushion']}/{b['safety']}/{b['tau']}"


if __name__ == "__main__":
    main()
