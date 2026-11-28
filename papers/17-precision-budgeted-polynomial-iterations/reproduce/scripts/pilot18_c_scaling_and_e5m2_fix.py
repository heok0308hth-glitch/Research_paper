#!/usr/bin/env python3
"""
Pilot 18 - the two open measurement items.

 [A] The dimension trend of the floor constant.  Pilot 17's spectrum-only proxy
     P = vhat * ||W||_F, W = [1/(s_i+s_j)], matches the four spectrum families at the
     median (0.484-0.535) but drifts with d (0.68 at d=48 down to 0.35 at d=256).  Here:
     the drift exponent is fitted per (family, ell, p); eight candidate predictors are
     scored by the spread of their ratio over 4 families x 5 dimensions x 2 ell x 2 p x
     3 seeds, by the residual d-exponent, and by the fraction of cells inside +-25% of
     the pooled median; the winner is then re-read on a held-out grid (other ell, other
     p, other seeds) that shares no cell with the fit.  The predictors are spectrum-only
     readouts of the entry-weighted pair operator W and the cast amplitude.

 [B] E5M2 (p = 4) at ell = 1e-3, where the shipped safety 1.01 diverges in 12 of 12
     runs and 1.031 was measured critical.  Two repairs are priced: raise the safety
     factor; or run the input cast and the first one or two steps in a wider format
     (E4M3 = p 5, bf16 = p 8) and the rest at p = 4.  For every rule: the divergence
     census over the same 12 spectra, the floor it lands on, the exact-arithmetic floor
     E_K of its own design, and the critical safety by bisection.  Also measured: how
     much the repair costs at the benign dynamic ranges where nothing diverges, and
     whether p = 4 is usable at ell = 1e-3 at all once the blow-up is suppressed.

Kernel model as in pilots 1-17: state and intermediates stored in p significant bits,
products accumulated in float64, exact matrix products in between.  Writes
pilot18_results.json.
"""
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pilot7_pert_aware_design import q, pe_coeffs, stats
from pilot6_s_pert import gain

R = {}
C_CERT = 0.142
K_DEF, CUSH, SAFE = 6, 0.02, 1.01
DIV_THR = 5.0
CAP = 2.0


def n2(X):
    return float(np.linalg.norm(X, 2))


_CACHE = {}


def spectra(kind, d, L, seed=7):
    key = (kind, d, L, seed)
    if key in _CACHE:
        return _CACHE[key]
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
    _CACHE[key] = (M, U @ V.T, sv)
    return _CACHE[key]


def kernel(co, X0, p):
    X = q(X0, p)
    for a, b, c in co:
        A = X @ X.T
        B = q(b * A + c * (A @ A), p)
        X = q(a * X + B @ X, p)
    return X


def kernel_steps(co, X0, p_cast, precs):
    """cast at p_cast, step t at precs[t]; one precision list per design."""
    X = q(X0, p_cast)
    for (a, b, c), p in zip(co, precs):
        A = X @ X.T
        B = q(b * A + c * (A @ A), p)
        X = q(a * X + B @ X, p)
    return X


def sub_err(X, ref):
    Wk, _, Vtk = np.linalg.svd(X, full_matrices=False)
    return n2(Wk @ Vtk - ref)


# ================================================================== [A] predictors
def pair_ratchet(co, sv):
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


def predictors(s, d, u, S, Sp):
    """Eight spectrum readouts of the cast response, each in units of the floor.

    s      : singular values of the normalised start (s_max = 1), descending
    u      : 2^-p
    S, Sp  : prefix ratchet of the scalar orbit and of the pair orbit
    """
    W = 1.0 / (s[:, None] + s[None, :])
    h = (s[:, None] * s[None, :]) * W                  # entry weight of the cast response
    W0 = W.copy()
    np.fill_diagonal(W0, 0.0)
    h0 = h.copy()
    np.fill_diagonal(h0, 0.0)
    fro_s = float(np.sqrt((s ** 2).sum()))             # ||Mhat||_F with s_max = 1
    v = (u / np.sqrt(3.0)) * fro_s / d                 # rms entry of the cast error
    nWf = float(np.linalg.norm(W, "fro"))
    nW2 = n2(W)
    nhf = float(np.linalg.norm(h0, "fro"))
    nh2 = n2(h)
    # stable rank of the pair-weight operator
    deff = nhf ** 2 / nh2 ** 2 if nh2 > 0 else float(d)
    row = float(np.sqrt((h0 ** 2).sum(axis=1).max()))  # largest row degree (variance)
    return {
        "S": u * S,
        "Sp": u * Sp,
        "F": v * nWf,                                  # pilot 17 proxy
        "Fd": v * nWf / np.sqrt(d),                    # 1/sqrt(d) normalisation
        "2d": v * nW2 / np.sqrt(d),
        "hd": (u / np.sqrt(6.0)) * nhf / np.sqrt(d),   # entry-weighted operator, 1/sqrt(d)
        "hrow": 2.0 * (u / np.sqrt(6.0)) * row,        # largest row degree
        "heff": np.sqrt(2.0 / max(deff, 1.0)) * (u / np.sqrt(6.0)) * nhf,
        "d_eff": deff,
        "s_eff": float((s.sum() ** 2) / (s ** 2).sum()),
    }


def cell(d, kind, L, p, seed):
    M, ref, sv = spectra(kind, d, L, seed)
    co, _ = pe_coeffs(L, K_DEF, CUSH, SAFE)
    if not np.all(np.isfinite(np.array(co, float))):
        raise ValueError(f"design coefficients are not finite at ell={L:g}")
    X0 = M / n2(M)
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            e = sub_err(kernel(co, X0, p), ref)
        if not np.isfinite(e):
            e = float("inf")
    except Exception:
        e = float("inf")
    grid = np.logspace(np.log10(L), 0, 400)
    EK, S, _ = stats(co, grid, K_DEF)
    u = 2.0 ** -p
    s = sv / sv.max()
    Sp = pair_ratchet(co, s)
    P = predictors(s, d, u, S, Sp)
    out = dict(d=d, spectrum=kind, ell=L, p=p, seed=seed, err=e, EK=EK, S=S, S_pair=Sp,
               u=u, target=(e - EK) if np.isfinite(e) else None, preds=P, diverged=not np.isfinite(e))
    out["over_pred"] = {k: (e - EK) / P[k] for k in P if k not in ("d_eff", "s_eff")
                        and P[k] > 0 and np.isfinite(e)}
    out["over_fit"] = {k: (e - EK) / P[k] / C_CERT for k in out["over_pred"]}
    return out


def fit_exponent(xs, ys):
    """least-squares slope of log y on log x (the residual d-exponent)."""
    x = np.log(np.asarray(xs, float))
    y = np.log(np.asarray(ys, float))
    if x.size < 2 or not np.all(np.isfinite(y)):
        return None
    A = np.vstack([x, np.ones_like(x)]).T
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(b[0]), float(b[1])


KEYS = ("S", "Sp", "F", "Fd", "2d", "hd", "hrow", "heff")


def sectionA():
    print("=" * 108)
    print("[A1] the dimension trend of the pilot-17 proxy, per (family, ell, p): fitted exponent")
    print("=" * 108)
    dims = (32, 48, 64, 128, 256)
    fams = ("logspace", "plateau", "gap", "power")
    cells = []
    for kind in fams:
        for L in (1e-1, 1e-2):
            for p in (8, 12):
                for seed in (0, 1, 2):
                    for d in dims:
                        cells.append(cell(d, kind, L, p, seed))
    R["a1_cells"] = cells
    print(f"  {len(cells)} cells (4 families x 2 ell x 2 p x 3 seeds x 5 dims)")
    exp = {}
    for kind in fams:
        for L in (1e-1, 1e-2):
            for p in (8, 12):
                for k in KEYS:
                    row = [c for c in cells if c["spectrum"] == kind and c["ell"] == L
                           and c["p"] == p and k in c["over_pred"]]
                    # median over seeds at each d, then the slope
                    ds, rs = [], []
                    for d in dims:
                        vals = [c["over_pred"][k] for c in row if c["d"] == d]
                        if vals:
                            ds.append(d)
                            rs.append(float(np.median(vals)))
                    f = fit_exponent(ds, rs)
                    exp[f"{kind}_L{L:g}_p{p}_{k}"] = f, list(zip(ds, rs))
    R["a1_exponent"] = {k: {"slope": v[0][0] if v[0] else None,
                            "intercept": v[0][1] if v[0] else None,
                            "points": [[d, r] for d, r in v[1]]}
                        for k, v in exp.items()}
    for k in ("F", "Fd", "hd", "hrow"):
        sl = [v[0][0] for key, v in exp.items() if key.endswith("_" + k) and v[0]]
        print(f"  {k:>5}: residual d-exponent per (family, ell, p) = "
              f"{min(sl):+.2f} .. {max(sl):+.2f}  (median {np.median(sl):+.2f})")
    print("\n  pilot-17 proxy ratio vs d (median over seeds):")
    for kind in fams:
        for L in (1e-1, 1e-2):
            pts = R["a1_exponent"][f"{kind}_L{L:g}_p8_F"]["points"]
            print(f"   {kind:>9} ell={L:6g} p=8: " + "  ".join(f"{d}:{r:.3f}" for d, r in pts))

    print("\n" + "=" * 108)
    print("[A2] candidate predictors: spread over all cells, residual exponent, +-25% hit rate")
    print("=" * 108)
    summ = {}
    for k in KEYS:
        v = np.array([c["over_pred"][k] for c in cells if k in c["over_pred"]], float)
        v = v[np.isfinite(v) & (v > 0)]
        med = float(np.median(v))
        sl = []
        for kind in fams:
            for L in (1e-1, 1e-2):
                for p in (8, 12):
                    f = R["a1_exponent"][f"{kind}_L{L:g}_p{p}_{k}"]["slope"]
                    if f is not None:
                        sl.append(f)
        fam_med = {}
        for kind in fams:
            w = np.array([c["over_pred"][k] for c in cells
                          if c["spectrum"] == kind and k in c["over_pred"]], float)
            fam_med[kind] = float(np.median(w))
        dmed = {}
        for d in dims:
            w = np.array([c["over_pred"][k] for c in cells if c["d"] == d and k in c["over_pred"]],
                         float)
            dmed[d] = float(np.median(w))
        summ[k] = dict(n=int(v.size), median=med, lo=float(v.min()), hi=float(v.max()),
                       spread=float(v.max() / v.min()),
                       frac_25=float(np.mean(np.abs(v / med - 1) <= 0.25)),
                       frac_40=float(np.mean(np.abs(v / med - 1) <= 0.40)),
                       exponent_median=float(np.median(sl)), exponent_absmax=float(np.max(np.abs(sl))),
                       family_median=fam_med, d_median=dmed)
        print(f"  {k:>5}: median {med:8.3f}  [{v.min():7.3f}, {v.max():7.3f}] spread "
              f"{v.max() / v.min():5.2f}x  +-25% {np.mean(np.abs(v / med - 1) <= 0.25) * 100:5.1f}% "
              f" d-exp {np.median(sl):+.2f} (|max| {np.max(np.abs(sl)):.2f})")
    R["a2_predictors"] = summ
    print("\n  per-family median and per-d median of the two best candidates:")
    for k in ("Fd", "hd", "hrow"):
        if k not in summ:
            continue
        print(f"   {k}: family " + " ".join(f"{f}:{summ[k]['family_median'][f]:7.3f}" for f in fams))
        print(f"       d      " + " ".join(f"{d}:{summ[k]['d_median'][d]:7.3f}" for d in dims))

    print("\n" + "=" * 108)
    print("[A3] the winner on a held-out grid (ell 5e-2, 3e-3; p 6, 16; seeds 3, 4; d 48, 128)")
    print("=" * 108)
    ho = []
    for kind in fams:
        for L in (5e-2, 3e-3):
            for p in (6, 16):
                for seed in (3, 4):
                    for d in (48, 128):
                        ho.append(cell(d, kind, L, p, seed))
    R["a3_heldout"] = ho
    ho_summ = {}
    for k in KEYS:
        v = np.array([c["over_pred"][k] for c in ho if k in c["over_pred"]], float)
        v = v[np.isfinite(v) & (v > 0)]
        if v.size == 0:
            continue
        med = float(np.median(v))
        ref = summ[k]["median"]
        ho_summ[k] = dict(n=int(v.size), median=med, lo=float(v.min()), hi=float(v.max()),
                          spread=float(v.max() / v.min()),
                          ratio_to_fit=med / ref,
                          frac_fit_band=float(np.mean(np.abs(v / ref - 1) <= 0.25)),
                          frac_fit_band40=float(np.mean(np.abs(v / ref - 1) <= 0.40)))
    R["a3_summary"] = ho_summ
    for k in KEYS:
        if k in ho_summ:
            s = ho_summ[k]
            print(f"  {k:>5}: median {s['median']:8.3f} (fit {summ[k]['median']:.3f}, "
                  f"x{s['ratio_to_fit']:.2f})  [{s['lo']:.3f}, {s['hi']:.3f}] spread "
                  f"{s['spread']:.2f}x  inside fit +-25%: {s['frac_fit_band'] * 100:.0f}%")

    print("\n" + "=" * 108)
    print("[A4] what the two best predictors look like on the 48 known cells of pilot 16/17")
    print("=" * 108)
    for kind in fams:
        for L in (1e-1, 1e-2):
            vals = {k: [c["over_pred"][k] for c in cells
                        if c["spectrum"] == kind and c["ell"] == L and c["p"] == 8
                        and k in c["over_pred"]] for k in KEYS}
            print(f"   {kind:>9} ell={L:6g}: " + "  ".join(
                f"{k}={np.median(v):.3f}" for k, v in vals.items() if k in ("F", "Fd", "hd", "hrow")))


# ================================================================== [B] E5M2 repairs
def run_cell(co, X0, ref, p_cast, precs):
    """two criteria: the state distance (pilot 17's divergence test) and the subspace floor."""
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            X = kernel_steps(co, X0, p_cast, precs)
        state = n2(X - ref)
        if not np.isfinite(state):
            return float("inf"), float("inf")
        sub = sub_err(X, ref)
        return state, (sub if np.isfinite(sub) else float("inf"))
    except Exception:
        return float("inf"), float("inf")


def lattice(L, p, safety, p_cast=None, precs=None):
    """12 spectra: d in {48,64}, six seeds, logspace."""
    p_cast = p if p_cast is None else p_cast
    precs = [p] * K_DEF if precs is None else precs
    co, _ = pe_coeffs(L, K_DEF, CUSH, safety)
    if not np.all(np.isfinite(np.array(co, float))):
        raise ValueError(f"design coefficients are not finite at ell={L:g}, safety={safety:g}")
    grid = np.logspace(np.log10(L), 0, 400)
    EK, S, _ = stats(co, grid, K_DEF)
    out = []
    for d in (48, 64):
        for seed in range(6):
            M, ref, sv = spectra("logspace", d, L, seed)
            X0 = M / n2(M)
            st, sub = run_cell(co, X0, ref, p_cast, precs)
            out.append(dict(d=d, seed=seed, state=st, sub=sub,
                            div=(not np.isfinite(st)) or st > DIV_THR,
                            at_cap=(not np.isfinite(sub)) or sub >= 0.9 * CAP))
    fin = [c["sub"] for c in out if np.isfinite(c["sub"])]
    sts = [c["state"] for c in out if np.isfinite(c["state"])]
    return dict(EK=EK, S=S, cells=out,
                n_div=sum(c["div"] for c in out), n_cap=sum(c["at_cap"] for c in out),
                median_sub=float(np.median(fin)) if fin else float("inf"),
                max_sub=float(max(fin)) if fin else float("inf"),
                median_state=float(np.median(sts)) if sts else float("inf"),
                max_state=float(max(sts)) if sts else float("inf"))


RULES = [
    ("ship_p4_s1.010", dict(p=4, safety=1.010)),
    ("p4_s1.020", dict(p=4, safety=1.020)),
    ("p4_s1.030", dict(p=4, safety=1.030)),
    ("p4_s1.035", dict(p=4, safety=1.035)),
    ("p4_s1.050", dict(p=4, safety=1.050)),
    ("p4_cast8_s1.010", dict(p=4, safety=1.010, p_cast=8)),
    ("p4_cast5_s1.010", dict(p=4, safety=1.010, p_cast=5)),
    ("p4_cast8_step1_8", dict(p=4, safety=1.010, p_cast=8, precs=[8, 4, 4, 4, 4, 4])),
    ("p4_cast8_step12_8", dict(p=4, safety=1.010, p_cast=8, precs=[8, 8, 4, 4, 4, 4])),
    ("p4_cast5_step1_5", dict(p=4, safety=1.010, p_cast=5, precs=[5, 4, 4, 4, 4, 4])),
    ("p4_cast5_step12_5", dict(p=4, safety=1.010, p_cast=5, precs=[5, 5, 4, 4, 4, 4])),
    ("p4_step1_8_cast4", dict(p=4, safety=1.010, p_cast=4, precs=[8, 4, 4, 4, 4, 4])),
    ("p4_s1.020_cast5_step1_5", dict(p=4, safety=1.020, p_cast=5, precs=[5, 4, 4, 4, 4, 4])),
    ("p5_s1.010", dict(p=5, safety=1.010)),
    ("p8_s1.010", dict(p=8, safety=1.010)),
]


def show(name, res, cfg=None):
    print(f"  {name:<26} div {res['n_div']:2d}/12  cap {res['n_cap']:2d}/12  "
          f"floor {res['median_sub']:.3e} [{res['max_sub']:.2e}]  "
          f"state {res['median_state']:.2e} [max {res['max_state']:.2e}]  "
          f"EK {res['EK']:.2e}  S {res['S']:.1f}")


def sectionB():
    print("\n" + "=" * 108)
    print("[B1] E5M2 (p = 4) repairs at ell = 1e-3: 12 spectra, K = 6, shipped design (cushion 0.02)")
    print("     div = state distance above 5 (the blow-up pilot 17 measured); cap = subspace floor >= 1.8")
    print("=" * 108)
    tab = {}
    for name, cfg in RULES:
        res = lattice(1e-3, cfg["p"], cfg["safety"], cfg.get("p_cast"), cfg.get("precs"))
        tab[name] = res
        show(name, res)
    R["b1_ell1e-3"] = tab

    print("\n" + "=" * 108)
    print("[B2] the same rules where nothing blows up (ell = 1e-1 and 1e-2): floor and its cost")
    print("=" * 108)
    cost = {}
    for L in (1e-1, 1e-2):
        base = None
        for name, cfg in RULES:
            if cfg["p"] != 4:
                continue
            res = lattice(L, 4, cfg["safety"], cfg.get("p_cast"), cfg.get("precs"))
            cost[f"L{L:g}_{name}"] = res
            if name == "ship_p4_s1.010":
                base = res
            print(f"  ell={L:6g}", end=" ")
            show(name, res)
        for p in (5, 8):
            res = lattice(L, p, 1.010)
            cost[f"L{L:g}_p{p}_s1.010"] = res
            print(f"  ell={L:6g}", end=" ")
            show(f"p{p}_s1.010 (reference)", res)
        if base:
            for k, v in cost.items():
                if k.startswith(f"L{L:g}_") and np.isfinite(v["median_sub"]) and v["median_sub"] > 0:
                    v["ratio_to_ship"] = v["median_sub"] / base["median_sub"]
    R["b2_cost"] = cost
    for L in (1e-1, 1e-2):
        print(f"\n  ell={L:6g}, floor relative to the shipped p=4 rule:")
        print("   " + "  ".join(f"{k.split('_', 1)[1]}={v['ratio_to_ship']:.2f}"
                                for k, v in cost.items() if k.startswith(f"L{L:g}_")
                                and "ratio_to_ship" in v))

    print("\n" + "=" * 108)
    print("[B3] critical safety of each rule at ell = 1e-3 (bisection on the blow-up criterion)")
    print("=" * 108)
    crit = {}
    for name, cfg in RULES:
        if cfg["p"] == 8 and cfg.get("p_cast") is None:
            continue
        def ok(s):
            return lattice(1e-3, cfg["p"], s, cfg.get("p_cast"), cfg.get("precs"))["n_div"] == 0
        if ok(1.0):
            crit[name] = dict(s_crit=1.0, note="no damping needed")
            print(f"  {name:<26} no damping needed (stable at safety 1.0)")
            continue
        lo, hi = 1.0, 1.20
        if not ok(hi):
            crit[name] = dict(s_crit=None, note="still blows up at 1.20")
            print(f"  {name:<26} still blows up at safety 1.20")
            continue
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            if ok(mid):
                hi = mid
            else:
                lo = mid
        crit[name] = dict(s_crit=float(hi))
        print(f"  {name:<26} critical safety {hi:.4f}  (margin {hi - 1:.4f} = "
              f"{(hi - 1) / 2.0 ** -(cfg['p'] + 1):.2f} x u/2)")
    R["b3_critical"] = crit

    print("\n" + "=" * 108)
    print("[B4] is p = 4 usable at ell = 1e-3 once the blow-up is suppressed?")
    print("=" * 108)
    use = {}
    for name, res in tab.items():
        use[name] = dict(median=res["median_sub"], max=res["max_sub"],
                         at_cap=float(np.mean([c["at_cap"] for c in res["cells"]])),
                         div=res["n_div"])
    R["b4_usability"] = use
    for name, v in use.items():
        print(f"  {name:<26} floor {v['median']:.3e}  max {v['max']:.3e}  "
              f"at the cap: {v['at_cap'] * 100:3.0f}% of 12  div {v['div']}/12")
    co, _ = pe_coeffs(1e-3, K_DEF, CUSH, 1.035)
    grid = np.logspace(-3, 0, 400)
    EK, S, _ = stats(co, grid, K_DEF)
    u = 2.0 ** -4
    print(f"\n  law at the repaired point (safety 1.035, p=4, ell=1e-3): pred "
          f"{EK + C_CERT * u * S:.3e} (EK {EK:.2e}, c*u*S {C_CERT * u * S:.2e}), "
          f"measured median {tab['p4_s1.035']['median_sub']:.3e}")
    R["b4_law_at_repair"] = dict(EK=EK, S=S, pred=EK + C_CERT * u * S,
                                 measured=tab["p4_s1.035"]["median_sub"])


def main():
    R["config"] = {"K": K_DEF, "cushion": CUSH, "safety": SAFE, "c": C_CERT,
                   "div_threshold": DIV_THR, "numpy": np.__version__}
    sectionA()
    sectionB()
    path = os.path.join(HERE, "pilot18_results.json")
    with open(path, "w") as f:
        json.dump(R, f, indent=1, default=float)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
