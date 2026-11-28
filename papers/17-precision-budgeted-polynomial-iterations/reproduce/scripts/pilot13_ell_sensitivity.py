#!/usr/bin/env python3
"""
Pilot 13 - the stopping rule when the dynamic range is not known.

The certificate

        err ~ E_K + c u S_pert(K),   c = 0.142,

is computed from the coefficients and one number: the dynamic range
ell = sigma_min/sigma_max of the matrix to be orthogonalised.  A caller is given
M, not its spectrum, so the question this script answers is what happens when
ell is only estimated.  Two things move when ell is replaced by ell_hat = alpha*ell:

  (i)  the design.  pe_coeffs(ell_hat, K) fits the minimax quintic to the window
       [ell_hat, 1]; a narrower window than the true one leaves the polynomial
       outside its minimax interval below ell_hat;
  (ii) the certificate.  E_K and S_pert(K) are evaluated on the believed window
       [ell_hat, 1], hence also the floor the stopping rule refuses below and the
       step count it returns.

Protocol per cell (d, ell, p, eps):
  * build the design for ell_hat and, for K = 1..KMAX, its believed certificate
    J_hat(K) = E_K(ell_hat) + c u S_pert(K; ell_hat);
  * K_hat = min{K : J_hat(K) <= eps}, refusal when none;
  * run the real kernel on a matrix whose true range is ell, with that design at
    that step count, and measure err = ||X_K - polar(M)||_2;
  * also evaluate the *same design* on the true window, J_true(K) = E_K(ell) +
    c u S_pert(K; ell), which is what the certificate would have said with the
    spectrum known.

 [A] design only: fixed K = 6, err(alpha)/err(1) over d, ell, p, alpha.  Shows the
     direction of the error without any stopping decision in the way.
 [B] the full decision: K_hat, measured error, err/eps, and the two failure
     classifications -- "missed" (a K was returned and the target was not met) and
     "refused but reachable" (no K was returned and some K with the true design
     would have met the target).
 [C] summary by alpha: which side of the estimate is safe, at what cost.
 [D] cost of being conservative: extra steps and error change for alpha < 1.

Kernel model as in pilots 1-12: X and B stored in p significant bits, A = XX^T not
stored, products accumulated in float64.  alpha = 1 reproduces the published cells.
"""
import json
import os
import numpy as np

from pilot7_pert_aware_design import q, pe_coeffs, make_M
from pilot10_subspace_theory import n2, orbit, start

HERE = os.path.dirname(os.path.abspath(__file__))
R = {}
CUSH, SAFE = 0.02, 1.01
C_CERT = 0.142
KMAX = 12
ALPHAS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0)
RATIOS = (1.2, 3.0, 10.0)     # eps = ratio * min_K J_true(K): always decidable with the true range
NG = 400


def cert(co, L):
    """E_K and the prefix-ratchet S_pert = max_{t<=K} S(t) on the window [L, 1]."""
    g = np.logspace(np.log10(L), 0.0, NG)
    x = g.copy()
    cur = None
    pref = []
    with np.errstate(over="ignore", invalid="ignore"):
        for a, b, c in co:
            d = np.abs(a + 3.0 * b * x ** 2 + 5.0 * c * x ** 4)
            cur = d[None, :] if cur is None else np.vstack([cur * d[None, :], d[None, :]])
            pref.append(float(cur.max()))
            x = a * x + b * x ** 3 + c * x ** 5
    if not np.all(np.isfinite(x)):
        return float("inf"), float("inf"), pref
    return float(np.max(np.abs(x - 1.0))), max(pref), pref


def designs(L_hat, L_true):
    """for each K: design for [L_hat,1], its certificate on the believed and the true window."""
    out = {}
    for K in range(1, KMAX + 1):
        co, _ = pe_coeffs(L_hat, K, CUSH, SAFE)
        EK_h, Sp_h, _ = cert(co, L_hat)
        EK_t, Sp_t, _ = cert(co, L_true)
        out[K] = {"co": co, "EK": EK_h, "Sp": Sp_h, "EK_true": EK_t, "Sp_true": Sp_t}
    return out


def J_of(tab, u, suffix=""):
    return {K: tab[K]["EK" + suffix] + C_CERT * u * tab[K]["Sp" + suffix] for K in tab}


def run_err(co, Mn, p, ref):
    with np.errstate(over="ignore", invalid="ignore"):
        X, _ = orbit(co, q(Mn, p), [p] * len(co))
    if X is None or not np.all(np.isfinite(X)):
        return None
    return n2(X - ref)


def pick(J, eps):
    ks = [K for K in J if J[K] <= eps]
    return (min(ks) if ks else None)


# ------------------------------------------------------------------ [A]
def part_A():
    print("=" * 112)
    print("[A] design only: same K = 6, design built for ell_hat = alpha*ell, run on the true spectrum")
    print("=" * 112)
    K = 6
    R["design_only"] = {}
    print(f"  {'d':>3s} {'ell':>7s} {'p':>3s} " +
          " ".join(f"{'a=' + str(a):>10s}" for a in ALPHAS))
    rows = {}
    for d in (64, 128):
        for L in (1e-2, 1e-3, 1e-4):
            M, ref = make_M(d, L, 1)
            Mn, _ = start(M, 1)
            for p in (8, 12):
                errs = {}
                for a in ALPHAS:
                    co, _ = pe_coeffs(a * L, K, CUSH, SAFE)
                    errs[a] = run_err(co, Mn, p, ref)
                base = errs[1.0]
                key = f"d{d}_L{L:g}_p{p}"
                rows[key] = {"err": {str(a): errs[a] for a in ALPHAS},
                             "ratio": {str(a): (errs[a] / base if errs[a] is not None and base else None)
                                       for a in ALPHAS}}
                R["design_only"][key] = rows[key]
                cells = ["div" if errs[a] is None else f"{errs[a] / base:10.2f}" for a in ALPHAS]
                print(f"  {d:3d} {L:7g} {p:3d} " + " ".join(f"{c:>10s}" for c in cells))
    print("\n  by alpha (design only, K=6, 12 cells):")
    for a in ALPHAS:
        v = [rows[k]["ratio"][str(a)] for k in rows if rows[k]["ratio"][str(a)] is not None]
        nd = sum(1 for k in rows if rows[k]["ratio"][str(a)] is None)
        print(f"    alpha={a:<5g}: median {np.median(v):6.2f}x  range "
              f"{min(v):.2f}-{max(v):.2f}x  diverged {nd}/{len(rows)}")
        R.setdefault("design_summary", {})[str(a)] = {
            "median": float(np.median(v)), "min": float(min(v)), "max": float(max(v)),
            "diverged": nd, "n": len(rows)}


# ------------------------------------------------------------------ [B]
def part_B():
    print("\n" + "=" * 112)
    print("[B] the decision: K_hat from the believed certificate, measured error on the true spectrum")
    print("=" * 112)
    R["stopping"] = {}
    print(f"  {'d':>3s} {'ell':>7s} {'p':>3s} {'eps':>9s} {'a':>5s} {'K':>3s} "
          f"{'J_hat':>9s} {'J_true':>9s} {'meas':>9s} {'meas/eps':>8s} {'meas/J_true':>11s} {'flag':>22s}")
    nrun = 0
    for d in (64, 128):
        for L in (1e-2, 1e-3, 1e-4):
            M, ref = make_M(d, L, 1)
            Mn, _ = start(M, 1)
            tab1 = designs(L, L)
            for p in (8, 12):
                u = 2.0 ** -p
                orc = {K: run_err(tab1[K]["co"], Mn, p, ref) for K in range(1, KMAX + 1)}
                best = min((v for v in orc.values() if v is not None), default=None)
                cellid = f"d{d}_L{L:g}_p{p}"
                J1_floor = J_of(tab1, u)
                kfl = min(J1_floor, key=J1_floor.get)
                R["stopping"][cellid] = {f"r{r:g}": {} for r in RATIOS}
                R["stopping"][cellid]["meta"] = {"K_floor": kfl, "J_floor": J1_floor[kfl],
                                                 "oracle_err": best}
                for ratio in RATIOS:
                    eps = ratio * J1_floor[kfl]
                    J1 = J_of(tab1, u)
                    K_true = pick(J1, eps)
                    reachable = best is not None and best <= eps
                    for a in ALPHAS:
                        tab = tab1 if a == 1.0 else designs(a * L, L)
                        Jh = J_of(tab, u)
                        Jt = J_of(tab, u, "_true")
                        K = pick(Jh, eps)
                        if K is None:
                            row = {"K": None, "K_true": K_true, "reachable": reachable,
                                   "flag": "refused"}
                            print(f"  {d:3d} {L:7g} {p:3d} {eps:7.0e} {a:5g} {'-':>3s} "
                                  f"{'-':>9s} {'-':>9s} {'-':>9s} {'-':>8s} {'-':>11s} "
                                  f"{'refused (reachable)' if reachable else 'refused (honest)':>22s}")
                        else:
                            meas = run_err(tab[K]["co"], Mn, p, ref)
                            nrun += 1
                            jt = Jt[K]
                            flag = "diverged" if meas is None else (
                                "missed" if meas > 1.5 * eps else "met")
                            row = {"K": K, "J_hat": Jh[K], "J_true": jt, "meas": meas,
                                   "ratio_eps": (meas / eps) if meas is not None else None,
                                   "ratio_cert": (meas / jt) if meas is not None and np.isfinite(jt) and jt > 0 else None,
                                   "ratio_hat": (meas / Jh[K]) if meas is not None and Jh[K] > 0 else None,
                                   "K_true": K_true, "reachable": reachable, "flag": flag,
                                   "eps": eps}
                            s_m = f"{meas:9.2e}" if meas is not None else "      div"
                            s_r = f"{meas / eps:8.2f}" if meas is not None else "       -"
                            s_c = (f"{row['ratio_cert']:11.2f}"
                                   if row["ratio_cert"] is not None else "          -")
                            print(f"  {d:3d} {L:7g} {p:3d} {eps:7.0e} {a:5g} {K:3d} "
                                  f"{Jh[K]:9.2e} {jt:9.2e} {s_m} {s_r} {s_c} {flag:>22s}")
                        row["eps"] = eps
                        row["alpha"] = a
                        R["stopping"][cellid][f"r{ratio:g}"][str(a)] = row
    print(f"\n  kernel runs: {nrun}")


# ------------------------------------------------------------------ [C]
def part_C():
    print("\n" + "=" * 112)
    print("[C] summary by alpha over every decided cell")
    print("=" * 112)
    R["summary"] = {}
    for a in ALPHAS:
        cells = []
        for key, dct in R["stopping"].items():
            for ek, sub in dct.items():
                if ek.startswith("r") and str(a) in sub:
                    cells.append(sub[str(a)])
        n = len(cells)
        f = lambda name: sum(1 for c in cells if c["flag"] == name)
        rat = [c["ratio_eps"] for c in cells if c.get("ratio_eps") is not None]
        crt = [c["ratio_cert"] for c in cells if c.get("ratio_cert") is not None]
        rhat = [c["ratio_hat"] for c in cells if c.get("ratio_hat") is not None]
        R["summary"][str(a)] = {
            "n": n, "met": f("met"), "missed": f("missed"), "diverged": f("diverged"),
            "refused_honest": sum(1 for c in cells if c["flag"] == "refused" and not c["reachable"]),
            "refused_reachable": sum(1 for c in cells if c["flag"] == "refused" and c["reachable"]),
            "ratio_eps_median": float(np.median(rat)) if rat else None,
            "ratio_eps_max": float(max(rat)) if rat else None,
            "ratio_eps_min": float(min(rat)) if rat else None,
            "ratio_cert_median": float(np.median(crt)) if crt else None,
            "ratio_cert_max": float(max(crt)) if crt else None,
            "ratio_hat_median": float(np.median(rhat)) if rhat else None,
            "ratio_hat_max": float(max(rhat)) if rhat else None}
        s = R["summary"][str(a)]
        print(f"  alpha={a:<5g} n={n:3d}  met {s['met']:3d}  missed {s['missed']:3d}  "
              f"diverged {s['diverged']:3d}  refused-honest {s['refused_honest']:3d}  "
              f"refused-reachable {s['refused_reachable']:3d}  "
              f"meas/eps med {s['ratio_eps_median']} max {s['ratio_eps_max']}  "
              f"meas/J_true med {s['ratio_cert_median']}  "
              f"meas/J_hat med {s['ratio_hat_median']} max {s['ratio_hat_max']}")


# ------------------------------------------------------------------ [D]
def part_D():
    print("\n" + "=" * 112)
    print("[D] what each side costs: step count and error against alpha = 1")
    print("=" * 112)
    R["cost"] = {}
    for a in ALPHAS:
        dk, dr = [], []
        for key, dct in R["stopping"].items():
            for ek, sub in dct.items():
                if not ek.startswith("r"):
                    continue
                c1, ca = sub.get("1.0"), sub.get(str(a))
                if not c1 or not ca or c1["K"] is None or ca["K"] is None:
                    continue
                dk.append(ca["K"] - c1["K"])
                if c1.get("meas") and ca.get("meas") is not None:
                    dr.append(ca["meas"] / c1["meas"])
        R["cost"][str(a)] = {"dK_median": float(np.median(dk)) if dk else None,
                             "dK_min": int(min(dk)) if dk else None,
                             "dK_max": int(max(dk)) if dk else None,
                             "err_ratio_median": float(np.median(dr)) if dr else None,
                             "err_ratio_max": float(max(dr)) if dr else None,
                             "n": len(dk)}
        s = R["cost"][str(a)]
        print(f"  alpha={a:<5g}: dK {s['dK_min']}..{s['dK_max']} (median {s['dK_median']}), "
              f"err(alpha)/err(1) median {s['err_ratio_median']}, max {s['err_ratio_max']} (n={s['n']})")


def main():
    part_A()
    part_B()
    part_C()
    part_D()
    out = os.path.join(HERE, "pilot13_results.json")
    with open(out, "w") as f:
        json.dump(R, f, indent=1, default=float)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
