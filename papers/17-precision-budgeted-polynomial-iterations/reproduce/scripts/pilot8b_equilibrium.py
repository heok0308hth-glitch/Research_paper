#!/usr/bin/env python3
"""
Pilot 8b - Why the rounding floor saturates: a contraction-reinjection fixed point.

Pilot 5b found that the p-bit floor of a Polar-Express iteration is
`err ~ C * u` with C = 10-14 and does not fall when K grows.  Pilot 7 showed the
linear "whole-orbit gain" model cannot produce that (it predicts a floor falling
like 2^-K), and pilot 6 localised 87-90% of the floor in the first-step
quantisation channel.  Neither explains why the plateau is *flat* and why its
constant is what it is.

This script formalises the floor as the fixed point of a one-step recursion

        e_{t+1} = g_t * e_t + s_t
        g_t = || D p_t(x_t) ||_2      (amplification at the orbit point actually reached)
        s_t = || round(p_t(x_t)) - p_t(x_t) ||_2   (re-injection, relative rounding)

and tests it three ways:

  [1] plateau: err_t as a function of t for a fixed design, at several p, for
      logspace(ell,1) spectra.  Is the tail a stationary level, and is it C*u?
  [2] recursion: does e_{t+1} = g_t e_t + s_t reproduce the measured e_t
      trajectory (g_t measured exactly by finite differences -- the map is a
      homogeneous polynomial, so the difference quotient is the derivative, not
      an approximation)?
  [3] fixed point: is the tail level equal to s_bar/(1 - g_bar), and does the
      same formula predict the *design* dependence of C (the matched-window
      design vs the wide-window design)?

It also shows which step sets the level: extending the tail by repeating the
last (Newton-type, p'(1)=0) step drives the error to ~s_bar, while extending it
by repeating a damped safety step keeps it at s_bar/(1-1/safety).  C is
therefore a property of the tail cycle, not of K.

Kernel model identical to pilots 1-7.
"""
import json
import os
import numpy as np

from pilot7_pert_aware_design import pe_coeffs, q, stats, make_M

HERE = os.path.dirname(os.path.abspath(__file__))
R = {"config": {}, "plateau": {}, "recursion": {}, "tail_extend": {}, "predicted_C": {}}
D = 64
PS = (6, 8, 11, 16, 24)
D_CAL = 64


def step_map(X, a, b, c):
    A = X @ X.T
    return a * X + (b * A + c * (A @ A)) @ X


def frechet_gain(X, a, b, c, eps=1e-7, n=3):
    """spectral norm of the step's Frechet derivative at X (exact: homogeneous map)."""
    g = None
    rs = np.random.RandomState(7)
    for _ in range(n):
        U = rs.randn(*X.shape)
        U /= np.linalg.norm(U, 2)
        Y = U * eps
        d = step_map(X + Y, a, b, c) - step_map(X, a, b, c)
        v = np.linalg.norm(d, 2) / eps
        g = v if g is None else max(g, v)
    return float(g)


def run_trace(co, M, ref, p, extra=0, extra_mode="none", ell=None):
    """apply the coefficient sequence, then optionally extend the tail by repeating
    a step; record err / injected rounding / gain at every step."""
    X = q(M / np.linalg.norm(M, 2), p)
    seq = list(co) + ([co[-1]] * extra if extra_mode == "last" else
                      [co[-2]] * extra if extra_mode == "prev" else
                      [co[0]] * extra if extra_mode == "first" else [])
    rows = []
    for i, (a, b, c) in enumerate(seq):
        g = frechet_gain(X, a, b, c)
        Y = step_map(X, a, b, c)
        Xn = q(Y, p)
        s = float(np.linalg.norm(Xn - Y, 2))
        X = Xn
        rows.append({"i": i, "err": float(np.linalg.norm(X - ref, 2)), "g": g, "s": s,
                     "E_exact": float(np.linalg.norm(step_map(X, a, b, c) - ref, 2)) if False else None})
    return rows


def one(L, K, p, d=D, seed=1, extra=0, extra_mode="none"):
    M, ref = make_M(d, L, seed)
    co, _ = pe_coeffs(L, K, 0.02, 1.01)
    rows = run_trace(co, M, ref, p, extra, extra_mode)
    return rows, co


def main():
    print("=" * 100)
    print("Pilot 8b - saturation equilibrium of the rounding floor")
    print("=" * 100)

    # ---------------------------------------------------------------- [1] plateau
    print("\n[1] plateau: err_t over the iteration, several p, two designs (d=64)")
    for L, K in ((1e-3, 8), (5e-2, 6)):
        print(f"\n  design ell={L:g}, K={K}")
        R["plateau"][f"L{L:g}"] = {}
        for p in PS:
            u = 2.0 ** -p
            rows, co = one(L, K, p)
            e = [r["err"] for r in rows]
            tail = e[-3:]
            C = np.mean(tail) / u
            R["plateau"][f"L{L:g}"][str(p)] = {"u": u, "err": e, "C": float(C),
                                               "C_geomean": float(np.exp(np.mean(np.log(tail))) / u)}
            print(f"    p={p:2d}  err: " + " ".join(f"{v:8.2e}" for v in e) +
                  f"   C(tail 3)= {C:5.1f}   err/u={np.mean(tail)/u:5.1f}")
            if p == 8:
                grid = np.logspace(np.log10(L), 0, 400)
                for Kx in range(2, 11):
                    cox, _ = pe_coeffs(L, Kx, 0.02, 1.01)
                    ex, sx, _ = stats(cox, grid, Kx)
                    R["plateau"][f"L{L:g}"].setdefault("pred_vs_K", {})[str(Kx)] = {
                        "E_K": ex, "S_pert": sx, "pred_0p142": ex + 0.142 * u * sx}

    # certificate growth vs flat measurement (same design family, growing K)
    print("\n  [1b] the certificate's floor estimate grows with K while the measurement is flat")
    for L in (1e-3, 5e-2):
        print(f"    ell={L:g}")
        print(f"      {'K':>3s} {'err(p=8)':>10s} {'E_K':>10s} {'S_pert':>9s} {'0.142*u*S':>10s} {'pred':>10s}")
        rows_by_K = {}
        for K in range(2, 11):
            rows, co = one(L, K, 8)
            e = rows[-1]["err"]
            grid = np.logspace(np.log10(L), 0, 400)
            ex, sx, _ = stats(co, grid, K)
            rows_by_K[K] = {"err": e, "E_K": ex, "S_pert": sx, "floor_est": 0.142 * 2 ** -8 * sx}
            print(f"      {K:3d} {e:10.2e} {ex:10.2e} {sx:9.1f} {0.142*2**-8*sx:10.2e} "
                  f"{ex + 0.142*2**-8*sx:10.2e}")
        R["plateau"][f"L{L:g}"]["K_scan_p8"] = rows_by_K

    # ---------------------------------------------------------------- [2] recursion
    print("\n[2] one-step recursion  e_{t+1} = g_t e_t + s_t  (measured g_t, s_t)")
    for L, K in ((1e-3, 8), (5e-2, 6)):
        print(f"\n  design ell={L:g}, K={K}")
        R["recursion"][f"L{L:g}"] = {}
        for p in (8, 11):
            u = 2.0 ** -p
            rows, _ = one(L, K, p)
            pred = rows[0]["s"]; e_prev = rows[0]["err"] - rows[0]["s"]
            lines = []
            for r in rows[1:]:
                pred = r["g"] * e_prev + r["s"]
                lines.append((r["i"], r["err"], pred, r["g"], r["s"] / u))
                e_prev = r["err"]
            R["recursion"][f"L{L:g}"][str(p)] = {"rows": [
                {"i": r["i"], "err": r["err"], "g": r["g"], "s_over_u": r["s"] / u,
                 "pred": lines[j][2] if j < len(lines) else None} for j, r in enumerate(rows)]}
            print(f"    p={p:2d}")
            print(f"      {'t':>2s} {'err_t':>10s} {'g_t':>7s} {'s_t/u':>7s} {'g*e+s':>10s} {'pred/meas':>9s}")
            for j, r in enumerate(rows):
                pv = lines[j - 1][2] if j >= 1 else float("nan")
                print(f"      {r['i']:2d} {r['err']:10.3e} {r['g']:7.3f} {r['s']/u:7.3f} {pv:10.3e} "
                      f"{pv/r['err']:9.2f}")

    # ---------------------------------------------------------------- [3] fixed point
    print("\n[3] fixed point of the tail: e* = s_bar/(1-g_bar); does it give C?")
    print(f"      {'design':>10s} {'p':>3s} {'g_tail':>7s} {'s/u':>7s} {'e*/u':>8s} {'C_meas':>8s} {'ratio':>6s}")
    R["predicted_C"] = {}
    for L, K in ((1e-3, 8), (5e-2, 6)):
        for p in PS:
            u = 2.0 ** -p
            rows, _ = one(L, K, p)
            tail = rows[-3:]
            g = float(np.exp(np.mean([np.log(max(r["g"], 1e-12)) for r in tail])))
            s = float(np.mean([r["s"] for r in tail]))
            estar = s / (1 - g) if g < 1 else float("nan")
            C_meas = np.mean([r["err"] for r in tail]) / u
            R["predicted_C"][f"L{L:g}_p{p}"] = {"g_tail": g, "s_over_u": s / u,
                                                "e_star_over_u": estar / u, "C_meas": C_meas,
                                                "ratio": (estar / u) / C_meas}
            print(f"      {L:10g} {p:3d} {g:7.3f} {s/u:7.3f} {estar/u:8.1f} {C_meas:8.1f} "
                  f"{(estar/u)/C_meas:6.2f}")

    # ---------------------------------------------------------------- [4] tail extension
    print("\n[4] what sets the level: extend the tail by repeating a step (p=8, 8 extra steps)")
    for L, K in ((1e-3, 8), (5e-2, 6)):
        for mode in ("none", "last", "prev"):
            rows, _ = one(L, K, 8, extra=8 if mode != "none" else 0, extra_mode=mode)
            print(f"    ell={L:<6g} repeat={mode:>5s}: " +
                  " ".join(f"{r['err']:7.2e}" for r in rows))
            R["tail_extend"][f"L{L:g}_{mode}"] = [r["err"] for r in rows]

    # ---------------------------------------------------------------- [5] mechanism
    print("\n[5] mechanism: is the plateau a singular-SUBSPACE error (which the polar map cannot repair)?")
    print("      decomposition  err_total = ||X_K - ref||,  sigma_err = ||X_K - polar(X_K)||,")
    print("                     sub_err  = ||polar(X_K) - ref||")
    print(f"      {'ell':>8s} {'p':>3s} {'K':>2s} {'total':>9s} {'sigma':>9s} {'subspace':>9s} {'sub/total':>9s} {'C=err/u':>8s}")
    R["mechanism"] = {}
    for L, K in ((1e-1, 5), (3e-2, 5), (1e-2, 6), (3e-3, 6), (1e-3, 6), (1e-4, 7)):
        for p in (8, 12):
            u = 2.0 ** -p
            M, ref = make_M(D, L, 1)
            co, _ = pe_coeffs(L, K, 0.02, 1.01)
            X = q(M / np.linalg.norm(M, 2), p)
            for a, b, c in co:
                A = X @ X.T
                B = q(b * A + c * (A @ A), p)
                X = q(a * X + B @ X, p)
            w, sv, vt = np.linalg.svd(X, full_matrices=False)
            pol = w @ vt
            tot = float(np.linalg.norm(X - ref, 2))
            sig = float(np.linalg.norm(X - pol, 2))
            sub = float(np.linalg.norm(pol - ref, 2))
            R["mechanism"][f"L{L:g}_p{p}"] = {"total": tot, "sigma_err": sig, "subspace": sub,
                                              "K": K, "u": u, "C": tot / u}
            print(f"      {L:8g} {p:3d} {K:2d} {tot:9.3e} {sig:9.3e} {sub:9.3e} {sub/tot:9.3f} {tot/u:8.1f}")

    # how strongly does C depend on the conditioning?  (p=8, converged K)
    print("\n      C(ell) at p=8, converged K:  (fit C ~ ell^-beta)")
    ls, cs = [], []
    for L in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
        M, ref = make_M(D, L, 1)
        co, _ = pe_coeffs(L, 10, 0.02, 1.01)
        X = q(M / np.linalg.norm(M, 2), 8)
        with np.errstate(over="ignore", invalid="ignore"):
            for a, b, c in co:
                A = X @ X.T
                B = q(b * A + c * (A @ A), 8)
                X = q(a * X + B @ X, 8)
        if not np.all(np.isfinite(X)):
            print(f"        ell={L:8g}  diverged at p=8, K=10 (excluded)")
            continue
        w, sv, vt = np.linalg.svd(X, full_matrices=False)
        C = float(np.linalg.norm(X - ref, 2)) / 2.0 ** -8
        ls.append(L); cs.append(C)
        R.setdefault("C_of_ell", {})[f"{L:g}"] = C
        print(f"        ell={L:8g}  C={C:8.2f}  u/ell={2**-8/L:9.3f}  C*sqrt(ell)/u={C*np.sqrt(L):8.4f}")
    lx, ly = np.log(ls), np.log(cs)
    beta = -np.polyfit(lx, ly, 1)[0]
    R["C_of_ell_beta"] = float(beta)
    print(f"        log-log slope: C ~ ell^-{beta:.3f}")

    # ---------------------------------------------------------------- [6] where should the bits go?
    print("\n[6] if the floor is an early subspace error, bits belong to the FIRST steps")
    print("      ell=1e-3, K=6, per-step precision pattern -> err, polar-part, subspace-part")
    R["bit_placement"] = {}
    for name, ps_list in (("all p=8", [8] * 6),
                          ("step0 exact", [60, 8, 8, 8, 8, 8]),
                          ("steps0-1 exact", [60, 60, 8, 8, 8, 8]),
                          ("last step exact", [8, 8, 8, 8, 8, 60]),
                          ("all p=9 (same budget as step0-exact)", [9] * 6),
                          ("all p=10", [10] * 6)):
        M, ref = make_M(D, 1e-3, 1)
        co, _ = pe_coeffs(1e-3, 6, 0.02, 1.01)
        X = q(M / np.linalg.norm(M, 2), ps_list[0])
        for (a, b, c), pp in zip(co, ps_list):
            A = X @ X.T
            B = q(b * A + c * (A @ A), pp)
            X = q(a * X + B @ X, pp)
        w, sv, vt = np.linalg.svd(X, full_matrices=False)
        pol = w @ vt
        tot = float(np.linalg.norm(X - ref, 2))
        bits = sum(ps_list)
        R["bit_placement"][name] = {"err": tot, "bits": bits,
                                    "sigma_err": float(np.linalg.norm(X - pol, 2)),
                                    "subspace": float(np.linalg.norm(pol - ref, 2))}
        print(f"      {name:>38s}  err={tot:9.3e}  bits={bits:3d}  sigma={R['bit_placement'][name]['sigma_err']:.2e}  "
              f"subspace={R['bit_placement'][name]['subspace']:.2e}")

    out = os.path.join(HERE, "pilot8b_results.json")
    with open(out, "w") as f:
        json.dump(R, f, indent=1, default=float)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
