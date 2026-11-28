#!/usr/bin/env python3
"""
Pilot 8 - Muon orthogonalisation inside a real (numpy) training loop.

The paper's section 6.3 claims that the certificate

        err ~ E_K + c * u * S_pert(K),   c = 0.142,  u = 2^-p

predicts the orthogonalisation error of a Polar-Express update computed in a
p-bit storage format, and that the certified stopping rule K*(u) therefore says
how many orthogonalisation steps a Muon update should take at that precision.
Section 6.3 was empty because no GPU was available.  This script fills it with a
CPU-scale experiment that is honest about its size:

  task    : 2-layer MLP (hidden 128) on a synthetic 10-class problem whose labels
            are the argmax of a fixed nonlinear teacher net (input dim 32);
  optimiser: Muon on the hidden weight (momentum 0.95, Nesterov, polar
            orthogonalisation of the momentum), Adam on the output layer/biases;
  kernel  : the polar step is executed by the *same* model as pilots 1-7
            (X and B stored in p significant bits, products in float64).

Two measurement modes, because a coupled run confounds precision with the
trajectory it produces:

  [A] coupled runs  - precision inside the real loop: does the update error and
      the loss move with p and with the stopping rule?  (final/best val loss,
      accuracy, divergence, steps to best);
  [B] frozen runs   - one exact-trajectory momentum sequence, replayed through
      the p-bit kernel for every (p_X, p_B, K, design).  Same matrices, only
      the arithmetic changes, so the certificate and the max-structure of the
      bit allocation can be tested directly on *real* momentum spectra.

Output: pilot8_results.json.
"""
import json
import os
import sys
import time
import numpy as np

from pilot7_pert_aware_design import pe_coeffs, q, stats

HERE = os.path.dirname(os.path.abspath(__file__))
C_CERT = 0.142
GRID_N = 400
D_IN, H_HID, N_CLS = 32, 128, 10
N_TR, N_VA = 12000, 3000
BATCH = 256
STEPS = 500
MUON_LR = 0.02
MUON_MOM = 0.95
ADAM_LR = 3e-3
VAL_EVERY = 20
CUSHION, SAFETY = 0.02, 1.01


# ------------------------------------------------------------------ task
def make_task(seed=0):
    rs = np.random.RandomState(seed)
    W1 = rs.randn(64, D_IN) / np.sqrt(D_IN)
    b1 = 0.1 * rs.randn(64)
    W2 = rs.randn(N_CLS, 64) / np.sqrt(64)
    X = rs.randn(N_TR + N_VA, D_IN)
    lg = np.maximum(X @ W1.T + b1, 0) @ W2.T
    y = lg.argmax(1)
    return X[:N_TR], y[:N_TR], X[N_TR:], y[N_TR:]


def ce_loss(logits, y):
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    return float(-np.log(p[np.arange(len(y)), y] + 1e-30).mean())


def grads(W1, b1, W2, b2, X, y):
    z = X @ W1.T + b1
    h = np.maximum(z, 0)
    lg = h @ W2.T + b2
    zz = lg - lg.max(1, keepdims=True)
    p = np.exp(zz)
    p /= p.sum(1, keepdims=True)
    p[np.arange(len(y)), y] -= 1.0
    p /= len(y)
    dW2 = p.T @ h
    db2 = p.sum(0)
    dz = (p @ W2) * (z > 0)
    return dz.T @ X, dz.sum(0), dW2, db2


# ------------------------------------------------------------------ p-bit polar step
def polar_bits(U, co, pX, pB, ref=None):
    """K-step Polar Express on U with pX-bit iterate, pB-bit B.  Returns (X_K, ref, sv/s1)."""
    Uf = np.asarray(U, float)
    if ref is None or pX is None and ref is None:
        w, s, vt = np.linalg.svd(Uf, full_matrices=False)
        ref = w @ vt
        sv = s / (s[0] if s[0] > 0 else 1.0)
    X = q(Uf / (np.linalg.norm(Uf, 2) * (1 + 1e-12)), pX)
    for a, b, c in co:
        A = X @ X.T
        B = q(b * A + c * (A @ A), pB)
        X = q(a * X + B @ X, pX)
    return X, ref, sv


def precompute_refs(upds):
    out = []
    for U in upds:
        w, s, vt = np.linalg.svd(U, full_matrices=False)
        s1 = s[0] if s[0] > 0 else 1.0
        out.append((w @ vt, s / s1))
    return out


def certificate(co, ell, K, u):
    grid = np.logspace(np.log10(max(ell, 1e-12)), 0, GRID_N)
    EK, S, _ = stats(co, grid, K)
    return EK, S, EK + C_CERT * u * S


def kstar(p, L, Kmax=14):
    u = 2.0 ** -p
    grid = np.logspace(np.log10(L), 0, GRID_N)
    for K in range(2, Kmax + 1):
        co, _ = pe_coeffs(L, K, CUSHION, SAFETY)
        EK, S, _ = stats(co, grid, K)
        if EK <= u * S:
            return K
    return Kmax


# ------------------------------------------------------------------ [A] coupled training run
def run_coupled(policy, p, seed, pB=None, L=1e-3, steps=STEPS, collect=False, Xtr=None, ytr=None, pert=0.0):
    if Xtr is None:
        Xtr, ytr, Xva, yva = make_task()
    else:
        _, _, Xva, yva = make_task()
    rs = np.random.RandomState(100 + seed)
    if policy == "kstar":
        K = kstar(p, L) if p is not None else 8
    else:
        K = int(policy.replace("fixed", ""))
    co, _ = pe_coeffs(L, K, CUSHION, SAFETY)

    W1 = rs.randn(H_HID, D_IN) / np.sqrt(D_IN)
    b1 = np.zeros(H_HID)
    W2 = rs.randn(N_CLS, H_HID) / np.sqrt(H_HID)
    b2 = np.zeros(N_CLS)
    m1 = np.zeros_like(W1)
    aW2, ab1, ab2 = np.zeros_like(W2), np.zeros_like(b1), np.zeros_like(b2)
    vW2, vb1, vb2 = np.zeros_like(W2), np.zeros_like(b1), np.zeros_like(b2)
    u = 0.0 if p is None else 2.0 ** -p
    pX = 60 if p is None else p
    pB_use = pX if pB is None else pB
    scale = 0.2 * np.sqrt(max(H_HID, D_IN))

    orth, val_hist, upds, trace = [], [], [], []
    bad = False
    t0 = time.time()
    for t in range(steps):
        idx = rs.choice(N_TR, BATCH, replace=False)
        dW1, db1, dW2, db2 = grads(W1, b1, W2, b2, Xtr[idx], ytr[idx])
        m1 = MUON_MOM * m1 + dW1
        upd = dW1 + MUON_MOM * m1
        if pert > 0:
            d = rs.randn(*upd.shape)
            upd = upd + pert * np.linalg.norm(upd, 2) * d / np.linalg.norm(d, 2)
        if collect:
            upds.append(upd.copy())
        O, ref, sv = polar_bits(upd, co, pX, pB_use)
        e = float(np.linalg.norm(O - ref, 2))
        ell = float(sv[-1])
        EK, S, pred = certificate(co, ell, K, u)
        orth.append((e, pred, ell, EK, S))
        if (t % 50 == 0 or t == steps - 1) and not collect:
            trace.append({"t": t, "err": e, "pred": pred, "ell": ell, "S_pert": S})
        W1 = W1 - MUON_LR * scale * O
        for par, g, a_, v_ in ((W2, dW2, aW2, vW2), (b1, db1, ab1, vb1), (b2, db2, ab2, vb2)):
            a_ *= 0.9
            a_ += 0.1 * g
            v_ *= 0.999
            v_ += 0.001 * (g * g)
            par -= ADAM_LR * (a_ / (1 - 0.9 ** (t + 1))) / (np.sqrt(v_ / (1 - 0.999 ** (t + 1))) + 1e-8)
        if not np.all(np.isfinite(W1)) or np.abs(W1).max() > 1e3:
            bad = True
            break
        if (t + 1) % VAL_EVERY == 0 or t == steps - 1:
            lg = np.maximum(Xva @ W1.T + b1, 0) @ W2.T + b2
            vl = ce_loss(lg, yva)
            tl = ce_loss(np.maximum(Xtr[:2000] @ W1.T + b1, 0) @ W2.T + b2, ytr[:2000])
            val_hist.append((t + 1, tl, vl))
            if not np.isfinite(vl):
                bad = True
                break

    lg = np.maximum(Xva @ W1.T + b1, 0) @ W2.T + b2
    acc = float((lg.argmax(1) == yva).mean())
    oa = np.array(orth) if orth else np.zeros((1, 5))
    best = min(val_hist, key=lambda z: z[2]) if val_hist else (None, None, None)
    res = {
        "policy": policy, "K": K, "p": p, "pB": pB_use, "seed": seed, "u": u, "L": L, "pert": pert,
        "finished": not bad, "steps_run": len(oa), "acc": acc,
        "final_train": val_hist[-1][1] if val_hist else None,
        "final_val": val_hist[-1][2] if val_hist else None,
        "best_val": float(best[2]) if best[2] is not None else None,
        "best_step": int(best[0]) if best[0] is not None else None,
        "orth_mean": float(oa[:, 0].mean()), "orth_med": float(np.median(oa[:, 0])),
        "orth_max": float(oa[:, 0].max()),
        "pred_med": float(np.median(oa[:, 1])),
        "ratio_med": float(np.median(oa[:, 1] / oa[:, 0])),
        "frac_within2": float(np.mean(np.abs(np.log2(oa[:, 1] / oa[:, 0])) <= 1.0)),
        "ell_med": float(np.median(oa[:, 2])),
        "ell_p05": float(np.percentile(oa[:, 2], 5)),
        "EK_med": float(np.median(oa[:, 3])), "S_med": float(np.median(oa[:, 4])),
        "seconds": time.time() - t0, "val_hist": val_hist, "trace": trace,
    }
    if collect:
        res["upds"] = [u_.astype(np.float32) for u_ in upds]
    return res, (Xtr, ytr)


# ------------------------------------------------------------------ [B] frozen replay
def eval_frozen(refs, upds, co, pX, pB, K, u):
    errs, preds, lls = [], [], []
    for (ref, sv), U in zip(refs, upds):
        X = q(U / (np.linalg.norm(U, 2) * (1 + 1e-12)), pX)
        for a, b, c in co:
            A = X @ X.T
            B = q(b * A + c * (A @ A), pB)
            X = q(a * X + B @ X, pX)
        e = float(np.linalg.norm(X - ref, 2))
        ell = float(sv[-1])
        EK, S, pred = certificate(co, ell, K, u)
        errs.append(e)
        preds.append(pred)
        lls.append(ell)
    errs, preds = np.array(errs), np.array(preds)
    return {
        "orth_mean": float(errs.mean()), "orth_med": float(np.median(errs)),
        "orth_p90": float(np.percentile(errs, 90)),
        "pred_med": float(np.median(preds)), "ratio_med": float(np.median(preds / errs)),
        "frac_within2": float(np.mean(np.abs(np.log2(preds / errs)) <= 1.0)),
        "ell_med": float(np.median(lls)),
    }


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def agg(rows):
    if not rows:
        return {}
    fin = [r for r in rows if r["final_val"] is not None]
    return {
        "n": len(rows), "K": rows[0]["K"], "p": rows[0]["p"], "pB": rows[0]["pB"], "L": rows[0]["L"],
        "orth_mean": float(np.mean([r["orth_mean"] for r in rows])),
        "ratio_med": float(np.median([r["ratio_med"] for r in rows])),
        "frac_within2": float(np.mean([r["frac_within2"] for r in rows])),
        "final_val": float(np.mean([r["final_val"] for r in fin])) if fin else None,
        "best_val": float(np.mean([r["best_val"] for r in fin])) if fin else None,
        "best_step": float(np.mean([r["best_step"] for r in fin])) if fin else None,
        "final_train": float(np.mean([r["final_train"] for r in fin])) if fin else None,
        "acc": float(np.mean([r["acc"] for r in rows])),
        "diverged": int(sum(0 if r["finished"] else 1 for r in rows)),
        "ell_med": float(np.median([r["ell_med"] for r in rows])),
        "seconds": float(np.sum([r["seconds"] for r in rows])),
    }


def main(quick=False, budget=None):
    t0 = time.time()
    out = os.path.join(HERE, "pilot8_quick.json" if quick else "pilot8_results.json")

    def dump():
        with open(out, "w") as f:
            json.dump(R, f, indent=1, default=float)

    def spent():
        return budget is not None and time.time() - t0 > budget

    if os.path.exists(out):
        with open(out) as f:
            R = json.load(f)
        R.setdefault("runs", [])
        R.setdefault("frozen", {})
        R.setdefault("dose", {})
        done = {r.get("key") for r in R["runs"]}
        print(f"[resume] {len(R['runs'])} runs stored")
    else:
        R = {"config": {"D_IN": D_IN, "H": H_HID, "classes": N_CLS, "steps": STEPS, "batch": BATCH,
                        "n_train": N_TR, "muon_lr": MUON_LR, "muon_mom": MUON_MOM, "adam_lr": ADAM_LR,
                        "c_cert": C_CERT, "design": "cushion 0.02, safety 1.01"},
             "runs": [], "frozen": {}, "dose": {}}
        done = set()
    R["kstar"] = {f"{L:g}": {str(p): kstar(p, L) for p in (6, 8, 11, 16, 24)} for L in (1e-3, 5e-2)}
    print("=" * 108)
    print("Pilot 8 - Muon MLP (numpy) + Polar Express in p bits")
    print("certified K*(u):", json.dumps(R["kstar"]))

    seeds = (0,) if quick else (0, 1, 2)
    ps = (8, 16) if quick else (None, 6, 8, 11, 16, 24)
    pols = ("fixed8",) if quick else ("fixed5", "fixed8", "kstar")
    task = make_task()

    if "summary_A" not in R:
        print("\n" + "=" * 108)
        print("[A] coupled runs: Muon with p-bit polar inside the training loop")
        print("=" * 108)
        for L in (1e-3, 0.15):
            print(f"\n  -- design window ell={L:g}   certified K*(u) = {R['kstar'].get(f'{L:g}', {})}")
            for p in ps:
                for pol in pols:
                    for sd in seeds:
                        key = f"A|{L:g}|{p}|{pol}|{sd}"
                        if key in done:
                            continue
                        r, _ = run_coupled(pol, p, sd, L=L, Xtr=task[0], ytr=task[1])
                        r["config"], r["key"] = f"L{L:g}|p{p}|{pol}", key
                        R["runs"].append(r)
                        dump()
                        print(f"  L={L:<6g} p={str(p):>4s} {pol:>6s} K={r['K']:2d} sd={sd}  "
                              f"orth={r['orth_mean']:.3e} ratio={r['ratio_med']:5.2f} in2x={r['frac_within2']:.2f}  "
                              f"bestval={r['best_val']:.3f}@{r['best_step']} endval={r['final_val']:.3f} "
                              f"acc={r['acc']:.3f} ell={r['ell_med']:.2f} "
                              f"{'DIV' if not r['finished'] else ''} ({r['seconds']:.1f}s)", flush=True)
                        if spent():
                            print("[budget] pausing after this run"); return
        R["summary_A"] = {k: agg([r for r in R["runs"] if r.get("config") == k]) for k in
                          sorted({r["config"] for r in R["runs"] if r.get("key", "").startswith("A|")})}
        dump()

    if "summary_A2" not in R:
        print("\n" + "=" * 108)
        print("[A2] bit allocation inside the loop (K=8, p_X x p_B)")
        print("=" * 108)
        alloc = ([(8, 8), (16, 8)] if quick else
                 [(8, 8), (8, 16), (8, 24), (11, 8), (11, 16), (16, 8), (16, 16), (16, 24), (24, 8), (24, 16)])
        for (pX, pB) in alloc:
            for sd in ((0,) if quick else (0, 1)):
                key = f"A2|{pX}|{pB}|{sd}"
                if key in done:
                    continue
                r, _ = run_coupled("fixed8", pX, sd, pB=pB, L=0.15, Xtr=task[0], ytr=task[1])
                r["config"], r["key"] = f"alloc({pX},{pB})", key
                R["runs"].append(r)
                dump()
                print(f"  (pX,pB)=({pX},{pB}) sd={sd} orth={r['orth_mean']:.3e} ratio={r['ratio_med']:.3f} "
                      f"bestval={r['best_val']:.3f} acc={r['acc']:.3f} "
                      f"{'DIV' if not r['finished'] else 'ok'} ({r['seconds']:.1f}s)", flush=True)
                if spent():
                    print("[budget] pausing after this run"); return
        R["summary_A2"] = {k: agg([r for r in R["runs"] if r.get("config") == k]) for k in
                           sorted({r["config"] for r in R["runs"] if r.get("key", "").startswith("A2|")})}
        dump()

    if not R.get("dose"):
        print("\n" + "=" * 108)
        print("[C] dose-response: exact arithmetic + random perturbation of the Muon update, size eps*||upd||")
        print("=" * 108)
        for eps in (0.0, 1e-2, 3e-2, 6e-2, 1.3e-1, 2.6e-1):
            rows = []
            for sd in ((0,) if quick else (0, 1, 2)):
                key = f"C|{eps:g}|{sd}"
                if key in done:
                    continue
                r, _ = run_coupled("fixed8", None, sd, L=0.15, pert=eps, Xtr=task[0], ytr=task[1])
                r["config"], r["key"] = f"pert{eps:g}", key
                rows.append(r)
                R["runs"].append(r)
            a = agg(rows)
            R["dose"][f"{eps:g}"] = a
            dump()
            print(f"  eps={eps:6.3f}  best_val={a['best_val']:.4f}  final_val={a['final_val']:.4f}  "
                  f"final_train={a['final_train']:.4f}  acc={a['acc']:.4f}", flush=True)
            if spent():
                print("[budget] pausing after this run"); return

    # ------------------------------------------------ frozen replay
    exact, _ = run_coupled("fixed8", None, 0, L=0.15, collect=True, Xtr=task[0], ytr=task[1])
    upds = [u_.astype(float) for u_ in exact["upds"]]
    refs = precompute_refs(upds)
    lls = np.array([sv[-1] for _, sv in refs])
    meta = {"n": len(upds), "ell_med": float(np.median(lls)), "ell_p10": float(np.percentile(lls, 10)),
            "ell_p90": float(np.percentile(lls, 90)), "sv2_med": float(np.median([sv[1] for _, sv in refs])),
            "exact_best_val": exact["best_val"], "exact_acc": exact["acc"]}
    fresh = R["frozen"].get("meta") != meta
    R["frozen"]["meta"] = meta
    if fresh:
        print("\n" + "=" * 108)
        print("[B] frozen replay: exact-trajectory momentum replayed through the p-bit kernel")
        print("=" * 108)
        for k in list(R["frozen"]):
            if k != "meta":
                del R["frozen"][k]
    print(f"  momentum sequence: {meta['n']} steps of {upds[0].shape}, ell={meta['ell_med']:.3f} "
          f"(p10 {meta['ell_p10']:.3f}, p90 {meta['ell_p90']:.3f}), sv2/sv1={meta['sv2_med']:.3f}; "
          f"exact run best val {meta['exact_best_val']:.4f}, acc {meta['exact_acc']:.4f}")

    PVC = (6, 8, 11, 16, 24)
    for L in (1e-3, 5e-2):
        key = f"L{L:g}"
        R["frozen"].setdefault(key, {})
        R["frozen"][key].setdefault("K_curves", {})
        R["frozen"][key].setdefault("alloc", {})
        print(f"\n  -- design ell={L:g}: error vs K on the frozen sequence")
        print(f"     {'K':>3s} " + " ".join(f"{'p=' + str(p):>10s}" for p in PVC))
        for K in range(2, 11):
            if str(K) in R["frozen"][key]["K_curves"]:
                row = {int(k): v for k, v in R["frozen"][key]["K_curves"][str(K)].items()}
                print(f"     {K:3d} " + " ".join(f"{row[p]['orth_mean']:10.2e}" for p in PVC))
                continue
            co, _ = pe_coeffs(L, K, CUSHION, SAFETY)
            row = {p: eval_frozen(refs, upds, co, p, p, K, 2.0 ** -p) for p in PVC}
            R["frozen"][key]["K_curves"][str(K)] = row
            dump()
            print(f"     {K:3d} " + " ".join(f"{row[p]['orth_mean']:10.2e}" for p in PVC))
            if spent():
                print("[budget] pausing"); return
        curves = {int(k): {int(p): v for p, v in row.items()}
                  for k, row in R["frozen"][key]["K_curves"].items()}
        cert = {}
        for p in PVC:
            kk = kstar(p, L)
            bestK = min(curves, key=lambda k: curves[k][p]["orth_mean"])
            flat = [k for k in curves if curves[k][p]["orth_mean"] <= 1.1 * curves[bestK][p]["orth_mean"]]
            cert[str(p)] = {"K_star": kk, "K_argmin": bestK, "K_flat10pct": flat,
                            "err_at_Kstar": curves[kk][p]["orth_mean"],
                            "err_at_argmin": curves[bestK][p]["orth_mean"],
                            "ratio_at_Kstar": curves[kk][p]["ratio_med"],
                            "cost_of_rule": curves[kk][p]["orth_mean"] / curves[bestK][p]["orth_mean"]}
            c = cert[str(p)]
            print(f"     p={p:2d}: K*(u)={kk:2d} argmin K={bestK} flat(10%)={flat} err(K*)={c['err_at_Kstar']:.2e} "
                  f"err_min={c['err_at_argmin']:.2e} cost={c['cost_of_rule']:.2f}x ratio@K*={c['ratio_at_Kstar']:.2f}")
        R["frozen"][key]["certified"] = cert
        dump()

        print(f"\n  -- frozen allocation at K=8, design ell={L:g}")
        co, _ = pe_coeffs(L, 8, CUSHION, SAFETY)
        for pX in (8, 11, 16, 24, 60):
            for pB in (8, 11, 16, 24, 60):
                k2 = f"{pX},{pB}"
                if k2 not in R["frozen"][key]["alloc"]:
                    R["frozen"][key]["alloc"][k2] = eval_frozen(refs, upds, co, pX, pB, 8, 2.0 ** -min(pX, 60))
                    dump()
                    if spent():
                        print("[budget] pausing"); return
        tab = R["frozen"][key]["alloc"]
        for kk, v in sorted(tab.items(), key=lambda z: (int(z[0].split(",")[0]), int(z[0].split(",")[1]))):
            pX, pB = [int(s) for s in kk.split(",")]
            mx = max(tab[f"{pX},60"]["orth_mean"], tab[f"60,{pB}"]["orth_mean"])
            sm = tab[f"{pX},60"]["orth_mean"] + tab[f"60,{pB}"]["orth_mean"]
            print(f"     ({pX:2d},{pB:2d}) err={v['orth_mean']:.3e}  X-only={tab[f'{pX},60']['orth_mean']:.2e} "
                  f"B-only={tab[f'60,{pB}']['orth_mean']:.2e}  err/max={v['orth_mean']/mx:.2f} "
                  f"err/sum={v['orth_mean']/sm:.2f}")

    # ------------------------------------------------ post-hoc: rule quality on real spectra
    print("\n" + "=" * 108)
    print("[D] stopping-rule quality on the frozen spectra: crossing rule vs epsilon-target rule")
    print("=" * 108)
    ell_med = R["frozen"]["meta"]["ell_med"]
    R["rules"] = {}
    for L in (1e-3, 5e-2):
        key = f"L{L:g}"
        curves = {int(k): {int(p): v for p, v in row.items()}
                  for k, row in R["frozen"][key]["K_curves"].items()}
        R["rules"][key] = {}
        for p in PVC:
            u = 2.0 ** -p
            EK, SP, EST = {}, {}, {}
            for K in curves:
                co, _ = pe_coeffs(L, K, CUSHION, SAFETY)
                grid_ = np.logspace(np.log10(ell_med), 0, GRID_N)
                ek, sp, _ = stats(co, grid_, K)
                EK[K], SP[K], EST[K] = ek, sp, ek + C_CERT * u * sp
            argmin = min(curves, key=lambda k: curves[k][p]["orth_mean"])
            best = curves[argmin][p]["orth_mean"]
            plateau = float(np.median([curves[k][p]["orth_mean"] for k in curves if k >= argmin]))
            C_meas = plateau / u
            c_eff = C_meas / SP[argmin]
            k_cross = min((k for k in curves if EK[k] <= u * SP[k]), default=max(curves))
            eps_target = 1.5 * plateau
            k_target = min((k for k in curves if EST[k] <= eps_target), default=max(curves))
            k_target_ceff = min((k for k in curves if EK[k] + c_eff * u * SP[k] <= eps_target), default=max(curves))
            R["rules"][key][str(p)] = {
                "plateau": plateau, "C_meas": C_meas, "c_eff": c_eff, "K_argmin": argmin,
                "err_best": best, "err_plateau": plateau,
                "K_cross": k_cross, "err_cross": curves[k_cross][p]["orth_mean"],
                "cost_cross": curves[k_cross][p]["orth_mean"] / best,
                "K_target": k_target, "err_target": curves[k_target][p]["orth_mean"],
                "cost_target": curves[k_target][p]["orth_mean"] / best,
                "K_target_ceff": k_target_ceff, "err_target_ceff": curves[k_target_ceff][p]["orth_mean"],
                "cost_target_ceff": curves[k_target_ceff][p]["orth_mean"] / best,
            }
            r = R["rules"][key][str(p)]
            print(f"  L={L:<6g} p={p:2d}: plateau={C_meas:5.1f}u (c_eff={c_eff:.3f})  argmin K={argmin} "
                  f"| crossing K={k_cross} cost={r['cost_cross']:6.2f}x "
                  f"| eps-target K={k_target} cost={r['cost_target']:5.2f}x (c=0.142) "
                  f"/ K={k_target_ceff} cost={r['cost_target_ceff']:5.2f}x (c_eff)")
        dump()

    grid = [r for r in R["runs"] if r.get("key", "").startswith(("A|", "A2|"))]
    xs = [np.log(max(r["pred_med"], 1e-300)) for r in grid if r["final_val"]]
    ys = [np.log(max(r["orth_mean"], 1e-300)) for r in grid if r["final_val"]]
    bs = [r["best_val"] for r in grid if r["final_val"]]
    R["corr"] = {"spearman_logpred_vs_logorth": spearman(xs, ys),
                 "spearman_logpred_vs_bestval": spearman(xs, bs),
                 "spearman_logorth_vs_bestval": spearman(ys, bs), "n": len(xs)}
    print("\ncorrelations over coupled runs:", json.dumps(R["corr"]))
    dump()
    print(f"\ntotal {time.time() - t0:.1f}s")
    print("wrote", out)


if __name__ == "__main__":
    budget = None
    for a in sys.argv[1:]:
        if a.startswith("--budget="):
            budget = float(a.split("=")[1])
    main(quick="--quick" in sys.argv, budget=budget)
