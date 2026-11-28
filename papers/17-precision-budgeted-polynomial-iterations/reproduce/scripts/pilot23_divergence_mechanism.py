#!/usr/bin/env python3
"""
Pilot 23 - what makes a low-precision Muon run diverge, and why the task decides.

Pilot 21 censused 11 seed-runs x 7 conditions: E5M2 at the shipped safety s = 1.01 blows
up in 6 of 11 (diabetes 3/3, digits 2/5, residual 1/3), E4M3 once, E5M2 at s = 1.035 in
none.  It left two questions: why the task decides, and by what mechanism a few percent
of damping closes the boundary.  Three candidates are separable, and each is measured.

 [W] The window.  The design is a composition of minimax quintics, each certified on a
     window [lo_t, u_t] that the exact orbit of the edges visits; outside it the map's
     gain |p'(x)| exceeds 1, and at x well above u_t the quintic's tail takes over.  The
     state the kernel actually sees has sigma_max/||M||_F = 1/sqrt(PR_2) < 1 and a bottom
     far below lo_0, so it does not follow the window's orbit.  Measured here: the window
     of every step, the exact orbit of the edges, the excess the window tolerates
     (the largest start x0 = 1 + delta whose p-bit scalar orbit over K steps stays below
     10), and the map's gain beyond the edge.

 [A] The event.  Instrumented census of the diverging conditions, recording at every
     (sampled) step and layer: the top of every state in the six steps by power estimate,
     its ratio to that step's certified upper edge (the window violation), the rank
     deficiency and dynamic range of the momentum, the polar window error, the relative
     weight step, and the norm of the update actually applied.  The question the trace
     settles is whether the run drifts (a training-loop instability, mechanism [B]) or
     breaks in a single call of the iteration (an escape inside the kernel).

 [B] The loop.  For the curvature feedback story, the loop gain
        G = lambda_max(diag(step)^{1/2} H diag(step)^{1/2})
     is measured on the exact trajectory by power iteration on Hessian-vector products
     (central differences of the analytic gradient) every 25 steps, per task and seed.
     G > 1 is the condition under which a weight-space error grows; G < 1 everywhere
     excludes the mechanism.

 [C] The counterfactuals, arithmetic unchanged, one thing changed:
       (c0) E5M2 at s = 1.01 (the shipped point) - the baseline;
       (c1) E5M2 at s = 1.035 - the known cure;
       (c2) E5M2 at s = 1.01 with the state rescaled into the window whenever a step
            would leave it: if the escape is the mechanism, this must close the same runs;
       (c3) E5M2 at s = 1.01 with the learning rate cut by 10%: if the mode were a
            step-size (edge-of-stability) effect, this would close them instead.
     A run is a divergence when some weight exceeds 1e3 in magnitude or a weight, loss or
     update stops being finite.

 [D] The damping priced.  The composite polar factor at s = 1.02, 1.03, 1.035, 1.05
     against 1.01 on 80 real momentum matrices: how much of a step the safety costs (it
     should cost none: the design returns 1 to 1) and how much window it buys.

Kernel and loop as in pilot 21 (Frobenius input normalisation, per-sample LayerNorm,
momentum 0.95 Nesterov, Muon on hidden weights and Adam at the output, K = 6,
ell_0 = 1e-2).  Writes pilot23_results.json.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pilot21_loss_phase_and_scheduler as p21  # noqa: E402
from pilot21_loss_phase_and_scheduler import (  # noqa: E402
    CUSH, ELL0, P_BITS, WINDOW, exact_polar, get_data, layers_of, loss_and_grads,
    norm_est, preset, quant, window_err,
)
from pilot7_pert_aware_design import pe_coeffs, q as qbits  # noqa: E402

RES = os.path.join(HERE, "pilot23_results.json")
R = {}
DIV_THR = 1e3
TASKS = (("digits", 5), ("resmlp", 3), ("diabetes", 3))


def load():
    global R
    if os.path.exists(RES):
        with open(RES) as f:
            R = json.load(f)


def save():
    tmp = RES + ".tmp"
    with open(tmp, "w") as f:
        json.dump(R, f, indent=1)
    os.replace(tmp, RES)


def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else None


def span(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return (float(min(xs)), float(max(xs))) if xs else (None, None)


# ---------------------------------------------------------------- top by power iteration
def top_est(X, iters=4):
    m, n = X.shape
    if n <= m:
        v = np.ones(n) / np.sqrt(n)
        for _ in range(iters):
            w = X @ v
            w /= np.linalg.norm(w) + 1e-300
            v = X.T @ w
            v /= np.linalg.norm(v) + 1e-300
        return float(np.linalg.norm(X @ v))
    u = np.ones(m) / np.sqrt(m)
    for _ in range(iters):
        w = X.T @ u
        w /= np.linalg.norm(w) + 1e-300
        u = X @ w
        u /= np.linalg.norm(u) + 1e-300
    return float(np.linalg.norm(X.T @ u))


def polar_tracked(M, co, cast_fmt, step_fmts, nev, win, clamp=False):
    """pilot21's kernel with the top of every state recorded; clamp keeps it in the window."""
    X = quant(M / nev, cast_fmt)
    bits = P_BITS[cast_fmt]
    tops = [top_est(X)]
    for t, ((a, b, c), fmt) in enumerate(zip(co, step_fmts)):
        A = X @ X.T
        B = quant(b * A + c * (A @ A), fmt)
        X = quant(a * X + B @ X, fmt)
        tp = top_est(X)
        if clamp:
            ub = win[min(t + 1, len(win) - 1)][1]
            if tp > ub:
                X = X * (ub / tp)
                tp = ub
        tops.append(tp)
        bits += P_BITS[fmt]
    viol = None
    if win:
        viol = max(tops[t] / win[min(t, len(win) - 1)][1] for t in range(len(tops)))
    return X, bits, tops, (float(viol) if viol is not None else None)


# ---------------------------------------------------------------- tracked training
def run_tracked(task, cond, seed, steps=1200, K=6, lr=0.01, lr_scale=1.0, clamp=False,
                inst_every=1, G_every=0):
    Xtr, ytr, Xva, yva = get_data(task, seed)
    rs = np.random.RandomState(1000 + seed)
    Ws, W5, dims = layers_of(task, rs)
    b5 = np.zeros(W5.shape[0])
    ms = [np.zeros_like(W) for W in Ws]
    aW5 = np.zeros_like(W5); vW5 = np.zeros_like(W5)
    ab5 = np.zeros_like(b5); vb5 = np.zeros_like(b5)
    co, cast_fmt, step_fmts, safety = preset(cond, K)
    _, win = pe_coeffs(ELL0, K, CUSH, safety)
    trace = []
    div, reason = False, "completed"
    t0 = time.time()
    t = 0
    for t in range(steps):
        idx = rs.choice(len(ytr), min(256, len(ytr)), replace=False)
        loss, gs, dW5, db5 = loss_and_grads(task, Ws, W5, b5, Xtr[idx], ytr[idx])
        inst = (t % inst_every == 0)
        rec = {"t": t, "loss": float(loss)} if inst else None
        for l, (W, g) in enumerate(zip(Ws, gs)):
            ms[l] = 0.95 * ms[l] + g
            upd = g + 0.95 * ms[l]
            nev = norm_est(upd)
            O, bts, tops, viol = polar_tracked(upd, co, cast_fmt, step_fmts, nev, win, clamp)
            step_lr = lr * lr_scale * 0.2 * np.sqrt(max(W.shape))
            onorm = float(np.linalg.norm(O))
            wn = float(np.linalg.norm(W))
            if inst:
                sv = np.linalg.svd(upd, compute_uv=False)
                ref, svp, vt = exact_polar(upd)
                rec[f"l{l}"] = {
                    "smax_over_F": float(sv[0] / max(np.sqrt((sv ** 2).sum()), 1e-300)),
                    "decades": float(np.log10(sv[0] / max(sv[-1], 1e-300))),
                    "rank_def": int((sv < 1e-10 * sv[0]).sum()),
                    "winviol": viol, "top_K": float(tops[-1]),
                    "win_err": float(window_err(O, ref, vt, svp, WINDOW * nev)),
                    "onorm": onorm, "dW_rel": float(step_lr * onorm / (wn + 1e-300)),
                    "Wnorm": wn, "step": float(step_lr)}
            W -= step_lr * O
            if not np.all(np.isfinite(W)) or np.abs(W).max() > DIV_THR:
                div, reason = True, "weight_blowup"
                break
        if div:
            break
        aW5 = 0.9 * aW5 + 0.1 * dW5; vW5 = 0.999 * vW5 + 0.001 * dW5 ** 2
        W5 -= 3e-3 * (aW5 / (1 - 0.9 ** (t + 1))) / (np.sqrt(vW5 / (1 - 0.999 ** (t + 1))) + 1e-8)
        ab5 = 0.9 * ab5 + 0.1 * db5; vb5 = 0.999 * vb5 + 0.001 * db5 ** 2
        b5 -= 3e-3 * (ab5 / (1 - 0.9 ** (t + 1))) / (np.sqrt(vb5 / (1 - 0.999 ** (t + 1))) + 1e-8)
        if rec is not None:
            rec["W2"] = float(np.sqrt(sum(np.linalg.norm(w) ** 2 for w in Ws)))
            rec["gnorm"] = float(np.sqrt(sum(np.linalg.norm(g) ** 2 for g in gs)))
            if G_every and (t % G_every == 0):
                rec["G"] = loop_gain(task, Ws, W5, b5, Xtr[idx], ytr[idx], lr * lr_scale)
            trace.append(rec)
        if not np.isfinite(loss):
            div, reason = True, "nonfinite_loss"
            break
    return {"task": task, "cond": cond, "seed": seed, "lr_scale": lr_scale, "clamp": clamp,
            "finished": not div, "stop_reason": reason, "steps_run": t + 1, "safety": safety,
            "trace": trace, "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- [B] loop gain
def flat_grads(gs, dW5, db5):
    return np.concatenate([g.ravel() for g in gs] + [dW5.ravel(), db5.ravel()])


def loop_gain(task, Ws, W5, b5, X, y, lr, iters=5, seed=3):
    """lambda_max of diag(step)^{1/2} H diag(step)^{1/2}, by power iteration on an HVP."""
    shapes = [W.shape for W in Ws] + [W5.shape, b5.shape]
    c = np.concatenate([np.full(int(W.size), lr * 0.2 * np.sqrt(max(W.shape))) for W in Ws]
                       + [np.full(W5.size, 3e-3), np.full(b5.size, 3e-3)])
    sq = np.sqrt(c)
    n = sq.size
    rs = np.random.RandomState(seed)
    v = rs.randn(n)
    v /= np.linalg.norm(v)
    eps = 1e-5
    lam = None
    for _ in range(iters):
        gp = _grad_along(task, Ws, W5, b5, X, y, shapes, sq * v, eps)
        gm = _grad_along(task, Ws, W5, b5, X, y, shapes, sq * v, -eps)
        if gp is None:
            return None
        Hv = (gp - gm) / (2 * eps)
        w = sq * Hv
        nw = np.linalg.norm(w)
        if not np.isfinite(nw) or nw <= 0:
            return None
        v = w / nw
        lam = float(v @ w)
    return lam


def _grad_along(task, Ws, W5, b5, X, y, shapes, d, eps):
    out, i = [], 0
    note = 0
    Ws2 = []
    for W in Ws:
        n = W.size
        Ws2.append(W + eps * d[i:i + n].reshape(W.shape))
        i += n
    W52 = W5 + eps * d[i:i + W5.size].reshape(W5.shape)
    i += W5.size
    b52 = b5 + eps * d[i:i + b5.size]
    try:
        _, gs, dW5, db5 = loss_and_grads(task, Ws2, W52, b52, X, y)
    except Exception:
        return None
    g = flat_grads(gs, dW5, db5)
    return g if np.all(np.isfinite(g)) else None


# ---------------------------------------------------------------- [W] the window
def stageW():
    print("=" * 118)
    print("[W] the certified window of every step, the exact orbit of its edges, and the excess it")
    print("    tolerates: the largest delta with a p-bit scalar orbit from 1 + delta staying below 10")
    print("=" * 118)
    out = {}
    for s in (1.0, 1.01, 1.02, 1.035, 1.05):
        co, win = pe_coeffs(ELL0, 6, CUSH, s)
        l, u = fit_edge = 1.0, 1.0
        lo, hi = 0.02, 1.0
        orb = []
        for (a, b, c) in co:
            lo = a * lo + b * lo ** 3 + c * lo ** 5
            hi = a * hi + b * hi ** 3 + c * hi ** 5
            orb.append((lo, hi))
        row = {"win": [[float(x[0]), float(x[1])] for x in win], "edge_orbit": orb}
        for p in (4, 5, 8):
            row[f"delta_p{p}"] = tolerated_excess(p, s)
        out[f"s{s}"] = row
        print(f"  s={s:5.3f}: upper edge of the steps " +
              " ".join(f"{w[1]:.4f}" for w in win))
        print(f"          tolerated excess: p=4 {row['delta_p4']:+.4f}  p=5 {row['delta_p5']:+.4f}"
              f"  p=8 {row['delta_p8']:+.4f}")
    R["w_windows"] = out
    # the gain beyond the upper edge, step 5 and 6 (the tight ones)
    co, win = pe_coeffs(ELL0, 6, CUSH, 1.01)
    gains = {}
    for t in (3, 4, 5):
        a, b, c = co[t]
        ub = win[t][1]
        prof = []
        for m in (1.0, 1.02, 1.05, 1.1, 1.2, 1.5, 2.0):
            x = ub * m
            nx = a * x + b * x ** 3 + c * x ** 5
            prof.append((m, float(abs(nx / x))))
        gains[f"step{t + 1}"] = prof
        print(f"  step {t + 1} (upper edge {ub:.4f}): |p(x)/x| at x/edge = " +
              " ".join(f"{m:g}:{g:.2f}" for m, g in prof))
    R["w_gain_beyond_edge"] = gains
    save()


def tolerated_excess(p, s, K=6, lo0=ELL0):
    """largest delta in [0, 0.6] whose p-bit scalar orbit from 1+delta stays below 10."""
    co, _ = pe_coeffs(lo0, K, CUSH, s)
    best = None
    grid = np.arange(0.0, 0.6001, 0.0025)
    for delta in grid:
        x = 1.0 + delta
        ok = True
        for (a, b, c) in co:
            x = qbits(a * x + b * x ** 3 + c * x ** 5, p)
            if not np.isfinite(x) or abs(x) > 10.0:
                ok = False
                break
        if not ok:
            break
        best = float(delta)
    return best if best is not None else -1.0


# ---------------------------------------------------------------- [A] the census
def stageA(inst_every=10, conds=None):
    print("\n" + "=" * 118)
    print("[A] instrumented census: does the run drift, or does one call of the iteration explode?")
    print("=" * 118)
    todo = [("digits", sd) for sd in range(5)] + [("diabetes", sd) for sd in range(3)] \
        + [("resmlp", sd) for sd in range(3)]
    allc = [("e5m2", dict()), ("e5m2_s1035", dict()), ("e5m2", dict(clamp=True)),
            ("e5m2", dict(lr_scale=0.9))]
    if conds:
        allc = [c for c in allc if c[0] + ("_clamp" if c[1].get("clamp") else "")
                + ("_lr09" if c[1].get("lr_scale") else "") in conds]
    out = R.get("a_census", {})
    for cond, kw in allc:
        for task, seed in todo:
            steps = 400
            r = run_tracked(task, cond, seed, steps=steps, inst_every=inst_every, **kw)
            key = f"{task}_s{seed}_{cond}{'_clamp' if kw.get('clamp') else ''}" \
                  f"{'_lr09' if kw.get('lr_scale') else ''}"
            tr = r["trace"]
            lax = [(k, d) for s_ in tr for k, d in s_.items()
                   if k.startswith("l") and k[1:].isdigit()]
            viol = [d["winviol"] for _, d in lax]
            dec = [d["decades"] for _, d in lax]
            rd = [d["rank_def"] for _, d in lax]
            dw = max([d["dW_rel"] for _, d in lax], default=None)
            out[key] = dict(finished=r["finished"], steps=r["steps_run"], reason=r["stop_reason"],
                            safety=r["safety"], clamp=bool(kw.get("clamp")),
                            lr_scale=kw.get("lr_scale", 1.0),
                            viol_max=span(viol)[1], viol_med=med(viol),
                            decades_med=med(dec), decades_max=span(dec)[1],
                            rankdef_share=float(np.mean([x > 0 for x in rd])) if rd else None,
                            dW_rel_max=dw, trace=tr, seconds=r["seconds"])
            print(f"  {key:>32}: {'OK  ' if r['finished'] else 'DIV '} steps {r['steps_run']:5d} "
                  f"winviol {out[key]['viol_med']:.4f}/{out[key]['viol_max']:.4f} "
                  f"decades {out[key]['decades_med']:.1f} rankdef {out[key]['rankdef_share']:.2f} "
                  f"max dW {dw:.2e}")
            save()
    R["a_census"] = out
    save()
    for tag in ("e5m2", "e5m2_s1035", "e5m2_clamp", "e5m2_lr09"):
        if tag == "e5m2":
            keys = [k for k in out if k.endswith("_e5m2")]
        else:
            keys = [k for k in out if k.endswith(tag)]
        fin = [out[k]["finished"] for k in keys]
        print(f"  {tag:>12}: {sum(fin)}/{len(fin)} finish")


def stageB():
    print("\n" + "=" * 118)
    print("[B] the loop gain G = lambda_max(diag(step)^1/2 H diag(step)^1/2), exact trajectory")
    print("=" * 118)
    out = {}
    for task, nseed in (("digits", 2), ("diabetes", 2), ("resmlp", 1)):
        for seed in range(nseed):
            r = run_tracked(task, "fp32", seed, steps=400, inst_every=25, G_every=25)
            G = [s_["G"] for s_ in r["trace"] if "G" in s_ and s_["G"] is not None]
            out[f"{task}_s{seed}"] = dict(G=[float(g) for g in G],
                                          G_max=max(G) if G else None, G_med=med(G),
                                          frac_above_1=float(np.mean([g > 1 for g in G])) if G else None)
            print(f"  {task:>8} seed{seed}: G max {max(G):6.3f} median {med(G):6.3f} "
                  f"share>1 {np.mean([g > 1 for g in G]) * 100:4.0f}%")
            save()
    R["b_gain"] = out
    save()


def stageD():
    print("\n" + "=" * 118)
    print("[D] the damping priced on 80 real momentum matrices: what s costs and what it buys")
    print("=" * 118)
    Xtr, ytr, _, _ = get_data("digits", 0)
    rs = np.random.RandomState(1000)
    Ws, W5, dims = layers_of("digits", rs)
    b5 = np.zeros(W5.shape[0])
    ms = [np.zeros_like(W) for W in Ws]
    aW5 = np.zeros_like(W5); vW5 = np.zeros_like(W5)
    ab5 = np.zeros_like(b5); vb5 = np.zeros_like(b5)
    ups = []
    cof = preset("fp32", 6)[0]
    for t in range(400):
        idx = rs.choice(len(ytr), 256, replace=False)
        _, gs, dW5, db5 = loss_and_grads("digits", Ws, W5, b5, Xtr[idx], ytr[idx])
        for l, (W, g) in enumerate(zip(Ws, gs)):
            ms[l] = 0.95 * ms[l] + g
            upd = g + 0.95 * ms[l]
            if t % 20 == 0 and l in (0, 1):
                ups.append(upd.copy())
            O, _ = p21.polar(upd, cof, "fp32", ["fp32"] * 6)
            W -= 0.01 * 0.2 * np.sqrt(max(W.shape)) * O
        aW5 = 0.9 * aW5 + 0.1 * dW5; vW5 = 0.999 * vW5 + 0.001 * dW5 ** 2
        W5 -= 3e-3 * (aW5 / (1 - 0.9 ** (t + 1))) / (np.sqrt(vW5 / (1 - 0.999 ** (t + 1))) + 1e-8)
        ab5 = 0.9 * ab5 + 0.1 * db5; vb5 = 0.999 * vb5 + 0.001 * db5 ** 2
        b5 -= 3e-3 * (ab5 / (1 - 0.9 ** (t + 1))) / (np.sqrt(vb5 / (1 - 0.999 ** (t + 1))) + 1e-8)
    print(f"  {len(ups)} momentum matrices")
    rows = {}
    for s in (1.0, 1.01, 1.02, 1.03, 1.035, 1.05):
        co, win = pe_coeffs(ELL0, 6, CUSH, s)
        tops, fro, viols = [], [], []
        for M in ups:
            X = M / norm_est(M)
            tp = [top_est(X)]
            for (a, b, c) in co:
                A = X @ X.T
                X = a * X + (b * A + c * (A @ A)) @ X
                tp.append(top_est(X))
            tops.append(tp[-1])
            fro.append(float(np.linalg.norm(X, "fro")))
            viols.append(max(tp[t] / win[min(t, len(win) - 1)][1] for t in range(len(tp))))
        rows[s] = dict(top_med=med(tops), frob_med=med(fro), viol_med=med(viols))
        print(f"  s={s:5.3f}: composite top {med(tops):.5f}  Frobenius {med(fro):.4f}  "
              f"window violation (exact orbit) {med(viols):.4f}")
    b = rows[1.01]
    R["d_safety"] = {str(s): dict(rows[s], top_rel=rows[s]["top_med"] / b["top_med"],
                                  frob_rel=rows[s]["frob_med"] / b["frob_med"],
                                  viol_rel=rows[s]["viol_med"] / b["viol_med"]) for s in rows}
    for s in rows:
        print(f"  s={s:5.3f} / 1.01: top {rows[s]['top_med'] / b['top_med']:.4f}  "
              f"frobenius {rows[s]['frob_med'] / b['frob_med']:.4f}  "
              f"violation {rows[s]['viol_med'] / b['viol_med']:.4f}")
    save()


def stageE(conds=("e5m2", "e5m2_s1035")):
    """the escape itself: the top of the state at every step of the failing call, against the
    exact orbit and the certified edges."""
    print("\n" + "=" * 118)
    print("[E] the escape: tops inside one failing call, quantized against exact, and the edges")
    print("=" * 118)
    out = R.get("e_escape_traces", {})
    for cond in conds:
        for task, seed in (("diabetes", 0), ("diabetes", 2), ("digits", 0), ("digits", 1)):
            Xtr, ytr, _, _ = get_data(task, seed)
            rs = np.random.RandomState(1000 + seed)
            Ws, W5, dims = layers_of(task, rs)
            b5 = np.zeros(W5.shape[0])
            ms = [np.zeros_like(W) for W in Ws]
            aW5 = np.zeros_like(W5); vW5 = np.zeros_like(W5)
            ab5 = np.zeros_like(b5); vb5 = np.zeros_like(b5)
            co, cast_fmt, step_fmts, safety = preset(cond, 6)
            _, win = pe_coeffs(ELL0, 6, CUSH, safety)
            hit = None
            for t in range(400):
                idx = rs.choice(len(ytr), 256, replace=False)
                loss, gs, dW5, db5 = loss_and_grads(task, Ws, W5, b5, Xtr[idx], ytr[idx])
                for l, (W, g) in enumerate(zip(Ws, gs)):
                    ms[l] = 0.95 * ms[l] + g
                    upd = g + 0.95 * ms[l]
                    nev = norm_est(upd)
                    X = quant(upd / nev, cast_fmt)
                    tops = [top_est(X, 6)]
                    for (a, b, c), fmt in zip(co, step_fmts):
                        A = X @ X.T
                        B = quant(b * A + c * (A @ A), fmt)
                        X = quant(a * X + B @ X, fmt)
                        tops.append(top_est(X, 6))
                    Xe = upd / nev
                    tex = [top_est(Xe, 6)]
                    for (a, b, c) in co:
                        A = Xe @ Xe.T
                        Xe = a * Xe + (b * A + c * (A @ A)) @ Xe
                        tex.append(top_est(Xe, 6))
                    sv = np.linalg.svd(upd, compute_uv=False)
                    if np.linalg.norm(X) > 1e3:
                        hit = dict(task=task, seed=seed, cond=cond, t=t, layer=l,
                                   u=[float(w[1]) for w in win], tops=[float(x) for x in tops],
                                   exact=[float(x) for x in tex],
                                   top_over_F=float(sv[0] / nev),
                                   decades=float(np.log10(sv[0] / max(sv[-1], 1e-300))),
                                   rank_def=int((sv < 1e-10 * sv[0]).sum()),
                                   onorm=float(np.linalg.norm(X)))
                        break
                    W -= 0.01 * 0.2 * np.sqrt(max(W.shape)) * X
                if hit:
                    break
                aW5 = 0.9 * aW5 + 0.1 * dW5; vW5 = 0.999 * vW5 + 0.001 * dW5 ** 2
                W5 -= 3e-3 * (aW5 / (1 - 0.9 ** (t + 1))) / (np.sqrt(vW5 / (1 - 0.999 ** (t + 1))) + 1e-8)
                ab5 = 0.9 * ab5 + 0.1 * db5; vb5 = 0.999 * vb5 + 0.001 * db5 ** 2
                b5 -= 3e-3 * (ab5 / (1 - 0.9 ** (t + 1))) / (np.sqrt(vb5 / (1 - 0.999 ** (t + 1))) + 1e-8)
            key = f"{task}_s{seed}_{cond}"
            if hit:
                out[key] = hit
                print(f"  {key:>24}: escape at step {hit['t']} layer {hit['layer']}, "
                      f"|O| {hit['onorm']:.2e}")
                print("      exact   " + " ".join(f"{x:9.4f}" for x in hit["exact"]))
                print("      edges   " + " ".join(f"{x:9.4f}" for x in hit["u"]))
                print("      E5M2    " + " ".join(f"{x:9.4f}" for x in hit["tops"]))
            else:
                out[key] = dict(task=task, seed=seed, cond=cond, escape=None)
                print(f"  {key:>24}: no escape in 400 steps")
            save()
    R["e_escape_traces"] = out
    save()


def main():
    load()
    stageW()
    stageA()
    stageB()
    stageD()
    print("\nwrote", RES)


if __name__ == "__main__":
    load()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        main()
    elif which == "W":
        stageW()
    elif which == "B":
        stageB()
    elif which == "D":
        stageD()
    elif which == "E":
        stageE()
    else:
        stageA(inst_every=10, conds=which.split(","))
