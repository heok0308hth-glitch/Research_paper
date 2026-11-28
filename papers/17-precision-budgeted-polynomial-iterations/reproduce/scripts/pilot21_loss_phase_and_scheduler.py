#!/usr/bin/env python3
"""
Pilot 21 - two questions about precision inside a real (numpy) training loop.

 [A] Loss phase transition.  Pilot 8 (2-layer MLP, bf16) was null.  Here the loop is
     deeper and uses standard, unscaled real data (sklearn digits classification,
     sklearn diabetes regression), a realistic Muon learning rate and per-sample
     layer normalisation, so nothing is arranged to be ill-conditioned.  Conditions:
     fp32, bf16 (p=8), E4M3 (p=5), E5M2 (p=4), E5M2 with the corrected safety, and
     the mixed rule (bf16 cast + two bf16 steps, E5M2 afterwards, safety 1.035).
     Recorded: train/val loss trajectory, accuracy, divergence, the polar update
     error on the real momentum matrices and the bits each rule spends.

 [B] PSAP - Power-Spectrum Adaptive Polar.  Every layer and step has its own momentum
     spectrum, and the certificate err ~ E_K + c u S says precision only pays where
     the iteration has already converged.  The allocator works from two free sensors
     that no extra arithmetic is needed for: the per-step progress
         pi_t = ||X_t - X_{t-1}||_F / ||X_t||_F
     (the kernel produces X_t anyway) and the rounding residual
         r_t  = ||X_q - X_exact||_F
     (both sides of the quantiser are in hand).  While pi_t >> r_t the step is
     truncation-limited and stays in the cheap format; once pi_t falls to the noise
     level a further cheap step cannot pay, so the tail widens; when pi_t reaches the
     target the call stops.  A spectral sensor (power iteration + Hutchinson trace) is
     measured alongside and scored against the same criterion, to see whether the
     cheap spectrum readout buys anything the free sensors do not.

Kernel model as in pilots 1-19: tensors stored in the format's precision (fp8 with a
per-tensor power-of-two scale, a delayed-scaling picture), products accumulated in
float64.  sigma_max for the input normalisation is an 8-iteration power estimate with
a 5% safety factor, clamped by the Frobenius norm, as in a real Muon kernel.
"""
import argparse
import json
import os
import sys
import time
import numpy as np
from sklearn.datasets import load_digits, load_diabetes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pilot7_pert_aware_design import q, pe_coeffs  # noqa: E402
from pilot15_fp8_kernel import qfmt, FMT  # noqa: E402

RES = os.path.join(HERE, "pilot21_results.json")
NPZ = os.path.join(HERE, "pilot21_momenta.npz")
DIV_THR = 5.0
ELL0, CUSH = 1e-2, 0.02        # design window [1e-2, 1]; K=6 converges it (E_K ~ 3e-14)
WINDOW = 5e-2                  # certified window: sigma >= WINDOW * input scale
P_BITS = {"fp32": 24, "bf16": 8, "e4m3": 5, "e5m2": 4}
R = {}


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


# ---------------------------------------------------------------- arithmetic
def quant(x, fmt):
    """Store x in `fmt`.  fp8 uses a per-tensor power-of-two scale (delayed scaling)."""
    x = np.asarray(x, float)
    if fmt == "fp32":
        return x
    if fmt == "bf16":
        return q(x, 8)
    f = FMT[fmt]
    ax = float(np.max(np.abs(x))) if x.size else 0.0
    if not np.isfinite(ax) or ax <= 0.0:
        return x
    s = 2.0 ** np.floor(np.log2(0.8 * f["maxv"] / ax))
    return qfmt(x * s, fmt) / s


def norm_est(M):
    """Input scale: the Frobenius norm, as in the reference Muon kernel.

    It is an upper bound on sigma_max, so the iterate never enters the polynomial
    outside its certified window; a power estimate is not used here because an
    underestimate puts x_top > 1 and the quartic tail diverges (measured)."""
    return float(np.linalg.norm(M))


def polar(M, co, cast_fmt, step_fmts, nev=None):
    """K-step polar iteration with a per-step format list.  Returns (X, bits)."""
    nev = norm_est(M) if nev is None else nev
    if not np.isfinite(nev) or nev <= 0:
        return np.full_like(M, np.nan), 0
    X = quant(M / nev, cast_fmt)
    bits = P_BITS[cast_fmt]
    for (a, b, c), fmt in zip(co, step_fmts):
        A = X @ X.T
        B = quant(b * A + c * (A @ A), fmt)
        X = quant(a * X + B @ X, fmt)
        bits += P_BITS[fmt]
    return X, bits


def polar_psap(M, co, p_low, p_high, eta=1.0, eps=1e-3, nev=None, widen=True):
    """Adaptive polar from the two free sensors.  No SVD, no extra matvec.

    pi_t = ||X_t - X_{t-1}||_F / ||X_t||_F   (progress the step actually made)
    r_t  = ||X_t^q - X_t^exact||_F / ||X_t||  (rounding noise the format injected)
    While r_t < eta*pi_t the step is truncation-limited and the cheap format is
    enough; the first time r_t >= eta*pi_t a further cheap step cannot pay, so the
    tail widens (and stays wide).  pi_t <= eps ends the call.
    """
    nev = norm_est(M) if nev is None else nev
    if not np.isfinite(nev) or nev <= 0:
        return np.full_like(M, np.nan), 0, []
    Xin = M / nev
    X = quant(Xin, p_low)
    bits = P_BITS[p_low]
    fmts = ["cast:" + p_low]
    wide = False
    for (a, b, c) in co:
        fmt = p_high if wide else p_low
        A = X @ X.T
        B = quant(b * A + c * (A @ A), fmt)
        Xe = a * X + B @ X
        Xq = quant(Xe, fmt)
        nx = max(float(np.linalg.norm(Xq)), 1e-300)
        p = float(np.linalg.norm(Xq - X)) / nx
        r = float(np.linalg.norm(Xq - Xe)) / nx
        if not np.isfinite(p) or not np.isfinite(r):
            return np.full_like(M, np.nan), bits, fmts
        X = Xq
        bits += P_BITS[fmt]
        fmts.append(fmt)
        if widen and not wide and r >= eta * p:
            wide = True
        if p <= eps:
            break
    return X, bits, fmts


def exact_polar(M):
    """(polar factor, absolute singular values, right singular vectors)."""
    w, s, vt = np.linalg.svd(M, full_matrices=False)
    return w @ vt, s, vt


def spec_err(X, ref):
    try:
        v = float(np.linalg.norm(X - ref, 2))
    except Exception:
        return float("inf")
    return v if np.isfinite(v) else float("inf")


def window_err(X, ref, vt, sv, floor):
    """Error on the certified window: modes the design actually accelerates
    (sigma >= cushion * input scale).  Outside it the iterate lags by design, and
    that lag is arithmetic-independent, so it is excluded from the precision metric
    and reported separately as the full spectral error."""
    r = int(np.sum(sv >= floor))
    r = max(1, min(r, vt.shape[0]))
    Vr = vt[:r, :].T                    # right singular vectors are the rows of V^T
    try:
        v = float(np.linalg.norm((X - ref) @ Vr, 2))
    except Exception:
        return float("inf")
    return v if np.isfinite(v) else float("inf")


# ---------------------------------------------------------------- presets
def coefs(K, safety):
    co, _ = pe_coeffs(ELL0, K, CUSH, safety)
    return co


def preset(name, K=6):
    """(coefficients, cast format, per-step formats, safety)."""
    if name == "fp32":
        return coefs(K, 1.01), "fp32", ["fp32"] * K, 1.01
    if name == "bf16":
        return coefs(K, 1.01), "bf16", ["bf16"] * K, 1.01
    if name == "e4m3":
        return coefs(K, 1.01), "e4m3", ["e4m3"] * K, 1.01
    if name == "e5m2":
        return coefs(K, 1.01), "e5m2", ["e5m2"] * K, 1.01
    if name == "e5m2_s1035":
        return coefs(K, 1.035), "e5m2", ["e5m2"] * K, 1.035
    if name == "ours":                                  # bf16 cast + 2 steps, E5M2 body
        return coefs(K, 1.035), "bf16", ["bf16", "bf16"] + ["e5m2"] * (K - 2), 1.035
    if name == "ours4":                                 # same rule, E4M3 body
        return coefs(K, 1.035), "bf16", ["bf16", "bf16"] + ["e4m3"] * (K - 2), 1.035
    raise ValueError(name)


# ---------------------------------------------------------------- data & model
def get_data(task, seed):
    if task in ("digits", "resmlp"):
        d = load_digits()
        Xa = d.data / 16.0
        Xa = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-8)
        ya = d.target
        rs = np.random.RandomState(seed)
        p = rs.permutation(len(ya))
        return Xa[p[:1400]], ya[p[:1400]], Xa[p[1400:]], ya[p[1400:]]
    if task == "diabetes":
        d = load_diabetes()
        Xa = d.data.copy()
        Xa = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-8)
        ya = d.target.astype(float)
        ya = (ya - ya.mean()) / ya.std()
        rs = np.random.RandomState(seed)
        p = rs.permutation(len(ya))
        return Xa[p[:350]], ya[p[:350]], Xa[p[350:]], ya[p[350:]]
    raise ValueError(task)


def layers_of(task, rs):
    if task == "digits":
        dims = [(128, 64), (128, 128), (128, 128), (128, 128)]
    elif task == "resmlp":                       # input map + 2 pre-norm residual blocks
        dims = [(128, 64), (128, 128), (128, 128), (128, 128), (128, 128)]
    elif task == "diabetes":
        dims = [(64, 10), (64, 64), (64, 64)]
    else:
        raise ValueError(task)
    nout = 1 if task == "diabetes" else 10
    Ws = [rs.randn(a, b) / np.sqrt(b) for a, b in dims]
    out_dim = dims[-1][0]
    W5 = rs.randn(nout, out_dim) / np.sqrt(out_dim)
    return Ws, W5, dims


def ln(a):
    mu = a.mean(1, keepdims=True)
    sd = a.std(1, keepdims=True) + 1e-5
    return (a - mu) / sd, mu, sd


def forward(task, Ws, W5, b5, X, cache=False):
    hs = [X]
    zs = []
    h = X
    if task in ("digits", "diabetes"):
        for W in Ws:
            zp = h @ W.T
            z, mu, sd = ln(zp)
            h = np.maximum(z, 0)
            hs.append(h)
            if cache:
                zs.append((zp, mu, sd))
    else:                                        # resmlp: h + relu(LN(W2 relu(LN(W1 h))))
        zp0 = h @ Ws[0].T
        z0, mu0, sd0 = ln(zp0)
        h = np.maximum(z0, 0)
        hs.append(h)
        if cache:
            zs.append((zp0, mu0, sd0))
        for i in (1, 3):
            r = h
            zp1 = h @ Ws[i].T
            z1, mu1, sd1 = ln(zp1)
            a1 = np.maximum(z1, 0)
            zp2 = a1 @ Ws[i + 1].T
            z2, mu2, sd2 = ln(zp2)
            h = r + np.maximum(z2, 0)
            hs.extend([a1, h])
            if cache:
                zs.extend([(zp1, mu1, sd1), (zp2, mu2, sd2)])
    return h @ W5.T + b5, hs, zs


def ln_backward(dy, zp, mu, sd):
    """d/dz of zhat = (z-mu)/sd for per-sample normalisation."""
    zhat = (zp - mu) / sd
    return (dy - dy.mean(1, keepdims=True) - zhat * (dy * zhat).mean(1, keepdims=True)) / sd


def loss_and_grads(task, Ws, W5, b5, X, y):
    lg, hs, zs = forward(task, Ws, W5, b5, X, cache=True)
    if task == "diabetes":
        r = lg[:, 0] - y
        loss = float(np.mean(r ** 2))
        dlg = (2.0 * r / len(y))[:, None]
    else:
        z = lg - lg.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        loss = float(-np.log(p[np.arange(len(y)), y] + 1e-30).mean())
        p[np.arange(len(y)), y] -= 1.0
        dlg = p / len(y)
    dW5 = dlg.T @ hs[-1]
    db5 = dlg.sum(0)
    dh = dlg @ W5
    gs = [None] * len(Ws)

    def back_ln(dy, zp_, mu_, sd_):
        zhat = (zp_ - mu_) / sd_
        return ln_backward(dy * (zhat > 0), zp_, mu_, sd_)

    if task in ("digits", "diabetes"):
        for i in range(len(Ws) - 1, -1, -1):
            zp, mu, sd = zs[i]
            dz = back_ln(dh, zp, mu, sd)
            gs[i] = dz.T @ hs[i]
            dh = dz @ Ws[i]
    else:
        for base in (3, 1):
            zp2, mu2, sd2 = zs[base + 1]
            dz2 = back_ln(dh, zp2, mu2, sd2)
            gs[base + 1] = dz2.T @ hs[base + 1]
            da = dz2 @ Ws[base + 1]
            zp1, mu1, sd1 = zs[base]
            dz1 = back_ln(da, zp1, mu1, sd1)
            gs[base] = dz1.T @ hs[base]
            dh = dz1 @ Ws[base] + dh        # residual path
        zp0, mu0, sd0 = zs[0]
        dz0 = back_ln(dh, zp0, mu0, sd0)
        gs[0] = dz0.T @ hs[0]
    return loss, gs, dW5, db5


def ce(logits, y):
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    return float(-np.log(p[np.arange(len(y)), y] + 1e-30).mean())


# ---------------------------------------------------------------- training loop
def train(task, cond, seed, steps=400, K=6, lr=0.01, batch=256, adam_lr=3e-3,
          psap=None, probe_every=25, traj_every=50, eps=None):
    Xtr, ytr, Xva, yva = get_data(task, seed)
    rs = np.random.RandomState(1000 + seed)
    Ws, W5, dims = layers_of(task, rs)
    b5 = np.zeros(W5.shape[0])
    ms = [np.zeros_like(W) for W in Ws]
    aW5 = np.zeros_like(W5); vW5 = np.zeros_like(W5)
    ab5 = np.zeros_like(b5); vb5 = np.zeros_like(b5)
    if psap is None:
        co, cast_fmt, step_fmts, safety = preset(cond, K)
    else:
        co = coefs(K, 1.035)
        cast_fmt, step_fmts, safety = psap["low"], None, 1.035
    traj = []
    errs, errs_full, bits_used = [], [], []
    div = False
    reason = "completed"
    t0 = time.time()
    for t in range(steps):
        idx = rs.choice(len(ytr), batch, replace=False)
        xb, yb = Xtr[idx], ytr[idx]
        loss, gs, dW5, db5 = loss_and_grads(task, Ws, W5, b5, xb, yb)
        for l, (W, g) in enumerate(zip(Ws, gs)):
            ms[l] = 0.95 * ms[l] + g
            upd = g + 0.95 * ms[l]
            if psap is None:
                O, bts = polar(upd, co, cast_fmt, step_fmts)
            else:
                O, bts, _ = polar_psap(upd, co, psap["low"], psap["high"],
                                       psap.get("kappa", 4.0),
                                       psap.get("eps", 1e-3), None)
            bits_used.append(bts)
            if t % probe_every == 0:
                ref, sv, vt = exact_polar(upd)
                errs.append(window_err(O, ref, vt, sv, WINDOW * norm_est(upd)))
                errs_full.append(spec_err(O, ref))
            W -= lr * 0.2 * np.sqrt(max(W.shape)) * O
            if not np.all(np.isfinite(W)) or np.abs(W).max() > 1e3:
                div = True
                reason = "weight_blowup"
                break
        aW5 = 0.9 * aW5 + 0.1 * dW5; vW5 = 0.999 * vW5 + 0.001 * dW5 ** 2
        W5 -= adam_lr * (aW5 / (1 - 0.9 ** (t + 1))) / (np.sqrt(vW5 / (1 - 0.999 ** (t + 1))) + 1e-8)
        ab5 = 0.9 * ab5 + 0.1 * db5; vb5 = 0.999 * vb5 + 0.001 * db5 ** 2
        b5 -= adam_lr * (ab5 / (1 - 0.9 ** (t + 1))) / (np.sqrt(vb5 / (1 - 0.999 ** (t + 1))) + 1e-8)
        if div:
            break
        if (t + 1) % traj_every == 0 or t == steps - 1:
            ltr, _, _ = forward(task, Ws, W5, b5, Xtr)
            lva, _, _ = forward(task, Ws, W5, b5, Xva)
            if task == "diabetes":
                tl = float(np.mean((ltr[:, 0] - ytr) ** 2))
                vl = float(np.mean((lva[:, 0] - yva) ** 2))
                acc = None
            else:
                tl = ce(ltr, ytr); vl = ce(lva, yva)
                acc = float((lva.argmax(1) == yva).mean())
            traj.append({"t": t + 1, "train": tl, "val": vl, "acc": acc})
        if not np.isfinite(loss):
            div = True
            reason = "nonfinite_loss"
            break
        if not np.isfinite(np.linalg.norm(O)):
            div = True
            reason = "nonfinite_update"
            break
    if not traj:
        traj = [{"t": 0, "train": None, "val": None, "acc": None}]
    return {
        "task": task, "cond": cond if psap is None else "psap",
        "seed": seed, "K": K, "lr": lr, "safety": safety, "finished": not div,
        "stop_reason": reason,
        "steps_run": len(bits_used),
        "final_train": traj[-1]["train"], "final_val": traj[-1]["val"],
        "best_val": min([z["val"] for z in traj if z["val"] is not None], default=None),
        "acc": traj[-1]["acc"],
        "upd_err_med": float(np.median(errs)) if errs else None,
        "upd_err_p90": float(np.percentile(errs, 90)) if errs else None,
        "upd_err_full_med": float(np.median(errs_full)) if errs_full else None,
        "bits_mean": float(np.mean(bits_used)) if bits_used else None,
        "traj": traj, "seconds": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------- [B] frozen replay
def collect_momenta(seed=0, steps=400, every=20):
    task = "digits"
    Xtr, ytr, _, _ = get_data(task, seed)
    rs = np.random.RandomState(1000 + seed)
    Ws, W5, dims = layers_of(task, rs)
    b5 = np.zeros(W5.shape[0])
    ms = [np.zeros_like(W) for W in Ws]
    aW5 = np.zeros_like(W5); vW5 = np.zeros_like(W5)
    ab5 = np.zeros_like(b5); vb5 = np.zeros_like(b5)
    co, cast_fmt, step_fmts, _ = preset("fp32", 6)
    store = {f"W{l}": [] for l in range(len(Ws))}
    meta = []
    for t in range(steps):
        idx = rs.choice(len(ytr), 256, replace=False)
        _, gs, dW5, db5 = loss_and_grads(task, Ws, W5, b5, Xtr[idx], ytr[idx])
        for l, (W, g) in enumerate(zip(Ws, gs)):
            ms[l] = 0.95 * ms[l] + g
            upd = g + 0.95 * ms[l]
            if t % every == 0:
                store[f"W{l}"].append(upd.astype(np.float32))
                meta.append({"layer": l, "step": t, "shape": list(upd.shape),
                             "idx": len(store[f"W{l}"]) - 1})
            O, _ = polar(upd, co, cast_fmt, step_fmts)
            W -= 0.01 * 0.2 * np.sqrt(max(W.shape)) * O
        aW5 = 0.9 * aW5 + 0.1 * dW5; vW5 = 0.999 * vW5 + 0.001 * dW5 ** 2
        W5 -= 3e-3 * (aW5 / (1 - 0.9 ** (t + 1))) / (np.sqrt(vW5 / (1 - 0.999 ** (t + 1))) + 1e-8)
        ab5 = 0.9 * ab5 + 0.1 * db5; vb5 = 0.999 * vb5 + 0.001 * db5 ** 2
        b5 -= 3e-3 * (ab5 / (1 - 0.9 ** (t + 1))) / (np.sqrt(vb5 / (1 - 0.999 ** (t + 1))) + 1e-8)
    np.savez_compressed(NPZ, **{k: np.array(v, dtype=np.float32) for k, v in store.items()})
    return meta, store


def sensors(M):
    """Spectral readout with no SVD: 2 power iterations + 4-probe Hutchinson trace."""
    A = M @ M.T
    rng = np.random.RandomState(7)
    v = rng.randn(M.shape[1])
    v /= np.linalg.norm(v)
    for _ in range(2):
        v = M.T @ (M @ v)
        v /= np.linalg.norm(v)
    lam_max = float(np.linalg.norm(M @ v)) ** 2
    rg = np.random.RandomState(12345)
    tr = 0.0
    for _ in range(4):
        p = np.where(rg.rand(M.shape[0]) > 0.5, 1.0, -1.0)
        tr += float(p @ A @ p) / 4.0
    return {"lam_max": lam_max, "trace": tr, "spread": tr / max(lam_max, 1e-300)}


def stage_b2():
    z = np.load(NPZ)
    meta = R["B1"]["meta"]
    designs = ["fp32", "bf16", "e4m3", "e5m2", "e5m2_s1035", "ours", "ours4"]
    out = {k: {} for k in designs}
    for nm in ("psapA", "psapB", "psapC"):
        out[nm] = {}
    out["sens"] = {}
    co_psap = coefs(6, 1.035)
    for mm in meta:
        l, t, i = mm["layer"], mm["step"], mm["idx"]
        M = np.array(z[f"W{l}"][i], dtype=float)
        nev = norm_est(M)
        ref, sv, vt = exact_polar(M)
        ell = float(sv[-1] / sv[0])
        s = sensors(M)
        base = {"layer": l, "step": t, "ell": ell, "spread": s["spread"],
                "lam_max": s["lam_max"], "trace": s["trace"],
                "n_win": int(max(1, np.sum(sv >= WINDOW * nev)))}
        errs_by, bits_by = {}, {}
        for nm in designs:
            co, cf, sf, _ = preset(nm, 6)
            X, bits = polar(M, co, cf, sf, nev)
            e = window_err(X, ref, vt, sv, WINDOW * nev)
            ef = spec_err(X, ref)
            errs_by[nm], bits_by[nm] = e, bits
            out[nm][f"{l}@{t}"] = {**base, "err": e, "err_full": ef, "bits": bits,
                                   "div": e > DIV_THR}
        for nm, (lo, hi, eps) in (("psapA", ("e5m2", "bf16", 5e-3)),
                                  ("psapB", ("bf16", "bf16", 5e-3)),
                                  ("psapC", ("e4m3", "bf16", 5e-3))):
            X, bits, fmts = polar_psap(M, co_psap, lo, hi, 1.0, eps, nev)
            e = window_err(X, ref, vt, sv, WINDOW * nev)
            out[nm][f"{l}@{t}"] = {**base, "err": e, "err_full": spec_err(X, ref),
                                   "bits": bits, "div": e > DIV_THR,
                                   "fmts": fmts, "steps": len(fmts) - 1, "eps": eps}
        # what does widening from E5M2 to bf16 buy on this matrix?
        out["sens"][f"{l}@{t}"] = {**base, "err_e5m2": errs_by["e5m2"],
                                   "err_e4m3": errs_by["e4m3"], "err_bf16": errs_by["bf16"],
                                   "err_fp32": errs_by["fp32"], "err_ours": errs_by["ours"],
                                   "benefit": errs_by["e5m2"] - errs_by["bf16"],
                                   "bits_gap": bits_by["bf16"] - bits_by["e5m2"]}
    R["B2"] = out
    agg = {}
    fp32_med = float(np.median([z["err"] for z in out["fp32"].values()]))
    for nm, cells in out.items():
        if nm in ("sens",):
            continue
        v = list(cells.values())
        e = np.array([z["err"] for z in v])
        b = np.array([z["bits"] for z in v])
        agg[nm] = {"n": len(v), "err_med": float(np.median(e)),
                   "err_mean": float(np.mean(e)), "err_p90": float(np.percentile(e, 90)),
                   "err_full_med": float(np.median([z["err_full"] for z in v])),
                   "bits_mean": float(np.mean(b)),
                   "n_div": int(sum(z["div"] for z in v)),
                   "ratio_to_fp32": float(np.median(e) / max(fp32_med, 1e-300))}
    R["B2_agg"] = agg
    rows = list(out["sens"].values())
    ben = np.array([r["benefit"] for r in rows])
    cl = np.maximum(np.array([r["err_e5m2"] for r in rows]), 1e-300)
    rel = ben / cl                      # fraction of the E5M2 error the widening removes

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) > 0 and np.std(rb) > 0 else 0.0

    sc = {}
    for n in ("layer", "step", "ell", "spread", "lam_max", "trace", "n_win"):
        v = np.array([r[n] for r in rows], float)
        sc[n] = {"spearman_err_e5m2": spearman(v, np.array([r["err_e5m2"] for r in rows])),
                 "spearman_rel_benefit": spearman(v, rel)}
    sc["top_k_precision"] = {}
    for n in ("spread", "ell", "n_win"):
        v = np.array([r[n] for r in rows], float)
        for frac in (0.25, 0.5):
            k = max(1, int(frac * len(v)))
            idx = np.argsort(-v)[:k]
            sc["top_k_precision"][f"{n}_{frac}"] = float(ben[idx].sum() / max(ben.sum(), 1e-300))
    sc["rel_benefit_med"] = float(np.median(rel))
    sc["n"] = len(rows)
    R["B2_sensors"] = sc
    return agg, sc


def stage_d():
    """Step-count sweep, precision-ladder sweep, and the per-step cost of a cheap step."""
    z = np.load(NPZ)
    meta = R["B1"]["meta"]
    mats = []
    for mm in meta:
        M = np.array(z[f"W{mm['layer']}"][mm["idx"]], float)
        nev = norm_est(M)
        ref, sv, vt = exact_polar(M)
        mats.append((M, nev, ref, sv, vt, WINDOW * nev))
    out = {"ksweep": {}, "ladder": {}, "single_cheap_step": {}}
    for fmt in ("bf16", "e4m3", "e5m2"):
        for K in range(1, 7):
            co = coefs(K, 1.035)
            errs, bits = [], 0
            for M, nev, ref, sv, vt, fl in mats:
                X, bits = polar(M, co, fmt, [fmt] * K, nev)
                errs.append(window_err(X, ref, vt, sv, fl))
            out["ksweep"][f"{fmt}@K{K}"] = {"err_med": float(np.median(errs)),
                                            "err_p90": float(np.percentile(errs, 90)),
                                            "bits": bits, "K": K}
    K = 6
    co = coefs(K, 1.035)
    for lo in ("e4m3", "e5m2"):
        for m in range(K + 1):
            fmts = [lo] * m + ["bf16"] * (K - m)
            errs, bits = [], 0
            for M, nev, ref, sv, vt, fl in mats:
                X, bits = polar(M, co, "bf16", fmts, nev)
                errs.append(window_err(X, ref, vt, sv, fl))
            out["ladder"][f"{lo}^{m}+bf16^{K-m}"] = {"err_med": float(np.median(errs)),
                                                     "bits": bits, "m": m}
    base = []
    for M, nev, ref, sv, vt, fl in mats:
        X, _ = polar(M, co, "bf16", ["bf16"] * K, nev)
        base.append(window_err(X, ref, vt, sv, fl))
    b = float(np.median(base))
    for j in range(K):
        errs = []
        for M, nev, ref, sv, vt, fl in mats:
            fmts = ["bf16"] * K
            fmts[j] = "e5m2"
            X, _ = polar(M, co, "bf16", fmts, nev)
            errs.append(window_err(X, ref, vt, sv, fl))
        out["single_cheap_step"][f"step{j+1}"] = {"err_med": float(np.median(errs)),
                                                  "ratio_to_all_bf16": float(np.median(errs) / b)}
    out["all_bf16_err_med"] = b
    R["D"] = out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--task", default="digits")
    ap.add_argument("--cond", default="fp32")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()
    load()
    t0 = time.time()
    if args.stage == "a":
        seeds = [int(s) for s in args.seeds.split(",")]
        R.setdefault("A", {}).setdefault(args.task, {})
        for cond in args.cond.split(","):
            R["A"][args.task].setdefault(cond, {})
            for sd in seeds:
                r = train(args.task, cond, sd, steps=args.steps)
                R["A"][args.task][cond][str(sd)] = r
                print(f"{args.task:9s} {cond:11s} s{sd} fin={r['finished']} "
                      f"train={r['final_train']} val={r['final_val']} acc={r['acc']} "
                      f"win_err={r['upd_err_med']} full={r['upd_err_full_med']} "
                      f"bits={r['bits_mean']:.1f} {r['seconds']}s", flush=True)
                save()
    elif args.stage == "c":
        # loss x (format, K) grid: where does the coupled loss stop caring?
        seeds = [int(s) for s in args.seeds.split(",")]
        fmts = args.cond.split(",")
        Ks = [2, 3, 4, 5, 6]
        R.setdefault("C", {}).setdefault(args.task, {})
        for fmt in fmts:
            for K in Ks:
                key = f"{fmt}@K{K}"
                R["C"][args.task].setdefault(key, {})
                for sd in seeds:
                    r = train(args.task, fmt, sd, steps=args.steps, K=K)
                    R["C"][args.task][key][str(sd)] = r
                    print(f"{args.task:9s} {key:12s} s{sd} fin={r['finished']} "
                          f"train={r['final_train']} val={r['final_val']} acc={r['acc']} "
                          f"win_err={r['upd_err_med']} bits={r['bits_mean']:.1f} {r['seconds']}s",
                          flush=True)
                    save()
    elif args.stage == "d":
        out = stage_d()
        print(json.dumps(out, indent=1))
        save()
    elif args.stage == "b1":
        meta, store = collect_momenta(seed=0, steps=400, every=20)
        R["B1"] = {"meta": meta, "n": len(meta), "npz": os.path.basename(NPZ),
                   "source": "digits MLP5, fp32 Muon, seed 0, every 20 steps"}
        print("collected", len(meta), "matrices in", round(time.time() - t0, 1), "s", flush=True)
        save()
    elif args.stage == "b2":
        agg, sc = stage_b2()
        print(json.dumps(agg, indent=1))
        print(json.dumps(sc, indent=1))
        save()
    elif args.stage == "b3":
        seeds = [int(s) for s in args.seeds.split(",")]
        R.setdefault("B3", {}).setdefault(args.task, {})
        for cond in args.cond.split(","):
            R["B3"][args.task].setdefault(cond, {})
            for sd in seeds:
                if cond.startswith("psap"):
                    cfg = {"psapA": ("e5m2", "bf16", 1.0),
                           "psapB": ("bf16", "bf16", 1.0),
                           "psapC": ("e4m3", "bf16", 1.0)}[cond]
                    r = train(args.task, cond, sd, steps=args.steps,
                              psap={"kind": "A", "low": cfg[0], "high": cfg[1],
                                    "eta": cfg[2], "eps": 5e-3})
                else:
                    r = train(args.task, cond, sd, steps=args.steps)
                R["B3"][args.task][cond][str(sd)] = r
                print(f"{args.task:9s} {cond:12s} s{sd} fin={r['finished']} "
                      f"train={r['final_train']} val={r['final_val']} acc={r['acc']} "
                      f"win_err={r['upd_err_med']} bits={r['bits_mean']:.1f} {r['seconds']}s",
                      flush=True)
                save()
    print("stage", args.stage, "done in", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
