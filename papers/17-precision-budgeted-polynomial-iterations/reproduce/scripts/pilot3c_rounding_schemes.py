#!/usr/bin/env python3
"""
Pilot 3c - Does re-randomising the quantisation grid (stochastic rounding, and
stochastic rounding + error feedback) recover the accuracy lost in a
high-momentum low-bit momentum buffer?

Same setting as pilot 3: f = 0.5*sum lam_i th_i^2, m = beta*m + g,
th = th - eta*Q(m) with a relative block quantiser of b bits.
Rounds: det = round-to-nearest, sr = stochastic rounding (unbiased),
        sr_ef = stochastic rounding with error feedback (residual carried).
"""
import json
import numpy as np

np.random.seed(0)
R = {}


def quant_det(x, b):
    s = np.max(np.abs(x))
    if s == 0:
        return np.zeros_like(x)
    step = 2 * s / 2.0 ** b
    return np.round(x / step) * step


def quant_sr(x, b, rng):
    s = np.max(np.abs(x))
    if s == 0:
        return np.zeros_like(x)
    step = 2 * s / 2.0 ** b
    f = x / step
    fl = np.floor(f)
    return (fl + (rng.random(x.shape) < (f - fl))) * step


def run(eta, b, beta, mode="det", T=3000, d=8, seeds=6):
    lam = np.logspace(0, 2, d)
    outs = []
    for sd in range(seeds):
        rng = np.random.default_rng(7000 + sd)
        th = rng.standard_normal(d)
        m = np.zeros(d)
        res = np.zeros(d)
        tail = []
        for t in range(T):
            m = beta * m + lam * th
            if mode == "det":
                mq = quant_det(m, b)
            elif mode == "sr":
                mq = quant_sr(m, b, rng)
            else:
                v = m + res
                mq = quant_sr(v, b, rng)
                res = v - mq
            th = th - eta * mq
            if not np.all(np.isfinite(th)) or np.max(np.abs(th)) > 1e6:
                outs.append(np.inf)
                break
            if t >= T - 500:
                tail.append(0.5 * np.sum(lam * th ** 2))
        else:
            outs.append(float(np.mean(tail)))
    return float(np.mean(outs))


def main():
    print("=" * 88)
    print("[A] final loss: deterministic vs stochastic rounding vs SR+error feedback")
    print("=" * 88)
    for beta in (0.0, 0.9, 0.99):
        print(f"\n-- beta={beta}, eta=1e-2")
        print(f"{'b':>3s} {'det':>12s} {'stochastic':>12s} {'SR+EF':>12s}")
        for b in (2, 3, 4, 6):
            v = [run(1e-2, b, beta, m) for m in ("det", "sr", "sr_ef")]
            R.setdefault("cmp", {}).setdefault(str(beta), {})[str(b)] = v
            print(f"{b:3d} {v[0]:12.2e} {v[1]:12.2e} {v[2]:12.2e}")

    print("\n" + "=" * 88)
    print("[B] high-momentum, low-bit corner: how many orders does each scheme recover?")
    print("=" * 88)
    for (beta, b) in ((0.99, 2), (0.99, 3), (0.999, 2)):
        v = [run(1e-2, b, beta, m) for m in ("det", "sr", "sr_ef")]
        print(f"beta={beta}, b={b}: det={v[0]:.2e}  sr={v[1]:.2e}  sr_ef={v[2]:.2e}"
              f"   (log10 gain sr_ef/det = "
              f"{(np.log10(v[0]) - np.log10(v[2])) if np.isfinite(v[0]) and np.isfinite(v[2]) else float('nan'):.1f})")
        R.setdefault("corner", {})[f"{beta}_{b}"] = v

    with open("work/iclr-research/data/pilot3c_results.json", "w") as f:
        json.dump(R, f, indent=1)


if __name__ == "__main__":
    main()
