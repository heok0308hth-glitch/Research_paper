#!/usr/bin/env python3
"""
Pilot 3b - Stability threshold of momentum SGD under a b-bit quantised momentum
buffer, and the resulting bit/learning-rate exchange rate.

Pilot 3 (see pilot3_bits_floor.py) showed two things:
  (1) a scale-invariant (relative / block) quantiser produces no loss floor at
      all -- the iterate converges to ~1e-24 for every b in {2..12};
  (2) at high momentum the low-bit run *diverges* at learning rates that are
      stable at high precision.
This script measures that boundary: eta_max(b, beta) = largest eta whose final
loss stays below 1e-2 after 3000 steps.
"""
import json
import numpy as np

np.random.seed(0)
R = {}


def quant(x, b):
    s = np.max(np.abs(x))
    if s == 0 or not np.isfinite(s):
        return np.zeros_like(x)
    step = 2 * s / (2.0 ** b)
    return np.round(x / step) * step


def final_loss(eta, b, beta, T=3000, d=8, seeds=4):
    lam = np.logspace(0, 2, d)
    vals = []
    for sd in range(seeds):
        rng = np.random.default_rng(1000 + sd)
        th = rng.standard_normal(d)
        m = np.zeros(d)
        ok = True
        tail = []
        for t in range(T):
            m = beta * m + lam * th
            th = th - eta * quant(m, b)
            if not np.all(np.isfinite(th)) or np.max(np.abs(th)) > 1e6:
                ok = False
                break
            if t >= T - 400:
                tail.append(0.5 * np.sum(lam * th ** 2))
        if not ok:
            vals.append(np.inf)
        else:
            vals.append(float(np.mean(tail)))
    return float(np.mean(vals))


def eta_max(b, beta, lo=1e-4, hi=3.0, tol=1.02, thresh=1e-2):
    """Largest eta with final loss < thresh (log bisection)."""
    if final_loss(hi, b, beta) < thresh:
        return hi
    while hi / lo > tol:
        mid = np.sqrt(lo * hi)
        if final_loss(mid, b, beta) < thresh:
            lo = mid
        else:
            hi = mid
    return lo


def main():
    print("=" * 84)
    print("[A] largest stable learning rate eta_max(b, beta)   (final loss < 1e-2)")
    print("=" * 84)
    bs = [2, 3, 4, 5, 6, 8, 12]
    betas = [0.0, 0.5, 0.9, 0.99]
    print(f"{'b':>3s} " + "".join(f"beta={be:<9.2f}" for be in betas))
    for b in bs:
        row = []
        for be in betas:
            e = eta_max(b, be)
            row.append(e)
        R.setdefault("eta_max", {})[str(b)] = dict(zip(map(str, betas), row))
        print(f"{b:3d} " + "".join(f"{v:<13.3e}" for v in row))

    print("\n" + "=" * 84)
    print("[B] cost of losing one bit, in learning-rate terms: eta_max(b+1)/eta_max(b)")
    print("=" * 84)
    for be in betas:
        row = [R["eta_max"][str(b)][str(be)] for b in bs]
        rat = [row[i + 1] / row[i] for i in range(len(row) - 1)]
        R.setdefault("ratio", {})[str(be)] = rat
        print(f"beta={be:<5.2f} " + "".join(f"{r:6.2f} " for r in rat)
              + f"  (b={bs})")

    print("\n" + "=" * 84)
    print("[C] does the loss floor exist for a fixed-scale quantiser? (eta sweep)")
    print("=" * 84)
    def run_abs(eta, b, beta, T=3000, d=8, seeds=4):
        lam = np.logspace(0, 2, d)
        vals = []
        for sd in range(seeds):
            rng = np.random.default_rng(1000 + sd)
            th = rng.standard_normal(d)
            m = np.zeros(d)
            scale0 = None
            tail = []
            for t in range(T):
                m = beta * m + lam * th
                if scale0 is None:
                    scale0 = 2 * np.max(np.abs(m))
                step = 2 * scale0 / (2.0 ** b)
                mq = np.round(m / step) * step
                th = th - eta * mq
                if not np.all(np.isfinite(th)) or np.max(np.abs(th)) > 1e6:
                    vals.append(np.inf)
                    break
                if t >= T - 400:
                    tail.append(0.5 * np.sum(lam * th ** 2))
            else:
                vals.append(float(np.mean(tail)))
        return float(np.mean(vals))

    etas = [1e-4, 1e-3, 1e-2, 3e-2]
    print(f"{'b':>3s} " + "".join(f"eta={e:<9.0e}" for e in etas))
    for b in (2, 4, 6, 8, 12):
        row = [run_abs(e, b, 0.0) for e in etas]
        R.setdefault("abs_floor", {})[str(b)] = row
        print(f"{b:3d} " + "".join(f"{v:<13.2e}" for v in row))

    with open("work/iclr-research/data/pilot3b_results.json", "w") as f:
        json.dump(R, f, indent=1)


if __name__ == "__main__":
    main()
