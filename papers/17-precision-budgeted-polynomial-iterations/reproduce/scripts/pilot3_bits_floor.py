#!/usr/bin/env python3
"""
Pilot 3 - Loss floor of momentum SGD with a b-bit quantized momentum buffer.

Setting
-------
f(theta) = 0.5 * sum_i lambda_i theta_i^2 ,  lambda_i in logspace(0,2) (d=8)
m_{t} = beta*m_{t-1} + g_t ,  theta_{t} = theta_{t-1} - eta*Q(m_t)
Q = uniform relative (block) quantiser: scale = max|m|, step = 2*scale/2^b,
    Q(x) = round(x/step)*step.      (b bits => 2^b levels on [-scale, scale])

Measured: stationary loss (mean over the last 500 of 3000 steps, 5 restarts).
Question: how does the floor depend on eta, b and beta?
"""
import json
import numpy as np

np.random.seed(0)
R = {}


def quant(x, b):
    s = np.max(np.abs(x))
    if s == 0:
        return np.zeros_like(x)
    step = 2 * s / (2.0 ** b)
    return np.round(x / step) * step


def run(eta, b, beta, T=3000, d=8, seeds=5, qmode="rel"):
    lam = np.logspace(0, 2, d)
    finals = []
    for sd in range(seeds):
        rng = np.random.default_rng(1000 + sd)
        th = rng.standard_normal(d)
        m = np.zeros(d)
        scale0 = None
        tail = []
        for t in range(T):
            g = lam * th
            m = beta * m + g
            if qmode == "rel":
                mq = quant(m, b)
            else:                                   # absolute (fixed) scale
                if scale0 is None:
                    scale0 = 2 * np.max(np.abs(m))
                step = 2 * scale0 / (2.0 ** b)
                mq = np.round(m / step) * step
            th = th - eta * mq
            if t >= T - 500:
                tail.append(0.5 * np.sum(lam * th ** 2))
        finals.append(np.mean(tail))
    return float(np.mean(finals)), float(np.std(finals))


def main():
    print("=" * 96)
    print("[A] relative quantiser: floor vs learning rate (does any eta reduce it?)")
    print("=" * 96)
    etas = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0])
    for beta in (0.0, 0.9, 0.99):
        print(f"\n-- beta={beta}")
        print(f"{'b':>3s} " + "".join(f"eta={e:<9.0e}" for e in etas))
        for b in (2, 3, 4, 6, 8, 12):
            row = [run(e, b, beta)[0] for e in etas]
            R.setdefault("rel", {}).setdefault(str(beta), {})[str(b)] = {
                "eta": etas.tolist(), "loss": row}
            print(f"{b:3d} " + "".join(f"{v:<13.2e}" for v in row))

    print("\n" + "=" * 96)
    print("[B] does the floor follow 2^{-2b}?  (fixed eta, three betas)")
    print("=" * 96)
    print(f"{'b':>3s} {'loss(beta=0)':>14s} {'loss(0.9)':>14s} {'loss(0.99)':>14s}"
          f" {'ratio b/b-1':>12s}")
    for b in (2, 3, 4, 5, 6, 8, 10, 12):
        v = [run(1e-2, b, be)[0] for be in (0.0, 0.9, 0.99)]
        prev = run(1e-2, b - 1, 0.0)[0] if b > 2 else np.nan
        R.setdefault("floor", {})[str(b)] = v
        print(f"{b:3d} {v[0]:14.3e} {v[1]:14.3e} {v[2]:14.3e} "
              f"{(v[0]/prev if prev==prev else float('nan')):12.2f}")
    print(" (last column = loss(b)/loss(b-1) for beta=0; 4.0 means 2^{-2b})")

    print("\n" + "=" * 96)
    print("[C] absolute quantiser: same sweep (mechanism check)")
    print("=" * 96)
    for beta in (0.0, 0.9):
        print(f"\n-- beta={beta}")
        print(f"{'b':>3s} " + "".join(f"eta={e:<9.0e}" for e in etas))
        for b in (2, 3, 4, 6, 8):
            row = [run(e, b, beta, qmode="abs")[0] for e in etas]
            R.setdefault("abs", {}).setdefault(str(beta), {})[str(b)] = {
                "eta": etas.tolist(), "loss": row}
            print(f"{b:3d} " + "".join(f"{v:<13.2e}" for v in row))

    print("\n" + "=" * 96)
    print("[D] bit schedule: decay b with a decaying learning rate")
    print("=" * 96)
    lam = np.logspace(0, 2, 8)

    def run_sched(eta0, b0, decay, T=4000, bmin=2, sched=True, seeds=8):
        out = []
        for sd in range(seeds):
            rng = np.random.default_rng(2000 + sd)
            th = rng.standard_normal(8)
            m = np.zeros(8)
            for t in range(T):
                eta = eta0 * (decay ** t)
                b = (b0 + int(np.floor(np.log2(max(eta / eta0, 1e-12))))
                     if sched else b0)
                b = int(np.clip(b, bmin, b0))
                m = 0.9 * m + lam * th
                th = th - eta * quant(m, b)
            out.append(0.5 * np.sum(lam * th ** 2))
        return float(np.mean(out))

    eta0, b0 = 0.05, 8
    print(f"{'policy':30s} {'final loss':>12s} {'mean bits':>10s}")
    for name, sched, bmin in (("fixed 8-bit", False, 8),
                              ("scheduled (b0+log2 eta/eta0)", True, 2),
                              ("scheduled, floor 3 bits", True, 3)):
        # mean bits over the run, geometric eta decay
        T = 4000
        tot = sum(np.clip(b0 + int(np.floor(np.log2(max(0.05 * 0.9995 ** t / eta0, 1e-12)))), bmin, b0)
                  if sched else b0 for t in range(T))
        v = run_sched(eta0, b0, 0.9995, T=T, bmin=bmin, sched=sched)
        R.setdefault("sched", {})[name] = {"loss": v, "mean_bits": tot / T}
        print(f"{name:30s} {v:12.3e} {tot / T:10.2f}")

    with open("work/iclr-research/data/pilot3_results.json", "w") as f:
        json.dump(R, f, indent=1)


if __name__ == "__main__":
    main()
