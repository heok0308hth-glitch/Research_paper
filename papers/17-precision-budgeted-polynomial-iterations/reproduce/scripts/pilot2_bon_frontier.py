#!/usr/bin/env python3
"""
Pilot 2 - Sampling/verification budget frontier for best-of-N selection with a
noisy verifier (joint-Gaussian / copula model).

Model
-----
s_i ~ N(0,1)   true quality of candidate i (i.i.d., i = 1..N)
y_i = rho*s_i + sqrt(1-rho^2)*z_i    verifier score, z_i ~ N(0,1)
m independent verifier evaluations are averaged per candidate, which replaces
rho by rho_m = rho*sqrt(m) / sqrt(1 + (m-1)*rho^2).

Computed exactly (quadrature over the bivariate normal):
  F(N,m) = P(the selected candidate is the true best of the pool)
  Q(N,m) = E[true quality of the selected candidate]
Budget: B = N*(c_s + m*c_v).  Reported: the optimal (N,m) and the frontier.
"""
import json
import numpy as np
from math import sqrt
from scipy.special import ndtr as Phi_s
from scipy.stats import norm

np.random.seed(0)
R = {}
M = 400          # quadrature nodes per dimension


def phi(u):
    return norm.ppf(np.asarray(u, float))


def Phi(x):
    return Phi_s(np.asarray(x, float))


U = (np.arange(M) + 0.5) / M
S = phi(U)                       # nodes for s and z, equal weights 1/M
PHI_GRID_Y = np.linspace(-12, 12, 6000)


_GC = {}


def G_table(rho):
    """G(y) = P(y' < y) for an independent candidate, on PHI_GRID_Y."""
    if rho in _GC:
        return _GC[rho]
    r = np.sqrt(1 - rho ** 2)
    # G(y) = E_s'[ Phi((y - rho s')/r) ]
    # vectorised: (6000, M)
    _GC[rho] = Phi((PHI_GRID_Y[:, None] - rho * S[None, :]) / r).mean(axis=1)
    return _GC[rho]


def moments(N, rho, tau=None):
    """(F, E[s_selected]) for a pool of N with verifier correlation rho."""
    Gt = G_table(rho)
    r = np.sqrt(1 - rho ** 2)
    Y = rho * S[:, None] + r * S[None, :]          # (M,M): y(s,z)
    logG = np.log(np.clip(np.interp(Y.ravel(), PHI_GRID_Y, Gt), 1e-300, 1.0))
    logG = logG.reshape(M, M)
    w = np.exp((N - 1) * logG)                      # G^{N-1}
    base = w.mean()                                 # average over (u,v)
    F = N * base
    Q = N * (w * S[:, None]).mean()
    return float(F), float(Q)


def main():
    print("=" * 92)
    print("[A] selector fidelity F(N,rho) = P(pick the pool-best) and quality E[s]")
    print("=" * 92)
    Ns = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    for rho in (0.3, 0.5, 0.7, 0.9):
        Fs, Qs = [], []
        for N in Ns:
            f, q = moments(N, rho)
            Fs.append(f)
            Qs.append(q)
        R.setdefault("F", {})[str(rho)] = dict(zip(map(str, Ns), Fs))
        R.setdefault("Q", {})[str(rho)] = dict(zip(map(str, Ns), Qs))
        print(f"\nrho={rho}")
        print(" N     " + "".join(f"{n:<9d}" for n in Ns))
        print(" F     " + "".join(f"{v:<9.2e}" for v in Fs))
        print(" Q     " + "".join(f"{v:<9.3f}" for v in Qs))
        # local log-log slope of F at the largest N
        th = -(np.log(Fs[-1]) - np.log(Fs[-3])) / (np.log(Ns[-1]) - np.log(Ns[-3]))
        print(f" local exponent theta(N~{Ns[-1]}) = {th:.3f}   vs 1/rho^2 = {1/rho**2:.3f}"
              f"   vs (1-rho)/(1+rho) = {(1-rho)/(1+rho):.3f}")
        R.setdefault("theta", {})[str(rho)] = float(th)

    print("\n" + "=" * 92)
    print("[B] budget frontier: B = N*(c_s + m*c_v); maximise E[true quality]")
    print("=" * 92)
    ms = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    for cv in (0.01, 0.1, 0.5):
        print(f"\n-- c_v/c_s = {cv}")
        print(f"{'rho':>5s} " + "".join(f"B={b:<9d}" for b in (64, 256, 1024, 4096)))
        for rho in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            cells = []
            for B in (64, 256, 1024, 4096):
                best = (-1, None, None)
                for m in ms:
                    N = int(B / (1 + m * cv))
                    if N < 1:
                        continue
                    rm = rho * np.sqrt(m) / np.sqrt(1 + (m - 1) * rho ** 2)
                    _, q = moments(N, rm)
                    if q > best[0]:
                        best = (q, m, N)
                cells.append(best)
                R.setdefault("frontier", {}).setdefault(str(cv), {}).setdefault(
                    str(rho), {})[str(B)] = {"q": best[0], "m": best[1], "N": best[2]}
            print(f"{rho:5.2f} " + "".join(
                f"m={c[1]:<2d}N={c[2]:<5d}" for c in cells))

    print("\n" + "=" * 92)
    print("[C] does more sampling ever hurt?  E[quality] and F as N grows")
    print("    (quality is monotone; fidelity falls -- report the ratio)")
    print("=" * 92)
    for rho in (0.3, 0.5, 0.7, 0.9):
        qs = [R["Q"][str(rho)][str(n)] for n in Ns]
        print(f"rho={rho}: Q monotone increasing = "
              f"{all(qs[i] <= qs[i+1] + 1e-12 for i in range(len(qs)-1))}; "
              f"Q(1)={qs[0]:.3f} Q(4096)={qs[-1]:.3f}  "
              f"F(1)={R['F'][str(rho)]['1']:.3f} F(4096)={R['F'][str(rho)]['4096']:.2e}")

    with open("work/iclr-research/data/pilot2_results.json", "w") as f:
        json.dump(R, f, indent=1)


if __name__ == "__main__":
    main()
