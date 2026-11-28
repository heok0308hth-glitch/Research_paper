#!/usr/bin/env python3
"""
Pilot 2b - Sampling/verification budget frontier (corrected).

Exact identity used
-------------------
If (s, ybar) are jointly Gaussian with correlation rho_m, then E[s | ybar] = rho_m*ybar,
so the expected true quality of the sample that maximises ybar is
        Q(N, m) = rho_m * E[max of N i.i.d. y's]
                = rho_m * int_0^1 Phi^{-1}(t^{1/N}) dt
and the selection fidelity is the separate 2-D integral
        F(N, m) = N * int int f(s,y) H(y,s)^{N-1} ds dy,
        H(y,s) = P(y' < y, s' < s)   (bivariate normal CDF, correlation rho_m).
"""
import json
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

np.random.seed(0)
R = {}
T_GRID = (np.arange(6000) + 0.5) / 6000


def Emax(N):
    """E[max of N i.i.d. N(0,1)] exactly, via t = Phi(y)^N substitution."""
    return float(np.mean(norm.ppf(T_GRID ** (1.0 / N))))


def rho_m(rho, m):
    return rho * np.sqrt(m) / np.sqrt(1 + (m - 1) * rho ** 2)


# --------------------------------------------------------------- fidelity F
GY = np.linspace(-5.5, 5.5, 330)
GS = np.linspace(-5.5, 5.5, 330)
_T = np.linspace(-6.0, 6.0, 400)
_W = norm.pdf(_T)
_W = _W / _W.sum()


def H_table(rho):
    """H[i,j] = P(y' <= GY[i], s' <= GS[j]) for correlation rho."""
    r = np.sqrt(1 - rho ** 2)
    # Phi2(y,s) = int phi(t) Phi((s - rho t)/r) dt  over t <= y
    # build cumulative over the t grid, then interpolate at y nodes
    vals = ndtr((GS[None, :] - rho * _T[:, None]) / r)        # (T, S)
    cum = np.cumsum(vals * _W[:, None], axis=0)               # CDF in y
    X = np.empty((len(GY), len(GS)))
    for j in range(len(GS)):
        X[:, j] = np.interp(GY, _T, cum[:, j], left=0.0, right=1.0)
    return np.clip(X, 0.0, 1.0 - 1e-16)


def fidelity(N, rho, H=None):
    H = H_table(rho) if H is None else H
    r = np.sqrt(1 - rho ** 2)
    # joint density f(s,y): use nodes (s,y) directly
    Sg, Yg = np.meshgrid(GS, GY, indexing="ij")
    fs = norm.pdf(Sg)
    fy = norm.pdf((Yg - rho * Sg) / r) / r
    w = np.exp((N - 1) * np.log1p(-(1.0 - H)))     # H^{N-1}, stable
    dens = fs * fy * w
    ds = GS[1] - GS[0]
    dy = GY[1] - GY[0]
    return float(N * dens.sum() * ds * dy)


def main():
    print("=" * 90)
    print("[A] exact objective Q(N,m) = rho_m * E[max_N]")
    print("=" * 90)
    Ns = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    print(f"{'rho':>5s} " + "".join(f"{n:<8d}" for n in Ns))
    for rho in (0.3, 0.5, 0.7, 0.9):
        q = [rho * Emax(n) for n in Ns]
        R.setdefault("Q", {})[str(rho)] = dict(zip(map(str, Ns), q))
        print(f"{rho:5.2f} " + "".join(f"{v:<8.3f}" for v in q))
    print("\n E[max_N] for N=1,16,256,4096:",
          [round(Emax(n), 3) for n in (1, 16, 256, 4096)])
    print(" (Q is strictly increasing in N for fixed m: no interior optimum in N)")

    print("\n" + "=" * 90)
    print("[B] selection fidelity F(N) : P(the pick is the pool's true best)")
    print("=" * 90)
    for rho in (0.5, 0.7, 0.9):
        H = H_table(rho)
        Fs = [fidelity(n, rho, H) for n in (4, 16, 64, 256, 1024)]
        R.setdefault("F", {})[str(rho)] = dict(zip("4 16 64 256 1024".split(),
                                                   [float(v) for v in Fs]))
        print(f"rho={rho}:  " + "".join(f"N={n}:{v:.3f}  " for n, v in
                                        zip((4, 16, 64, 256, 1024), Fs)))

    print("\n" + "=" * 90)
    print("[C] budget frontier  B = N*(1 + m*c)  (c = c_v/c_s), maximise Q")
    print("=" * 90)
    ms = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    for c in (0.01, 0.05, 0.1, 0.25, 0.5):
        print(f"\n-- c_v/c_s = {c}")
        print(f"{'rho':>5s} " + "".join(f"B={b:<11d}" for b in (256, 1024, 4096, 16384)))
        for rho in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            cells = []
            for B in (256, 1024, 4096, 16384):
                best = (-1, None, None)
                for m in ms:
                    N = max(1, int(B / (1 + m * c)))
                    q = rho_m(rho, m) * Emax(N)
                    if q > best[0]:
                        best = (float(q), m, N)
                cells.append(best)
                R.setdefault("frontier", {}).setdefault(str(c), {}).setdefault(
                    str(rho), {})[str(B)] = {"q": best[0], "m": best[1], "N": best[2]}
            print(f"{rho:5.2f} " + "".join(f"m={x[1]:<3d}N={x[2]:<6d}" for x in cells))

    print("\n" + "=" * 90)
    print("[D] gain over the standard policy m=1 at equal budget (percent)")
    print("=" * 90)
    for c in (0.01, 0.1, 0.5):
        print(f"\n-- c_v/c_s = {c}")
        print(f"{'rho':>5s} " + "".join(f"B={b:<10d}" for b in (256, 1024, 4096, 16384)))
        for rho in (0.3, 0.5, 0.7, 0.9):
            row = []
            for B in (256, 1024, 4096, 16384):
                f = R["frontier"][str(c)][str(rho)][str(B)]
                base = rho * Emax(max(1, int(B / (1 + c))))
                row.append(100 * (f["q"] / base - 1))
                R.setdefault("gain", {}).setdefault(str(c), {}).setdefault(
                    str(rho), {})[str(B)] = row[-1]
            print(f"{rho:5.2f} " + "".join(f"{v:+9.1f}%" for v in row))

    with open("work/iclr-research/data/pilot2b_results.json", "w") as f:
        json.dump(R, f, indent=1)


if __name__ == "__main__":
    main()
