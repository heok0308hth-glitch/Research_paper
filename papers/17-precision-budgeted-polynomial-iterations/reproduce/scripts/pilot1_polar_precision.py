#!/usr/bin/env python3
"""
Pilot 1 - Precision floor and optimal stopping index for odd-polynomial
matrix-sign (polar) iterations, with faithful bf16/fp16 storage emulation.

Faithful kernel model (matches the public Polar Express / Muon loop
   A = X X^T ;  B = b*A + c*(A A) ;  X = a*X + B X ):
  operands are stored in p significant binary digits, products are formed
  exactly (fp32/fp64 accumulation), and only *stored* quantities are rounded
  (B and X).  A is never materialised, so it is not rounded.
"""
import json
import numpy as np

np.random.seed(0)
R = {}


def q(x, p):
    """Round to p significant binary digits, round-half-to-even."""
    if p is None or p >= 30:
        return x
    x = np.asarray(x, dtype=np.float64)
    m, e = np.frexp(x)
    s = float(2 ** (p - 1))
    return np.ldexp(np.round(m * s) / s, e)


# ------------------------------------------------------- polynomial designs
def optimal_quintic(l, u):
    if 1 - 5e-6 <= l / u:
        return (15 / 8) / u, (-10 / 8) / (u ** 3), (3 / 8) / (u ** 5)
    qq = (3 * l + u) / 4
    r = (l + 3 * u) / 4
    E, old_E = np.inf, None
    while not old_E or abs(old_E - E) > 1e-15:
        old_E = E
        LHS = np.array([[l, l**3, l**5, 1],
                        [qq, qq**3, qq**5, -1],
                        [r, r**3, r**5, 1],
                        [u, u**3, u**5, -1]])
        a, b, c, E = np.linalg.solve(LHS, np.ones(4))
        qq, r = np.sqrt((-3 * b + np.array([-1, 1]) * np.sqrt(9 * b**2 - 20 * a * c)) / (10 * c))
    return float(a), float(b), float(c)


def pe_coeffs(l0, K, cushion=0.02407327424182761, safety=1.0):
    u, l, out = 1.0, l0, []
    for _ in range(K):
        a, b, c = optimal_quintic(max(l, cushion * u), u)
        pl = a * l + b * l**3 + c * l**5
        pu = a * u + b * u**3 + c * u**5
        r = 2 / (pl + pu)
        a, b, c = a * r, b * r, c * r
        a, b, c = a / safety, b / safety**3, c / safety**5
        out.append((a, b, c))
        l = a * l + b * l**3 + c * l**5
        u = 2 - l
    return out


def lsq_coeffs(l0, K, n=800):
    """L2 (least-squares) fit of 1 on [l,u] by an odd quintic; the
    'less oscillatory' counterpart of the L-infinity design."""
    u, l, out = 1.0, l0, []
    for _ in range(K):
        x = np.linspace(l, u, n)
        A = np.stack([x, x**3, x**5], axis=1)
        a, b, c = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
        out.append((a, b, c))
        l = a * l + b * l**3 + c * l**5
        u = 2 - l
    return out


def ns_coeffs(K, deg=5):
    if deg == 5:
        return [(15 / 8, -10 / 8, 3 / 8)] * K
    return [(3 / 2, -1 / 2, 0.0)] * K


# ------------------------------------------------------------- iterations
def scalar_orbit(coeffs, x0):
    x = np.asarray(x0, float)
    traj = [x.copy()]
    maxamp = np.ones_like(x)
    for (a, b, c) in coeffs:
        traj.append(a * x + b * x**3 + c * x**5)
        x = traj[-1]
    # sensitivity: product of the remaining derivatives, maximised over the orbit
    for j in range(len(coeffs)):
        amp = np.ones_like(x0)
        for l in range(j, len(coeffs)):
            a, b, c = coeffs[l]
            xl = traj[l]
            amp = amp * np.abs(a + 3 * b * xl**2 + 5 * c * xl**4)
        maxamp = np.maximum(maxamp, amp)
    return np.array(traj), maxamp


def mat_orbit(coeffs, M, p=None, track=False):
    X = M / np.linalg.norm(M, 2)
    X = q(X, p)
    norms = [np.linalg.norm(X, 2)]
    for (a, b, c) in coeffs:
        A = X @ X.T                      # p-bit operands, exact products
        B = b * A + c * (A @ A)          # kept in fp32 registers
        X = q(a * X + q(B, p) @ X, p)    # B and X stored in p bits
        norms.append(np.linalg.norm(X, 2))
    return (X, norms) if track else X


def main():
    L = 1e-3
    KM = 10
    grid = np.logspace(np.log10(L), 0, 500)
    designs = {
        "NS5": ns_coeffs(KM, 5),
        "NS3": ns_coeffs(KM, 3),
        "PE": pe_coeffs(L, KM),
        "PE_cushion0": pe_coeffs(L, KM, cushion=0.0),
        "PE_safety1.01": pe_coeffs(L, KM, safety=1.01),
        "LSQ5": lsq_coeffs(L, KM),
    }

    print("=" * 96)
    print("[A] exact arithmetic: max_{sigma in [1e-3,1]} |P_K(sigma) - 1|")
    print("=" * 96)
    print(f"{'design':15s}" + "".join(f"K={k:<4d}" for k in range(1, KM + 1)))
    exact = {}
    for nm, co in designs.items():
        tr, _ = scalar_orbit(co, grid)
        e = [float(np.max(np.abs(tr[k] - 1))) for k in range(1, KM + 1)]
        exact[nm] = e
        print(f"{nm:15s}" + "".join(f"{v:6.1e} " for v in e))
    R["exact"] = exact

    print("\n" + "=" * 96)
    print("[B] worst-case sensitivity max_{j<=K, sigma} prod_{l>=j} |p_l'(sigma_l)|")
    print("    (amplification applied to one rounding error injected at step j)")
    print("=" * 96)
    print(f"{'design':15s}" + "".join(f"K={k:<8d}" for k in [1, 2, 3, 4, 5, 6, 8, 10]))
    sens = {}
    for nm, co in designs.items():
        row = []
        for k in [1, 2, 3, 4, 5, 6, 8, 10]:
            _, ma = scalar_orbit(co[:k], grid)
            row.append(float(np.max(ma)))
        sens[nm] = row
        print(f"{nm:15s}" + "".join(f"{v:9.2e} " for v in row))
    R["sensitivity"] = sens

    print("\n" + "=" * 96)
    print("[C] scalar elementwise model (rounding after every product/sum): "
          "max error vs K")
    print("=" * 96)
    elem = {}
    for pname, p in [("bf16", 8), ("fp16", 11), ("fp32", 24)]:
        print(f"\n-- {pname}")
        print(f"{'design':15s}" + "".join(f"K={k:<4d}" for k in range(1, KM + 1)))
        elem[pname] = {}
        for nm, co in designs.items():
            x = np.asarray(grid, float)
            errs = []
            for (a, b, c) in co:
                x3, x5 = q(q(x * x, p) * x, p), None
                x3 = q(q(q(x * x, p) * x, p), p)
                x5 = q(q(x3 * q(x * x, p), p), p)
                x = q(q(a * x, p) + q(q(b * x3, p) + q(c * x5, p), p), p)
                errs.append(float(np.max(np.abs(x - 1))))
            elem[pname][nm] = errs
            kst = int(np.argmin(errs)) + 1
            print(f"{nm:15s}" + "".join(f"{v:6.1e} " for v in errs)
                  + f" k*={kst} floor={min(errs):.1e}")
    R["elementwise"] = elem

    # ------------------------------------------------------------- matrices
    d = 64
    U, _ = np.linalg.qr(np.random.randn(d, d))
    V, _ = np.linalg.qr(np.random.randn(d, d))
    sig = np.logspace(0, -3, d)          # power-law spectrum, sigma in [1e-3,1]
    M = (U * sig) @ V.T
    ref = U @ V.T

    print("\n" + "=" * 96)
    print("[D] matrix kernel model, d=64, spectrum logspace(0,-3): "
          "||X_K - UV^T||_2")
    print("=" * 96)
    mat = {}
    for pname, p in [("exact", None), ("bf16", 8), ("fp16", 11), ("fp32", 24)]:
        mat[pname] = {}
        for nm in ("PE", "PE_safety1.01", "LSQ5", "NS5"):
            errs, norms = [], []
            for k in range(1, 9):
                X, nn = mat_orbit(designs[nm][:k], M, p, track=True)
                errs.append(float(np.linalg.norm(X - ref, 2)))
                norms.append(nn[-1])
            mat[pname][nm] = {"err": errs, "norm": norms}
            print(f"{pname:6s} {nm:14s}" + "".join(f"{v:7.1e} " for v in errs))
            print(f"{'':21s}" + "".join(f"{v:7.2f} " for v in norms) + "  <- ||X_K||_2")
    R["matrix"] = mat

    print("\n" + "=" * 96)
    print("[E] error by spectral band (bf16, K=8): top / middle / bottom third")
    print("=" * 96)
    band = {}
    for nm in ("PE", "PE_safety1.01", "LSQ5", "NS5"):
        X = mat_orbit(designs[nm][:8], M, 8)
        XtX = X.T @ X
        d2 = np.sort(np.linalg.eigvalsh(XtX))
        band[nm] = {"min_eig": float(d2[0]), "max_eig": float(d2[-1])}
        print(f"{nm:15s} eig(X^T X) range [{d2[0]:.3e}, {d2[-1]:.3e}] "
              f"(ideal 1.0)")
    R["band"] = band

    with open("work/iclr-research/data/pilot1_results.json", "w") as f:
        json.dump(R, f, indent=1)


if __name__ == "__main__":
    main()
