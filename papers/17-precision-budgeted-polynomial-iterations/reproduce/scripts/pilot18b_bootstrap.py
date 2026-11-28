#!/usr/bin/env python3
"""
Pilot 18b - intervals and diagnostics for pilot18.

Reads pilot18_results.json and writes pilot18b_results.json:

 [1] the dimension-normalised floor constant: median over the 240 fit cells with a
     bootstrap 95% interval, per-family and per-d medians with intervals, the share of
     cells inside +-25%, and a one-sided constant that covers every cell (fit and
     held-out) with its conservatism.
 [2] the held-out grid: median ratio against the fit constant, interval, band.
 [3] the E5M2 repairs: the paired per-matrix gain of each mixed-precision rule over the
     shipped p=4 rule at ell = 1e-2 and 1e-1, median with a bootstrap interval.
 [4] which cells are the extremes, and how many cells of each grid were lost to a blow-up.
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "pilot18_results.json")))
OUT = {}
RS = np.random.RandomState(11)
B = 10000


def boot_median(v, b=B, rs=RS):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    idx = rs.randint(0, v.size, size=(b, v.size))
    med = np.median(v[idx], axis=1)
    return {"n": int(v.size), "point": float(np.median(v)),
            "lo": float(np.percentile(med, 2.5)), "hi": float(np.percentile(med, 97.5))}


def boot_ratio(num, den, b=B, rs=RS):
    """paired bootstrap of median(num/den) and of median(num)/median(den)."""
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    ok = np.isfinite(num) & np.isfinite(den) & (den > 0)
    num, den = num[ok], den[ok]
    g = num / den
    idx = rs.randint(0, g.size, size=(b, g.size))
    med = np.median(g[idx], axis=1)
    rat = np.median(num[idx], axis=1) / np.median(den[idx], axis=1)
    return {"n": int(g.size), "ratio_of_medians": float(np.median(num) / np.median(den)),
            "ratio_lo": float(np.percentile(rat, 2.5)), "ratio_hi": float(np.percentile(rat, 97.5)),
            "median_paired_gain": float(np.median(g)),
            "paired_lo": float(np.percentile(med, 2.5)), "paired_hi": float(np.percentile(med, 97.5)),
            "p10": float(np.percentile(g, 10)), "p90": float(np.percentile(g, 90)),
            "improved": int(np.sum(g < 1.0)), "worse": int(np.sum(g > 1.0))}


fams = ("logspace", "plateau", "gap", "power")
KEY = "Fd"

print("=" * 100)
print("[1] the dimension-normalised constant over the 240 fit cells")
print("=" * 100)
fit = R["a1_cells"]
v = np.array([c["over_pred"][KEY] for c in fit if KEY in c["over_pred"]], float)
v = v[np.isfinite(v)]
OUT["fit"] = boot_median(v)
print(f"  median {np.median(v):.3f}  95% CI [{OUT['fit']['lo']:.3f}, {OUT['fit']['hi']:.3f}]  n={v.size}")
med = float(np.median(v))
OUT["fit"]["share_25"] = float(np.mean(np.abs(v / med - 1) <= 0.25))
OUT["fit"]["share_40"] = float(np.mean(np.abs(v / med - 1) <= 0.40))
OUT["fit"]["lo_cell"] = float(v.min())
OUT["fit"]["hi_cell"] = float(v.max())
OUT["fit"]["one_sided_max"] = float(v.max())
print(f"  cells inside +-25%: {OUT['fit']['share_25'] * 100:.1f}%   +-40%: {OUT['fit']['share_40'] * 100:.1f}%")
print(f"  cell range [{v.min():.3f}, {v.max():.3f}]  (one-sided constant {v.max():.3f})")

OUT["fit_family"] = {}
for f in fams:
    w = np.array([c["over_pred"][KEY] for c in fit if c["spectrum"] == f and KEY in c["over_pred"]],
                 float)
    OUT["fit_family"][f] = boot_median(w)
    print(f"  {f:>9}: median {OUT['fit_family'][f]['point']:.3f} "
          f"[{OUT['fit_family'][f]['lo']:.3f}, {OUT['fit_family'][f]['hi']:.3f}]")
fm = [OUT["fit_family"][f]["point"] for f in fams]
print(f"  family medians span {max(fm) / min(fm):.2f}x")

OUT["fit_d"] = {}
for d in (32, 48, 64, 128, 256):
    w = np.array([c["over_pred"][KEY] for c in fit if c["d"] == d and KEY in c["over_pred"]], float)
    OUT["fit_d"][d] = boot_median(w)
    print(f"  d={d:4d}: median {OUT['fit_d'][d]['point']:.3f} "
          f"[{OUT['fit_d'][d]['lo']:.3f}, {OUT['fit_d'][d]['hi']:.3f}]")
dm = [OUT["fit_d"][d]["point"] for d in (32, 48, 64, 128, 256)]
print(f"  d medians span {max(dm) / min(dm):.2f}x")

OUT["fit_p"] = {}
for p in (8, 12):
    for key in (KEY, "F"):
        w = np.array([c["over_pred"][key] for c in fit if c["p"] == p and key in c["over_pred"]],
                     float)
        OUT["fit_p"][f"p{p}_{key}"] = boot_median(w)
        print(f"  p={p:2d} {key:>2}: median {OUT['fit_p'][f'p{p}_{key}']['point']:.3f} "
              f"[{OUT['fit_p'][f'p{p}_{key}']['lo']:.3f}, {OUT['fit_p'][f'p{p}_{key}']['hi']:.3f}]")

lost = {}
for name, cells in (("fit", fit), ("heldout", R["a3_heldout"])):
    lost[name] = {"n": len(cells), "without_ratio": int(sum(1 for c in cells if KEY not in c["over_pred"])),
                  "diverged": int(sum(1 for c in cells if c.get("diverged")))}
OUT["lost"] = lost
print(f"  cells lost to a blow-up: fit {lost['fit']['diverged']}/{lost['fit']['n']}, "
      f"held-out {lost['heldout']['diverged']}/{lost['heldout']['n']}")
for name, cells in (("fit", fit), ("heldout", R["a3_heldout"])):
    dd = [c for c in cells if c.get("diverged")]
    if dd:
        print(f"   {name} blown up: " + ", ".join(
            f"{c['spectrum']}/d{c['d']}/L{c['ell']:g}/p{c['p']}/s{c['seed']}" for c in dd[:8]))

print("\n" + "=" * 100)
print("[2] the extremes of the fit grid (where the 2.2x band comes from)")
print("=" * 100)
order = np.argsort(v)
cells_ok = [c for c in fit if KEY in c["over_pred"]]
vals = np.array([c["over_pred"][KEY] for c in cells_ok], float)
order = np.argsort(vals)
for i in list(order[:5]) + list(order[-5:]):
    c = cells_ok[i]
    print(f"  {vals[i]:7.3f}  {c['spectrum']}/d{c['d']}/L{c['ell']:g}/p{c['p']}/seed{c['seed']}"
          f"  err {c['err']:.2e} EK {c['EK']:.1e} u {c['u']:.1e}")

print("\n" + "=" * 100)
print("[3] the held-out grid against the fit constant")
print("=" * 100)
ho = R["a3_heldout"]
hv = np.array([c["over_pred"][KEY] for c in ho if KEY in c["over_pred"]], float)
OUT["heldout"] = boot_median(hv)
OUT["heldout"]["ratio_to_fit"] = OUT["heldout"]["point"] / med
OUT["heldout"]["lo_cell"] = float(hv.min())
OUT["heldout"]["hi_cell"] = float(hv.max())
OUT["heldout"]["share_25"] = float(np.mean(np.abs(hv / med - 1) <= 0.25))
print(f"  median {np.median(hv):.3f} [{OUT['heldout']['lo']:.3f}, {OUT['heldout']['hi']:.3f}]  "
      f"= {OUT['heldout']['ratio_to_fit']:.3f}x the fit constant")
print(f"  cell range [{hv.min():.3f}, {hv.max():.3f}], inside fit +-25%: "
      f"{OUT['heldout']['share_25'] * 100:.0f}%")
print(f"  the fit one-sided constant {v.max():.3f} covers the held-out max {hv.max():.3f}: "
      f"{v.max() >= hv.max()}, conservatism {v.max() / hv.max():.2f}x")
OUT["heldout"]["covered_by_one_sided"] = bool(v.max() >= hv.max())
print("  held-out medians by (ell, p):")
OUT["heldout_split"] = {}
for L in (5e-2, 3e-3):
    for p in (6, 16):
        w = np.array([c["over_pred"][KEY] for c in ho if c["ell"] == L and c["p"] == p
                      and KEY in c["over_pred"]], float)
        OUT["heldout_split"][f"L{L:g}_p{p}"] = boot_median(w)
        print(f"   ell={L:6g} p={p:2d}: median {np.median(w):.3f} n={w.size}")

print("\n" + "=" * 100)
print("[4] E5M2 repairs: paired gain over the shipped p=4 rule")
print("=" * 100)
OUT["gains"] = {}
for L in ("0.1", "0.01"):
    base = R["b2_cost"][f"L{L}_ship_p4_s1.010"]["cells"]
    for name, res in R["b2_cost"].items():
        if not name.startswith(f"L{L}_") or name.endswith("ship_p4_s1.010"):
            continue
        num = [c["sub"] for c in base]
        den = [c["sub"] for c in res["cells"]]
        g = boot_ratio(num, den)
        OUT["gains"][f"L{L}_{name.split('_', 1)[1]}"] = g
        print(f"  ell={L:>4} {name.split('_', 1)[1]:<26} gain {g['ratio_of_medians']:.2f}x "
              f"[{g['ratio_lo']:.2f},{g['ratio_hi']:.2f}]  paired median {g['median_paired_gain']:.2f}x "
              f"improved {g['improved']}/{g['improved'] + g['worse']}")

print("\n" + "=" * 100)
print("[5] the floor at ell = 1e-3: the cap, per rule")
print("=" * 100)
OUT["cap_census"] = {}
for name, res in R["b1_ell1e-3"].items():
    OUT["cap_census"][name] = dict(n_cap=res["n_cap"], n_div=res["n_div"],
                                   median=res["median_sub"], max=res["max_sub"])
    print(f"  {name:<26} at the cap {res['n_cap']:2d}/12  blow-ups {res['n_div']:2d}/12  "
          f"floor {res['median_sub']:.3f}")

with open(os.path.join(HERE, "pilot18b_results.json"), "w") as f:
    json.dump(OUT, f, indent=1, default=float)
print("\nwrote pilot18b_results.json")
