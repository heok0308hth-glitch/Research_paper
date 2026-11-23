"""Reproduction script for paper 16.

    python3 reproduce.py all      # E1-E5, six figures (about 10 minutes on 2 cores)
    python3 reproduce.py smoke    # the same code path with two seeds per cell

Outputs results/*.json and figures/*.png next to this file.  Every random seed
is a constant in the script, so a rerun reproduces the numbers in the paper.
"""
import json, os, sys
import numpy as np
from math import sqrt, log
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from math import exp, log, sqrt, pi
from scipy.special import erf as _erf_sp

SQRT2 = sqrt(2.0)
INV_SQRT2PI = 1.0 / sqrt(2.0 * pi)
_erf = _erf_sp


def norm_cdf(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / SQRT2))


def norm_pdf(x):
    return INV_SQRT2PI * np.exp(-0.5 * np.asarray(x, dtype=float) ** 2)


def pickoff_cost(half, sigma, tau):
    """E[(|X| - half)^+] for X ~ N(0, tau sigma^2): the per-share cost of a
    stale quote at half-spread `half`."""
    s = sigma * sqrt(max(tau, 1.0))
    u = np.maximum(half / np.maximum(s, 1e-12), 0.0)
    return 2.0 * s * (norm_pdf(u) - u * (1.0 - norm_cdf(u)))


BASE = dict(
    N=100,             # market makers
    T=900,             # ticks per run
    burn=250,          # warm-up ticks
    Delta=1.0,         # tick size
    tau=2,             # latency (ticks)
    gamma=0.0,         # drawdown reflexivity (lots per tick per tick of drawdown)
    gamma_v=0.0,       # volatility reflexivity (share of branching headroom)
    n_cap=0.90,        # branching-ratio ceiling
    mu0=1.2,           # base taker arrivals per tick
    alpha0=0.020,      # baseline Hawkes amplitude
    beta=0.080,        # Hawkes decay per tick
    a_med=0.90,        # median risk aversion
    sigma_a=0.35,      # lognormal dispersion
    H=8.0,             # holding horizon (ticks)
    B_risk=1.00,       # risk budget (lots)
    m0=0.80,           # fixed half-spread markup (ticks)
    as_weight=1.0,     # weight of the adverse-selection term
    sig_as=0.60,       # scale of the participation response
    Q_cap=1.00,        # median maker capacity (lots)
    sigma_Q=0.40,      # lognormal dispersion of capacity
    k_imp=0.60,        # impact coefficient (ticks)
    alpha_imp=0.5,     # impact curvature
    D_deep=4.0,        # latent depth, unresponsive to volatility
    tau0=2.0,          # latency at which half the standing depth has gone stale
    eq_scale=8.0,      # equity loss (price units) that halves a maker's capacity
    v_f=0.0,           # fundamental innovation sd per tick
    theta=0.015,       # mid mean reversion to fundamental
    flat=0.010,        # inventory flattening per tick
    vg=1.50,           # value-sector demand slope (lots per tick per tick)
    vm=6.0,            # value-sector threshold (ticks)
    vmax=20.0,         # value-sector maximum flow (lots per tick)
    lot=1.0,           # mean taker order size
    vol_halflife=15.0,
    sigma_ref=0.30,
    rho=1.0,           # (kept for reference: trend flow = gamma mu0 dd/Delta)
    dd_sat=10.0,       # drawdown (ticks) at which trend-following saturates
    shock=0.0,         # total sell programme (lots)
    shock_len=20,      # ticks over which it is worked
    shock_delay=20,    # ticks after burn before it starts
    persist=1.0,       # share of the programme that survives a halt
    limit_L=0.0,       # breaker distance below the reference (0 = none)
    halt=30,           # halt length (ticks)
    gap=0.05,          # reopening gap per unit of pending flow over depth
    cap_recover=0.0,   # capacity released during a halt
    magnet=0.0,        # front-running strength near the limit
    magnet_width=8.0,  # anticipation window (ticks)
    fba=0,             # batch length (ticks; 0 = continuous)
    dyn_tick=0.0,      # tick multiplier per unit of sig/sigma_ref
    asym=0.0,          # asymmetric inventory penalty
    seed=0,
)


class Sim:
    def __init__(self, **kw):
        p = dict(BASE)
        p.update(kw)
        self.p = p
        self.rng = np.random.default_rng(p["seed"])
        N = p["N"]
        self.a = p["a_med"] * np.exp(self.rng.normal(0.0, p["sigma_a"], N))
        self.Qc = p["Q_cap"] * np.exp(self.rng.normal(0.0, p["sigma_Q"], N))
        self.q = np.zeros(N)
        self.cash = np.zeros(N)
        self.sig = 0.5 * p["sigma_ref"]
        self.var = self.sig ** 2
        self.R = 0.0
        self.lam = p["mu0"]
        self.m = 0.0
        self.F = 0.0
        self.mbuf = np.zeros(p["tau"] + 1)
        self.bi = 0
        self.Ref = None
        self.Ref0 = None
        self.pending = 0.0
        self.halt_until = -1
        self.halts = 0
        self.halt_ticks = 0
        self.batch_Q = 0.0
        self.rec = []

    def step(self, t):
        p = self.p
        rng = self.rng
        rec = dict(t=t, D=np.nan, S=np.nan, lam=np.nan, m=self.m, sig=self.sig,
                   halted=0.0, dd=0.0, V=0.0, Q=0.0)
        hl = 1.0 - 0.5 ** (1.0 / p["vol_halflife"])
        self.sig = min(sqrt(max(self.var, 1e-12)), 20.0 * p["Delta"])

        if t < self.halt_until:
            rec.update(halted=1.0, m=self.m)
            self.halt_ticks += 1
            if p["cap_recover"]:
                self.q *= (1.0 - p["cap_recover"])
            # the pressure that would have traded is held for the reopening
            lam_h = p["mu0"] + p["alpha0"] * self.R
            k_h = int(min(rng.poisson(max(lam_h, 1e-9)), 300))
            self.R = exp(-p["beta"]) * self.R + k_h
            pend = k_h * p["lot"] * 0.5
            if p["shock"] > 0:
                t0 = p["burn"] + p["shock_delay"]
                if t0 <= t < t0 + p["shock_len"]:
                    pend += p["shock"] / p["shock_len"]
            self.pending += pend
            self.bi = (self.bi + 1) % (p["tau"] + 1)
            self.mbuf[self.bi] = self.m
            self.rec.append(rec)
            return rec
        if self.pending > 0:
            self._pending_in = self.pending
            self.pending = 0.0

        # ---- flow: Hawkes with a bounded reflexivity in volatility
        g = self.sig / (self.sig + p["sigma_ref"])
        n0 = p["alpha0"] / (1.0 - exp(-p["beta"]))
        n = min(n0 + (p["n_cap"] - n0) * p["gamma_v"] * g, p["n_cap"])
        lam = p["mu0"] + n * (1.0 - exp(-p["beta"])) * self.R
        self.lam = lam
        rec["lam"] = lam
        k = int(min(rng.poisson(max(lam, 1e-9)), 300))
        self.R = exp(-p["beta"]) * self.R + k

        if self.Ref is None and t >= p["burn"]:
            self.Ref = self.m
            self.Ref0 = self.m
        dd = 0.0 if self.Ref0 is None else max(self.Ref0 - self.m, 0.0)
        rec["dd"] = dd

        m_obs = self.mbuf[self.bi]
        sig = self.sig
        Dlt = min(p["Delta"] * (1.0 + p["dyn_tick"] * sig / p["sigma_ref"]), 20.0 * p["Delta"])
        a = self.a
        tilt = np.tanh(1.5 * self.q / np.maximum(self.Qc, 1e-9)) if p["asym"] else 0.0
        a_bid = a * (1.0 + p["asym"] * tilt)
        a_ask = a * (1.0 - p["asym"] * tilt)
        risk = 0.5 * a * sig ** 2 * p["H"]
        risk_b = 0.5 * a_bid * sig ** 2 * p["H"]
        risk_a = 0.5 * a_ask * sig ** 2 * p["H"]
        h = p["m0"] * Dlt + risk
        for _ in range(2):
            h = p["as_weight"] * pickoff_cost(h, sig, p["tau"]) + risk + p["m0"] * Dlt
        h_touch = max(0.5 * Dlt, float(np.sort(h)[int(0.30 * len(h))]))
        h_touch = 0.5 * Dlt * max(1.0, round(2.0 * h_touch / Dlt))
        scale = p["sig_as"] * sig * sqrt(max(p["tau"], 1.0)) + 1e-9
        wgt = 1.0 / (1.0 + np.exp(-np.clip((h_touch - h) / scale, -40.0, 40.0)))
        eq = self.cash + self.q * self.m
        s_eq = np.clip(1.0 + eq / max(p["eq_scale"], 1e-6), 0.10, 1.6)
        room = np.maximum(self.Qc * s_eq - np.abs(self.q), 0.0)
        d_i = np.minimum(1.0, p["B_risk"] / (1.0 + a * sig ** 2 * p["H"] / p["B_risk"])) * wgt
        d_i = np.minimum(d_i, room)
        D = float(d_i.sum())
        rec["D"] = D
        rec["S"] = 2.0 * h_touch

        # ---- signed flow
        imb = (self.m - self.F) / max(sig * sqrt(10.0), 1e-9)
        p_buy = float(np.clip(0.5 + 0.14 * np.tanh(-imb), 0.05, 0.95))
        Q = 0.0
        if k:
            sgn = np.where(rng.random(k) < p_buy, 1.0, -1.0)
            sz = rng.lognormal(mean=log(p["lot"]) - 0.1, sigma=0.5, size=k)
            Q = float((sgn * sz).sum())
        if dd > 0 and p["gamma"] > 0:
            Q -= p["gamma"] * p["mu0"] * dd / (1.0 + dd / max(p["dd_sat"] * Dlt, 1e-9))
        if p["limit_L"] > 0 and self.Ref is not None and p["magnet"] > 0:
            gapL = p["limit_L"] - dd
            if 0 < gapL < p["magnet_width"] and Q < 0:
                Q *= (1.0 + p["magnet"] * (1.0 - gapL / p["magnet_width"]))
        if p["shock"] > 0:
            t0 = p["burn"] + p["shock_delay"]
            if t0 <= t < t0 + p["shock_len"]:
                Q -= p["shock"] / p["shock_len"]
        # value sector: patient demand around the fundamental
        mis = self.F - self.m
        if mis > p["vm"]:
            Q += min(p["vg"] * (mis - p["vm"]), p["vmax"])
        elif mis < -p["vm"]:
            Q -= min(p["vg"] * (-mis - p["vm"]), p["vmax"])
        if getattr(self, "_pending_in", 0.0) > 0:
            Q -= self._pending_in
            self._pending_in = 0.0
        rec["Q"] = Q

        if p["fba"]:
            self.batch_Q += Q
            if (t % p["fba"]) == p["fba"] - 1:
                Q = self.batch_Q
                self.batch_Q = 0.0
            else:
                Q = 0.0
        rec["V"] = abs(Q)

        # ---- impact and price
        D_eff = D / (1.0 + p["tau"] / p["tau0"]) + p["D_deep"]
        x = Q / max(D_eff, 1e-9)
        imp = p["k_imp"] * np.sign(x) * abs(x) ** p["alpha_imp"] * Dlt
        imp = float(np.clip(imp, -8.0 * Dlt, 8.0 * Dlt))
        self.F += rng.normal(0.0, p["v_f"])
        ret = imp + p["theta"] * (self.F - self.m) + rng.normal(0.0, p["v_f"] * 0.5)
        self.m += ret
        self.var = (1.0 - hl) * self.var + hl * ret * ret
        self.var = min(self.var, (20.0 * p["Delta"]) ** 2)
        rec["m"] = self.m
        rec["sig"] = self.sig

        # ---- fills at the stale quote, marked at the new mid
        share = d_i / max(d_i.sum(), 1e-12)
        qty = Q * share
        buy = qty > 0                       # maker buys when the taker sells
        self.q += qty
        self.cash -= np.where(buy, qty * (m_obs - h_touch), qty * (m_obs + h_touch))
        self.q *= (1.0 - p["flat"])

        # ---- breaker
        if p["limit_L"] > 0 and self.Ref is not None and self.m <= self.Ref - p["limit_L"]:
            self.halts += 1
            self.halt_until = t + p["halt"]
            pend = abs(Q)
            self.m -= p["gap"] * pend / max(D_eff, 1e-6) * Dlt
            rec["halted"] = 1.0
            rec["m"] = self.m
            self.Ref = self.m

        self.bi = (self.bi + 1) % (p["tau"] + 1)
        self.mbuf[self.bi] = self.m
        self.rec.append(rec)
        return rec


def run(**kw):
    p = dict(BASE)
    p.update(kw)
    s = Sim(**p)
    for t in range(p["T"]):
        s.step(t)
    r = s.rec[p["burn"]:]
    m = np.array([x["m"] for x in r])
    D = np.array([x["D"] for x in r], dtype=float)
    S = np.array([x["S"] for x in r], dtype=float)
    lam = np.array([x["lam"] for x in r], dtype=float)
    dd = np.array([x["dd"] for x in r], dtype=float)
    h = np.array([x["halted"] for x in r], dtype=float)
    D0 = float(np.nanmedian(D[:40])) if len(D) > 40 else np.nan
    with np.errstate(invalid="ignore"):
        return dict(
            m=m, D=D, S=S, lam=lam, dd=dd, h=h,
            D0=D0, D_min=float(np.nanmin(D)), D_end=float(np.nanmean(D[-60:])),
            S_mean=float(np.nanmean(S)), S_max=float(np.nanmax(S)),
            dd_max=float(np.nanmax(dd)) if len(dd) else 0.0,
            m_end=float(m[-1]) if len(m) else 0.0,
            halts=int(s.halts), halt_ticks=int(s.halt_ticks),
            ret_sd=float(np.nanstd(np.diff(m))) if len(m) > 2 else np.nan,
        )



OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def _stat(r, Delta):
    return dict(dd=r["dd_max"], D_end=r["D_end"] / max(r["D0"], 1e-9), Dmin=r["D_min"], D0=r["D0"], S=r["S_mean"],
                halt=r["halts"], sig=r["ret_sd"], m_end=r["m_end"] / Delta)


def ensemble(n, **kw):
    Delta = kw.get("Delta", 1.0)
    rows = [_stat(run(seed=1000 + s, **kw), Delta) for s in range(n)]
    dd = np.array([x["dd"] for x in rows])
    D0 = float(np.nanmedian([x["D0"] for x in rows]))
    coll = np.array([x["D_end"] for x in rows]) < 0.2
    k = max(1, int(0.05 * len(dd)))
    return dict(n=n, dd=float(dd.mean()), dd_se=float(dd.std(ddof=1) / sqrt(n)),
                dd_med=float(np.median(dd)), dd_p95=float(np.quantile(dd, 0.95)),
                cvar95=float(np.sort(dd)[-k:].mean()), p_collapse=float(coll.mean()),
                D0=D0, Dmin=float(np.mean([x["Dmin"] for x in rows])),
                D_end=float(np.mean([x["D_end"] for x in rows])),
                S=float(np.mean([x["S"] for x in rows])),
                halts=float(np.mean([x["halt"] for x in rows])),
                sig=float(np.mean([x["sig"] for x in rows])),
                m_end=float(np.mean([x["m_end"] for x in rows])))


def gamma_half(gammas, ys):
    gs = np.asarray(gammas, float); ys = np.asarray(ys, float)
    for i in range(len(ys) - 1):
        if (ys[i] - 0.5) * (ys[i + 1] - 0.5) <= 0 and ys[i] != ys[i + 1]:
            t = (0.5 - ys[i]) / (ys[i + 1] - ys[i])
            return float(gs[i] + t * (gs[i + 1] - gs[i]))
    return float("nan")


def gamma_at(gammas, ys, level):
    gs = np.asarray(gammas, float); ys = np.asarray(ys, float)
    for i in range(len(ys) - 1):
        if (ys[i] - level) * (ys[i + 1] - level) <= 0 and ys[i] != ys[i + 1]:
            t = (level - ys[i]) / (ys[i + 1] - ys[i])
            return float(gs[i] + t * (gs[i + 1] - gs[i]))
    return float("nan")


def logistic_gamma_c(gammas, ps):
    gs = np.asarray(gammas, float); ps = np.asarray(ps, float)
    ok = (ps > 0.02) & (ps < 0.98)
    if ok.sum() < 2:
        return float("nan"), float("nan")
    x = np.log(ps[ok] / (1 - ps[ok]))
    A = np.vstack([gs[ok], np.ones(ok.sum())]).T
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    if coef[0] <= 0:
        return float("nan"), float(coef[0])
    return float(-coef[1] / coef[0]), float(coef[0])


def exp1(nseed=20, tag=""):
    gammas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    path = os.path.join(OUT, "e1.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    res.setdefault("gammas", gammas)
    res["shock"] = 90.0
    cells = res.setdefault("cells", [])
    done = {(c["tau"], c["Delta"]) for c in cells}
    for tau in (1, 2, 4, 8):
        for Delta in (0.5, 1.0, 2.0):
            if (tau, Delta) in done and not tag:
                continue
            if tag and tag != "%d_%.1f" % (tau, Delta):
                continue
            row = [dict(gamma=g, **ensemble(nseed, gamma=g, tau=tau, Delta=Delta, shock=90.0))
                   for g in gammas]
            rho = [r["D_end"] for r in row]
            gc = gamma_half(gammas, rho)
            gc2 = gamma_half(gammas, [r["p_collapse"] for r in row])
            w = gamma_at(gammas, rho, 0.2) - gamma_at(gammas, rho, 0.8)
            sl = float(4.0 / w) if w and w == w and w > 0 else float("nan")
            cells = [c for c in cells if (c["tau"], c["Delta"]) != (tau, Delta)]
            cells.append(dict(tau=tau, Delta=Delta, rows=row, gamma_c=gc, slope=sl,
                              gamma_c_p=gc2, width=w))
            cells.sort(key=lambda c: (c["tau"], c["Delta"]))
            res["cells"] = cells
            print("E1 tau=%d D=%.1f gc=%.3f slope=%.2f" % (tau, Delta, gc, sl), flush=True)
            json.dump(res, open(path, "w"), indent=1)
    res["latency"] = [dict(tau=t, **ensemble(nseed, gamma=0.0, tau=t, shock=0.0))
                      for t in (1, 2, 4, 8, 16)]
    res["ticks"] = [dict(Delta=d, **ensemble(nseed, gamma=0.0, Delta=d, shock=90.0))
                    for d in (0.5, 1.0, 2.0, 4.0)]
    json.dump(res, open(os.path.join(OUT, "e1.json"), "w"), indent=1)
    return res


def exp2(nseed=30):
    gammas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    res = dict(gammas=gammas, cells=[])
    for N in (25, 50, 100, 200, 400):
        Qc = 100.0 / N
        row = [dict(gamma=g, **ensemble(nseed, gamma=g, N=N, Q_cap=Qc, shock=90.0))
               for g in gammas]
        rho = [r["D_end"] for r in row]
        gc = gamma_half(gammas, rho)
        w = gamma_at(gammas, rho, 0.2) - gamma_at(gammas, rho, 0.8)
        res["cells"].append(dict(N=N, Q_cap=Qc, rows=row, gamma_c=gc, width=w,
                                 slope=float(4.0 / w) if w and w == w and w > 0 else float("nan")))
        print("E2 N=%d gc=%.3f width=%.3f" % (N, gc, res["cells"][-1]["width"]), flush=True)
        json.dump(res, open(os.path.join(OUT, "e2.json"), "w"), indent=1)
    return res


def exp3(nseed=60):
    res = dict(L=[4.0, 6.0, 8.0], magnet=[0.0, 1.0, 4.0], shock=90.0, gamma=0.5, cells=[])
    for L in res["L"]:
        for mg in res["magnet"]:
            off, on = [], []
            for s in range(nseed):
                kw = dict(gamma=0.5, tau=2, shock=90.0)
                off.append(_stat(run(seed=2000 + s, **kw), 1.0))
                on.append(_stat(run(seed=2000 + s, limit_L=L, magnet=mg, **kw), 1.0))
            d_off = np.array([x["dd"] for x in off]); d_on = np.array([x["dd"] for x in on])
            k = max(1, int(0.05 * nseed))
            row = dict(L=L, magnet=mg,
                       dd_off=float(d_off.mean()), dd_on=float(d_on.mean()),
                       cvar_off=float(np.sort(d_off)[-k:].mean()),
                       cvar_on=float(np.sort(d_on)[-k:].mean()),
                       p95_off=float(np.quantile(d_off, 0.95)),
                       p95_on=float(np.quantile(d_on, 0.95)),
                       p_over_L_off=float((d_off > L).mean()),
                       p_over_L_on=float((d_on > L).mean()),
                       p_worse=float((d_on > d_off + 1e-9).mean()),
                       p_better=float((d_on < d_off - 1e-9).mean()),
                       dd_diff_se=float((d_on - d_off).std(ddof=1) / sqrt(nseed)),
                       halts=float(np.mean([x["halt"] for x in on])))
            res["cells"].append(row)
            print("E3 L=%.0f m=%.1f dd %.2f->%.2f cv %.2f->%.2f worse=%.2f" % (
                L, mg, row["dd_off"], row["dd_on"], row["cvar_off"], row["cvar_on"],
                row["p_worse"]), flush=True)
            json.dump(res, open(os.path.join(OUT, "e3.json"), "w"), indent=1)
    return res


def exp4(nseed=60):
    policies = [dict(name="nopolicy"), dict(name="fba5", fba=5), dict(name="fba15", fba=15),
                dict(name="dyntick", dyn_tick=0.8), dict(name="asym", asym=0.8),
                dict(name="breaker", limit_L=6.0), dict(name="fba15+asym", fba=15, asym=0.8),
                dict(name="fba15+breaker", fba=15, limit_L=6.0)]
    res = dict(policies=[], gamma=0.6)
    for g in (0.6, 0.9):
        for pol in policies:
            kw = {k: v for k, v in pol.items() if k != "name"}
            st = ensemble(nseed, gamma=g, tau=4, shock=90.0, **kw)
            res["policies"].append(dict(name=pol["name"], params=kw, gamma=g, **st))
            print("E4 g=%.1f %-12s dd=%.2f cv=%.2f S=%.2f coll=%.2f" % (
                g, pol["name"], st["dd"], st["cvar95"], st["S"], st["p_collapse"]), flush=True)
            json.dump(res, open(os.path.join(OUT, "e4.json"), "w"), indent=1)
    return res


def exp5(nseed=30):
    res = dict()
    res["shock_scan"] = [dict(shock=S, **ensemble(nseed, gamma=0.0, shock=float(S)))
                         for S in (10, 20, 40, 60, 80, 120, 160, 240, 320)]
    json.dump(res, open(os.path.join(OUT, "e5.json"), "w"), indent=1)
    print("E5a done", flush=True)
    res["depth_elasticity"] = [dict(v_f=v, **ensemble(nseed, gamma=0.0, v_f=v, shock=0.0))
                               for v in (0.02, 0.04, 0.08, 0.16, 0.32)]
    res["latency_depth"] = [dict(tau=t, **ensemble(nseed, gamma=0.0, tau=t, shock=0.0))
                            for t in (1, 2, 4, 8, 16, 32)]
    res["capacity"] = [dict(Q_cap=q, C=100 * q, **ensemble(nseed, gamma=0.0, Q_cap=q, shock=90.0))
                       for q in (0.4, 0.7, 1.0, 1.5, 2.5)]
    json.dump(res, open(os.path.join(OUT, "e5.json"), "w"), indent=1)
    print("E5 done", flush=True)
    return res




import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({
    "font.size": 8.2, "axes.titlesize": 8.6, "axes.labelsize": 8.2,
    "legend.fontsize": 7.0, "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.4,
})
C = ["#1f3b73", "#b3392c", "#2e7d5b", "#8a5a1b", "#5b3a86", "#1a7f8c"]


def load(name):
    return json.load(open(os.path.join(OUT, name + ".json")))


def fig1():
    d = load("e1")
    fig, ax = plt.subplots(2, 2, figsize=(5.6, 4.1))
    cells = {(c["tau"], c["Delta"]): c for c in d["cells"]}
    g = d["gammas"]
    for i, tau in enumerate((1, 2, 4, 8)):
        c = cells[(tau, 1.0)]
        y = [r["D_end"] for r in c["rows"]]
        ax[0, 0].plot(g, y, "o-", ms=2.8, color=C[i], label="$\\tau=%d$" % tau)
        ax[0, 1].plot(g, [r["dd"] for r in c["rows"]], "o-", ms=2.8, color=C[i],
                      label="$\\tau=%d$" % tau)
    ax[0, 0].set_xlabel("reflexivity gain $\\gamma$")
    ax[0, 0].set_ylabel("terminal depth $\\rho=D_{end}/D_0$")
    ax[0, 0].legend(frameon=False, ncol=2)
    ax[0, 0].set_ylim(-0.05, 1.15)
    ax[0, 0].text(0.02, 0.06, "(a)", transform=ax[0, 0].transAxes)
    ax[0, 1].set_xlabel("reflexivity gain $\\gamma$")
    ax[0, 1].set_ylabel("peak drawdown (price units)")
    ax[0, 1].legend(frameon=False, ncol=2)
    ax[0, 1].text(0.02, 0.86, "(b)", transform=ax[0, 1].transAxes)
    # critical gain against latency and tick
    taus = sorted({c["tau"] for c in d["cells"]})
    for j, D in enumerate((0.5, 1.0, 2.0)):
        y = [cells[(t, D)]["gamma_c"] for t in taus]
        ax[1, 0].plot(taus, y, "o-", ms=3.0, color=C[j], label="$\\Delta=%.1f$" % D)
    ax[1, 0].set_xscale("log", base=2)
    ax[1, 0].set_xticks(taus)
    ax[1, 0].set_xticklabels([str(t) for t in taus])
    ax[1, 0].set_xlabel("latency $\\tau$ (ticks)")
    ax[1, 0].set_ylabel("critical gain $\\gamma_c$")
    ax[1, 0].legend(frameon=False)
    ax[1, 0].text(0.02, 0.06, "(c)", transform=ax[1, 0].transAxes)
    # transition width
    for j, D in enumerate((0.5, 1.0, 2.0)):
        y = [cells[(t, D)]["width"] for t in taus]
        ax[1, 1].plot(taus, y, "o-", ms=3.0, color=C[j], label="$\\Delta=%.1f$" % D)
    ax[1, 1].set_xscale("log", base=2)
    ax[1, 1].set_xticks(taus)
    ax[1, 1].set_xticklabels([str(t) for t in taus])
    ax[1, 1].set_xlabel("latency $\\tau$ (ticks)")
    ax[1, 1].set_ylabel("width of the collapse window")
    ax[1, 1].legend(frameon=False)
    ax[1, 1].text(0.02, 0.86, "(d)", transform=ax[1, 1].transAxes)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIG, "fig1_transition.png"))
    plt.close(fig)


def fig2():
    d = load("e2")
    fig, ax = plt.subplots(1, 3, figsize=(5.6, 2.0))
    g = d["gammas"]
    for i, c in enumerate(d["cells"]):
        ax[0].plot(g, [r["D_end"] for r in c["rows"]], "o-", ms=2.4, color=C[i],
                   label="$N=%d$" % c["N"])
    ax[0].set_xlabel("$\\gamma$")
    ax[0].set_ylabel("$\\rho$")
    ax[0].legend(frameon=False, fontsize=6.2)
    ax[0].set_ylim(-0.05, 1.12)
    ax[0].text(0.03, 0.05, "(a)", transform=ax[0].transAxes)
    N = [c["N"] for c in d["cells"]]
    gc = [c["gamma_c"] for c in d["cells"]]
    ax[1].plot(N, gc, "o-", ms=3, color=C[0])
    ax[1].set_xscale("log")
    ax[1].set_xlabel("number of makers $N$")
    ax[1].set_ylabel("$\\gamma_c$")
    ax[1].text(0.05, 0.85, "(b)", transform=ax[1].transAxes)
    w = [c["width"] for c in d["cells"]]
    ax[2].plot(N, w, "s-", ms=3, color=C[1])
    ax[2].set_xscale("log")
    ax[2].set_xlabel("number of makers $N$")
    ax[2].set_ylabel("collapse width")
    ax[2].text(0.05, 0.05, "(c)", transform=ax[2].transAxes)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIG, "fig2_finite_size.png"))
    plt.close(fig)


def fig3():
    d = load("e5"); d1 = load("e1")
    fig, ax = plt.subplots(2, 2, figsize=(5.6, 3.9))
    lat = d["latency_depth"]
    t = np.array([r["tau"] for r in lat]); D = np.array([r["D0"] for r in lat])
    ax[0, 0].loglog(t, D, "o-", ms=3, color=C[0])
    s = -np.polyfit(np.log(t), np.log(D), 1)[0]
    ax[0, 0].loglog(t, D[0] * (t / t[0]) ** (-s), "k--", lw=0.8,
                    label="slope $-%.2f$" % s)
    ax[0, 0].set_xlabel("latency $\\tau$")
    ax[0, 0].set_ylabel("touch depth $D_0$")
    ax[0, 0].legend(frameon=False)
    ax[0, 0].text(0.03, 0.08, "(a)", transform=ax[0, 0].transAxes)
    ax[0, 1].plot(t, [r["sig"] for r in lat], "o-", ms=3, color=C[1])
    ax[0, 1].set_xscale("log", base=2)
    ax[0, 1].set_xticks(t); ax[0, 1].set_xticklabels([str(int(x)) for x in t])
    ax[0, 1].set_xlabel("latency $\\tau$")
    ax[0, 1].set_ylabel("per-tick volatility $\\sigma$")
    ax[0, 1].text(0.03, 0.85, "(b)", transform=ax[0, 1].transAxes)
    el = d["depth_elasticity"]
    sg = np.array([r["sig"] for r in el]); dd = np.array([r["D0"] for r in el])
    ax[1, 0].loglog(sg, dd, "o-", ms=3, color=C[2])
    e = -np.polyfit(np.log(sg), np.log(dd), 1)[0]
    ax[1, 0].loglog(sg, dd[0] * (sg / sg[0]) ** (-e), "k--", lw=0.8,
                    label="elasticity $\\varepsilon_D=%.2f$" % e)
    ax[1, 0].set_xlabel("volatility $\\sigma$")
    ax[1, 0].set_ylabel("touch depth $D_0$")
    ax[1, 0].legend(frameon=False)
    ax[1, 0].text(0.03, 0.08, "(c)", transform=ax[1, 0].transAxes)
    sc = d["shock_scan"]
    S = np.array([r["shock"] for r in sc], float); y = np.array([r["dd"] for r in sc])
    ax[1, 1].loglog(S, y, "o-", ms=3, color=C[3])
    a = np.polyfit(np.log(S), np.log(y), 1)
    ax[1, 1].loglog(S, np.exp(a[1]) * S ** a[0], "k--", lw=0.8,
                    label="slope $%.2f$" % a[0])
    ax[1, 1].set_xlabel("sell programme $S$ (lots)")
    ax[1, 1].set_ylabel("peak drawdown")
    ax[1, 1].legend(frameon=False)
    ax[1, 1].text(0.03, 0.85, "(d)", transform=ax[1, 1].transAxes)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIG, "fig3_latency_scaling.png"))
    plt.close(fig)
    return s, e, a[0]


def fig4():
    d = load("e3")
    fig, ax = plt.subplots(2, 2, figsize=(5.6, 3.9))
    cells = {(c["L"], c["magnet"]): c for c in d["cells"]}
    Ls = d["L"]; mags = d["magnet"]
    for j, m in enumerate(mags):
        ax[0, 0].plot(Ls, [cells[(L, m)]["dd_on"] for L in Ls], "o-", ms=3, color=C[j],
                      label="magnet $=%.0f$" % m)
        ax[0, 1].plot(Ls, [cells[(L, m)]["cvar_on"] for L in Ls], "o-", ms=3, color=C[j],
                      label="magnet $=%.0f$" % m)
    nb = cells[(Ls[0], mags[0])]["dd_off"]
    ax[0, 0].axhline(nb, color="k", ls="--", lw=0.9, label="no breaker")
    ax[0, 0].set_xlabel("price limit $L$ (price units)")
    ax[0, 0].set_ylabel("mean peak drawdown")
    ax[0, 0].legend(frameon=False, fontsize=6.2)
    ax[0, 0].text(0.03, 0.86, "(a)", transform=ax[0, 0].transAxes)
    ax[0, 1].axhline(cells[(Ls[0], mags[0])]["cvar_off"], color="k", ls="--", lw=0.9,
                     label="no breaker")
    ax[0, 1].set_xlabel("price limit $L$ (price units)")
    ax[0, 1].set_ylabel("CVaR$_{95}$ of drawdown")
    ax[0, 1].legend(frameon=False, fontsize=6.2)
    ax[0, 1].text(0.03, 0.86, "(b)", transform=ax[0, 1].transAxes)
    for j, m in enumerate(mags):
        ax[1, 0].plot(Ls, [cells[(L, m)]["p_worse"] for L in Ls], "o-", ms=3, color=C[j],
                      label="magnet $=%.0f$" % m)
    ax[1, 0].axhline(0.5, color="k", ls=":", lw=0.8)
    ax[1, 0].set_xlabel("price limit $L$")
    ax[1, 0].set_ylabel("share of paired runs made worse")
    ax[1, 0].legend(frameon=False, fontsize=6.2)
    ax[1, 0].set_ylim(0.3, 1.05)
    ax[1, 0].text(0.03, 0.08, "(c)", transform=ax[1, 0].transAxes)
    # example paths
    a = run(seed=3001, gamma=0.5, shock=90.0)
    b = run(seed=3001, gamma=0.5, shock=90.0, limit_L=4.0)
    tt = np.arange(len(a["m"]))
    ax[1, 1].plot(tt, a["m"], color="k", lw=1.0, label="no breaker")
    ax[1, 1].plot(tt, b["m"], color=C[1], lw=1.0, label="breaker $L=4$")
    h = b["h"]
    on = h > 0
    ax[1, 1].fill_between(tt, -9.5, 1.2, where=on, color=C[1], alpha=0.12, lw=0)
    ax[1, 1].set_xlim(0, 320); ax[1, 1].set_ylim(-9.5, 1.2)
    ax[1, 1].set_xlabel("tick after the shock")
    ax[1, 1].set_ylabel("mid price")
    ax[1, 1].legend(frameon=False, fontsize=6.4, loc="lower left")
    ax[1, 1].text(0.60, 0.86, "(d)", transform=ax[1, 1].transAxes)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIG, "fig4_breaker.png"))
    plt.close(fig)


def fig5():
    d = load("e4")
    fig, ax = plt.subplots(1, 3, figsize=(5.6, 2.1))
    names = [p["name"] for p in d["policies"] if p["gamma"] == 0.6]
    dd6 = [p["dd"] for p in d["policies"] if p["gamma"] == 0.6]
    dd9 = [p["dd"] for p in d["policies"] if p["gamma"] == 0.9]
    x = np.arange(len(names))
    ax[0].bar(x - 0.2, dd6, 0.38, color=C[0], label="$\\gamma=0.6$")
    ax[0].bar(x + 0.2, np.minimum(dd9, 20), 0.38, color=C[1], label="$\\gamma=0.9$")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(["none", "FBA\n5", "FBA\n15", "tick", "asym", "CB", "FBA+asym",
                           "FBA+CB"], fontsize=5.8)
    ax[0].set_ylabel("mean peak drawdown")
    ax[0].legend(frameon=False, fontsize=6.4)
    ax[0].text(0.03, 0.88, "(a)", transform=ax[0].transAxes)
    cv6 = [p["cvar95"] for p in d["policies"] if p["gamma"] == 0.6]
    cv9 = [p["cvar95"] for p in d["policies"] if p["gamma"] == 0.9]
    ax[1].bar(x - 0.2, cv6, 0.38, color=C[0])
    ax[1].bar(x + 0.2, np.minimum(cv9, 30), 0.38, color=C[1])
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(["none", "FBA\n5", "FBA\n15", "tick", "asym", "CB", "FBA+asym",
                           "FBA+CB"], fontsize=5.8)
    ax[1].set_ylabel("CVaR$_{95}$ of drawdown")
    ax[1].text(0.03, 0.88, "(b)", transform=ax[1].transAxes)
    S6 = [p["S"] for p in d["policies"] if p["gamma"] == 0.6]
    ax[2].scatter(S6, cv6, s=14, color=C[0])
    for i, nm in enumerate(names):
        ax[2].annotate(nm, (S6[i], cv6[i]), fontsize=5.2, xytext=(2, 2),
                       textcoords="offset points")
    ax[2].set_xscale("symlog")
    ax[2].set_xlabel("spread cost (ticks)")
    ax[2].set_ylabel("CVaR$_{95}$")
    ax[2].text(0.03, 0.88, "(c)", transform=ax[2].transAxes)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIG, "fig5_policies.png"))
    plt.close(fig)


def fig6():
    d = load("e1")
    cells = d["cells"]
    taus = sorted({c["tau"] for c in cells})
    fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.2))
    for j, D in enumerate((0.5, 1.0, 2.0)):
        row = [(c["gamma_c"]) for t in taus for c in cells if c["tau"] == t and c["Delta"] == D]
        ax[0].plot(taus, row, "o-", ms=3, color=C[j], label="$\\Delta=%.1f$" % D)
    # benchmark with measured volatility of the baseline run
    lat = load("e5")["latency_depth"]
    N, S0, mu0, T, a, H = 100, 90.0, 1.2, 20.0, 0.90, 8.0
    sig = {r["tau"]: r["sig"] for r in lat}
    pred = []
    for t in taus:
        s2 = sig.get(t, sig[max(sig)])
        pred.append(N / (2 * S0 * mu0 * T * a * s2 ** 2 * H))
    ax[0].plot(taus, pred, "k--", lw=1.0, label="benchmark (4.2)")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xticks(taus); ax[0].set_xticklabels([str(t) for t in taus])
    ax[0].set_xlabel("latency $\\tau$")
    ax[0].set_ylabel("$\\gamma_c$")
    ax[0].legend(frameon=False, fontsize=6.4)
    ax[0].set_ylim(0, 1.4)
    ax[0].text(0.04, 0.86, "(a)", transform=ax[0].transAxes)
    # depth against capacity
    cap = load("e5")["capacity"]
    Cc = [r["C"] for r in cap]; dd = [r["dd"] for r in cap]
    ax[1].plot(Cc, dd, "o-", ms=3, color=C[0])
    ax[1].set_xlabel("aggregate maker capacity $C=N\\bar Q$ (lots)")
    ax[1].set_ylabel("peak drawdown at $\\gamma=0$")
    ax[1].text(0.04, 0.86, "(b)", transform=ax[1].transAxes)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIG, "fig6_phase.png"))
    plt.close(fig)



def main():
    global OUT, FIG, RES
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "smoke":
        OUT = os.path.join(HERE, "results_smoke")
        RES = OUT
        FIG = os.path.join(HERE, "figures_smoke")
        os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
        n = 2
        for f in ("e1.json", "e2.json", "e3.json"):
            p = os.path.join(OUT, f)
            if os.path.exists(p):
                os.remove(p)
        exp1(nseed=n); exp2(nseed=n); exp3(nseed=n); exp4(nseed=n); exp5(nseed=n)
        fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
        print("smoke ok")
        return
    os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    exp1(); exp2(); exp3(); exp4(); exp5()
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print("done")


if __name__ == "__main__":
    main()
