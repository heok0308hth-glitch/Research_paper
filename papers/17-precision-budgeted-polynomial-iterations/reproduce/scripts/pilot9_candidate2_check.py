#!/usr/bin/env python3
"""
Pilot 9 - Candidate 2 (insurance track): does verification multiplicity raise rho in a REAL validator?

The paper draft for candidate 2 rests on two claims:
  (T1) with independent verifier noise, a candidate's expected true quality under
       best-of-N is exactly rho_m * E[max_N], with rho_m = rho*sqrt(m)/sqrt(1+(m-1)rho^2);
  (T2) the budget splits between N and m and the optimal m* is single-peaked.
The known fatal precondition is that the m verifier calls must carry *independent*
noise: repeating the same deterministic evaluation gives nothing.

This script measures the whole chain on real data (no synthetic Gaussian quality):
  pool  : sklearn digits (10 classes, 64 features)
  model : 400 candidates = logistic-regression classifiers on random 40-feature
          subsets, each fitted once on a fixed 700-sample training split;
  truth : accuracy on a 500-sample gold set never used for anything else;
  verif.: accuracy on 60-sample labelled blocks drawn from a disjoint 597-sample
          pool (10 disjoint blocks).  m = number of blocks averaged.

Three regimes, to separate the theory from its precondition:
  (a) repeat      - the same block and rule m times          (no new information)
  (b) independent - m disjoint blocks, same rule             (theory's assumption)
  (c) mixed rules - m disjoint blocks, rules alternated      (realistic heterogen.)

Measured: rho_m against the sqrt(m) law, and the best-of-N frontier
(selected candidate's true accuracy vs (N, m)) at a fixed cost budget.
"""
import json
import os
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
R = {"config": {}}


def build(n_cand=400, n_train=700, n_gold=450, block=60, n_block=10, seed=0):
    d = load_digits()
    X, y = d.data / 16.0, d.target
    rs = np.random.RandomState(seed)
    idx = rs.permutation(len(y))
    itr, ig, ip = idx[:n_train], idx[n_train:n_train + n_gold], idx[n_train + n_gold:]
    Xtr, ytr = X[itr], y[itr]
    Xg, yg = X[ig], y[ig]
    pool = ip[:n_block * block]
    Xp, yp = X[pool], y[pool]
    acc = np.zeros((n_cand, n_block))
    nll = np.zeros((n_cand, n_block))
    feats_all = []
    for i in range(n_cand):
        feats = rs.choice(X.shape[1], 40, replace=False)
        feats_all.append(feats)
        clf = LogisticRegression(max_iter=200, C=1.0)
        clf.fit(Xtr[:, feats], ytr)
        for b in range(n_block):
            sl = slice(b * block, (b + 1) * block)
            P = clf.predict_proba(Xp[sl][:, feats])
            pred = P.argmax(1)
            acc[i, b] = float((pred == yp[sl]).mean())
            nll[i, b] = float(-np.log(P[np.arange(block), yp[sl]] + 1e-12).mean())
    return X, y, itr, ig, ip, Xtr, ytr, Xg, yg, Xp, yp, acc, nll, np.array(feats_all)


def gold_acc(X, y, itr, Xg, yg, feats_all):
    g = np.zeros(len(feats_all))
    for i, feats in enumerate(feats_all):
        clf = LogisticRegression(max_iter=200, C=1.0)
        clf.fit(X[itr][:, feats], y[itr])
        g[i] = float((clf.predict(Xg[:, feats]) == yg).mean())
    return g


def z(a):
    a = np.asarray(a, float)
    s = a.std()
    return (a - a.mean()) / (s if s > 1e-12 else 1.0)


def rho(a, b):
    return float(np.corrcoef(z(a), z(b))[0, 1])


def law(r1, m):
    return r1 * np.sqrt(m) / np.sqrt(1 + (m - 1) * r1 ** 2)


def run_regime(name, n_train, n_cand=400, n_gold=450, block=60, n_block=10):
    rs = np.random.RandomState(11)
    X, y, itr, ig, ip, Xtr, ytr, Xg, yg, Xp, yp, acc, nll, feats_all = build(
        n_cand=n_cand, n_train=n_train, n_gold=n_gold, block=block, n_block=n_block)
    gold = gold_acc(X, y, itr, Xg, yg, feats_all)
    r = rho(acc[:, 0], gold)
    r_nll = rho(-nll[:, 0], gold)
    print(f"\n  [{name}] train={n_train}: gold {gold.mean():.3f}+-{gold.std():.3f} "
          f"[{gold.min():.3f},{gold.max():.3f}]  rho_1(acc)={r:.3f}  rho_1(-nll)={r_nll:.3f}")
    R["config"][name] = {"n_train": n_train, "rho_1": r, "rho_1_nll": r_nll,
                         "gold_mean": float(gold.mean()), "gold_sd": float(gold.std()),
                         "gold_range": [float(gold.min()), float(gold.max())]}
    out = {"rho_m": {"a_repeat": {}, "b_indep": {}, "c_mixed": {}, "law": {}},
           "frontier": {}, "budget_best": {}, "oracle": {}}
    ms = (1, 2, 3, 5, 10)
    for m in ms:
        sb = acc[:, :m].mean(1)
        cols = [z(acc[:, b] if b % 2 == 0 else -nll[:, b]) for b in range(m)]
        sc = np.mean(np.stack(cols), axis=0)
        out["rho_m"]["a_repeat"][str(m)] = rho(acc[:, 0], gold)
        out["rho_m"]["b_indep"][str(m)] = rho(sb, gold)
        out["rho_m"]["c_mixed"][str(m)] = rho(sc, gold)
        out["rho_m"]["law"][str(m)] = float(law(r, m))
    print("    rho_m:  m   repeat  indep  indep/law   mixed   mixed/law")
    for m in ms:
        print(f"           {m:2d}  {out['rho_m']['a_repeat'][str(m)]:.3f}  "
              f"{out['rho_m']['b_indep'][str(m)]:.3f}   {out['rho_m']['b_indep'][str(m)]/out['rho_m']['law'][str(m)]:.2f}      "
              f"{out['rho_m']['c_mixed'][str(m)]:.3f}   {out['rho_m']['c_mixed'][str(m)]/out['rho_m']['law'][str(m)]:.2f}")
    n_draw = 400
    Ns = (4, 8, 16, 32, 64, 128)
    for c in (0.02, 0.1, 0.3):
        for m in ms:
            row = {}
            for N in Ns:
                vals = []
                for _ in range(n_draw):
                    pick = rs.choice(acc.shape[0], N, replace=False)
                    s = acc[np.ix_(pick, np.arange(m))].mean(1)
                    vals.append(gold[pick][int(np.argmax(s))])
                row[N] = float(np.mean(vals))
            out["frontier"][f"c{c}_m{m}"] = row
    for N in Ns:
        vals = [gold[rs.choice(acc.shape[0], N, replace=False)].max() for _ in range(n_draw)]
        out["oracle"][N] = float(np.mean(vals))
    print("    best (m,N) at equal sampling budget; gain over m=1; share of the oracle gap closed")
    print(f"      {'c':>5s} {'B':>6s} {'m*':>3s} {'N':>4s} {'Q':>7s} {'Q(m=1)':>7s} {'gain':>6s} {'oracle':>7s} {'gap closed':>10s}")
    for c in (0.02, 0.1, 0.3):
        for N1 in Ns:
            B = N1 * (1 + c)
            best = None
            for m in ms:
                N = int(B / (1 + m * c))
                cand = [n for n in Ns if n <= N]
                if not cand:
                    continue
                Nn = max(cand)
                q = out["frontier"][f"c{c}_m{m}"][Nn]
                if best is None or q > best[0]:
                    best = (q, m, Nn)
            q1 = out["frontier"][f"c{c}_m1"][N1]
            o1 = out["oracle"][N1]
            closed = (best[0] - q1) / (o1 - q1) if o1 > q1 else float("nan")
            out["budget_best"][f"c{c}_B{B}"] = {"m": best[1], "N": best[2], "Q": best[0],
                                                "Q_m1": q1, "gain": best[0] / q1 - 1,
                                                "oracle_m1": o1, "gap_closed": closed}
            print(f"      {c:5.2f} {B:6.1f} {best[1]:3d} {best[2]:4d} {best[0]:7.4f} {q1:7.4f} "
                  f"{best[0]/q1-1:5.1%} {o1:7.4f} {closed:9.1%}")
    est = []
    for _ in range(20):
        gsel = rs.choice(acc.shape[0], 200, replace=False)
        est.append(rho(acc[gsel, 0], gold[gsel]))
    out["gold_est"] = {"mean": float(np.mean(est)), "sd": float(np.std(est)), "true": r,
                       "bias": float(np.mean(est) - r)}
    print(f"    rho from a 200-candidate gold set: {np.mean(est):.3f}+-{np.std(est):.3f} (true {r:.3f})")
    R[name] = out


def main():
    print("=" * 100)
    print("Pilot 9 - verification multiplicity on real data (digits, 400 candidate classifiers)")
    print("=" * 100)
    run_regime("easy_train700", 700)
    run_regime("hard_train120", 120)
    out = os.path.join(HERE, "pilot9_results.json")
    with open(out, "w") as f:
        json.dump(R, f, indent=1, default=float)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
