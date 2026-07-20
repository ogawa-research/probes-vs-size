"""T02 B5 supplement: bootstrap CI for the MAE gap (best mechanism - best size
baseline), requested by the paperization review (2026-07-18, A-6 item 5 / R2
defense: turn the power confession into a quantified detectable-effect bound).

Data source = frozen b5_results.json['dataset'] (N=19; independently recomputed
4-digit match per the review). Specs fixed to the B5 winners:
  mech = mult_R_Amass_W (best composite, MAE 0.8912)
  base = base_params    (best baseline,  MAE 0.8027)

Method: per-model LOFO-CV absolute errors are computed once on the full data
(each model is predicted exactly once, when its family fold is held out), then
the paired errors are resampled two ways:
  (a) model-level paired bootstrap (resample 19 error pairs)  -- standard
  (b) family-level cluster bootstrap (resample 7 families with replacement,
      pooling their members' error pairs) -- honest to the LOFO structure,
      since errors within a family come from the same held-out fold.
Percentile CIs, B=10000, fixed seeds. Deterministic.

Run: python3 b5_bootstrap_ci.py  (writes b5_bootstrap_ci.json next to b5_results.json)
"""
import json, math, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b4_regression as b4  # noqa: E402

B5 = os.path.join(b4.DATA, 't02_b5_2026-07-15')
MECH, BASE = 'mult_R_Amass_W', 'base_params'
B_BOOT, SEED_MODEL, SEED_FAM = 10000, 20260720, 20260721


def lofo_abs_errors(data, feats):
    """per-model absolute LOFO-CV errors, keyed by model name (b4.lofo_mae logic
    with per-row errors kept instead of averaged)."""
    fams = sorted({r['family'] for r in data})
    errs = {}
    for hold in fams:
        tr = [r for r in data if r['family'] != hold]
        te = [r for r in data if r['family'] == hold]
        if not tr or not te:
            continue
        mu = {f: sum(r[f] for r in tr) / len(tr) for f in feats}
        sd = {f: (sum((r[f] - mu[f]) ** 2 for r in tr) / len(tr)) ** 0.5 or 1.0
              for f in feats}
        X = [[1.0] + [(r[f] - mu[f]) / sd[f] for f in feats] for r in tr]
        beta = b4.ols_fit(X, [r['y'] for r in tr])
        for r in te:
            pred = beta[0] + sum(beta[j + 1] * (r[feats[j]] - mu[feats[j]]) / sd[feats[j]]
                                 for j in range(len(feats)))
            errs[r['name']] = abs(pred - r['y'])
    return errs


def pctl(sorted_v, q):
    i = q * (len(sorted_v) - 1)
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return sorted_v[lo] + (sorted_v[hi] - sorted_v[lo]) * (i - lo)


def ci(diffs):
    s = sorted(diffs)
    return {'mean': sum(s) / len(s), 'p2.5': pctl(s, 0.025), 'p50': pctl(s, 0.5),
            'p97.5': pctl(s, 0.975), 'frac_le_0': sum(1 for d in s if d <= 0) / len(s)}


def main():
    data = b4.add_derived([dict(r) for r in
                           json.load(open(os.path.join(B5, 'b5_results.json')))['dataset']])
    names = [r['name'] for r in data]
    fam_of = {r['name']: r['family'] for r in data}
    e_mech = lofo_abs_errors(data, b4.MODEL_SPECS[MECH])
    e_base = lofo_abs_errors(data, b4.MODEL_SPECS[BASE])
    assert set(e_mech) == set(e_base) == set(names) and len(names) == 19

    mae_mech = sum(e_mech.values()) / len(names)
    mae_base = sum(e_base.values()) / len(names)
    diff_obs = mae_mech - mae_base
    # cross-check against the recorded suite values (must match to ~1e-9)
    rec = json.load(open(os.path.join(B5, 'b5_results.json')))['main']
    assert abs(mae_mech - rec[MECH]) < 1e-9 and abs(mae_base - rec[BASE]) < 1e-9

    pairs = [(e_mech[n], e_base[n]) for n in names]

    rng = random.Random(SEED_MODEL)
    d_model_lvl = []
    for _ in range(B_BOOT):
        samp = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        d_model_lvl.append(sum(a for a, _ in samp) / len(samp)
                           - sum(b for _, b in samp) / len(samp))

    fams = sorted({fam_of[n] for n in names})
    by_fam = {f: [(e_mech[n], e_base[n]) for n in names if fam_of[n] == f] for f in fams}
    rng = random.Random(SEED_FAM)
    d_fam_lvl = []
    for _ in range(B_BOOT):
        samp = []
        for _ in range(len(fams)):
            samp.extend(by_fam[fams[rng.randrange(len(fams))]])
        d_fam_lvl.append(sum(a for a, _ in samp) / len(samp)
                         - sum(b for _, b in samp) / len(samp))

    out = {
        'note': ('paired bootstrap CI for LOFO-CV MAE(mech) - MAE(base_params); '
                 'supplement to the immutable B4/B5 verdicts, review A-6 item 5'),
        'specs': {'mech': MECH, 'base': BASE},
        'observed': {'mae_mech': mae_mech, 'mae_base': mae_base, 'diff': diff_obs},
        'per_model_abs_err': {n: {'mech': e_mech[n], 'base': e_base[n],
                                  'family': fam_of[n]} for n in names},
        'bootstrap': {'B': B_BOOT, 'seeds': {'model': SEED_MODEL, 'family': SEED_FAM}},
        'model_level_paired': ci(d_model_lvl),
        'family_cluster': ci(d_fam_lvl),
    }
    p = os.path.join(B5, 'b5_bootstrap_ci.json')
    with open(p, 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"observed diff (mech - base) = {diff_obs:+.4f}"
          f"  [mech {mae_mech:.4f} vs base {mae_base:.4f}]")
    for k in ('model_level_paired', 'family_cluster'):
        c = out[k]
        print(f"{k:20s} mean {c['mean']:+.4f}  95% CI [{c['p2.5']:+.4f},"
              f" {c['p97.5']:+.4f}]  P(diff<=0) = {c['frac_le_0']:.3f}")
    print('written:', p)


if __name__ == '__main__':
    main()
