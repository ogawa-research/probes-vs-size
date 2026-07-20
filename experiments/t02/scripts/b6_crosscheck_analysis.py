"""T02 B6: crosscheck-driven supplementary analyses (researcher, 2026-07-20).

Source of the repair list: the internal crosscheck review (2026-07-20, available on request), sec.3.2
(items 1-4 of the analysis list). All analyses here are post hoc relative to the
frozen B4/B5 protocol and are labeled as such where they enter the paper; the
B4/B5 verdicts are immutable and are not recomputed here.

  1. N=19 correlations on the B5 dataset (replaces the stale N=15 values that
     had been transcribed from the B4 judgement; crosscheck M2/5').
  2. Joint size+factor LOFO models (the direct "incremental over size" test the
     paper's claim language implies; crosscheck M3). Multiplicity disclosed:
     five joint specs, only size+R improves on size alone, permutation p is
     selection-uncorrected.
  3. Incremental paired bootstrap CI for MAE(size+R) - MAE(size), model-level
     and family-cluster (same machinery/seeds convention as b5_bootstrap_ci).
  4. Uniform-85.6 Gemma threshold sensitivity rerun at N=19 (was B4/N=15 only).
  5. Censored-imputation sensitivity (imputed at 1024/2048/4096).

Run: python3 b6_crosscheck_analysis.py   (deterministic; writes
data/t02_b6_2026-07-20/b6_results.json)
"""
import json, math, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b4_regression as b4  # noqa: E402
from b5_bootstrap_ci import lofo_abs_errors, ci  # noqa: E402

B5 = os.path.join(b4.DATA, 't02_b5_2026-07-15')
OUT_DIR = os.path.join(b4.DATA, 't02_b6_2026-07-20')
B_BOOT, SEED_MODEL, SEED_FAM = 10000, 20260722, 20260723

JOINT_SPECS = {
    'joint_params_R': ['log10_params', 'R'],
    'joint_params_Amass': ['log10_params', 'A_mass'],
    'joint_params_Agini': ['log10_params', 'A_gini'],
    'joint_params_W': ['log10_params', 'W_erank_all'],
    'joint_params_multRAW': ['log10_params', 'mult_RAW'],
}

# uniform-85.6 Gemma effective lengths (transcribed from b4_regression.main()'s
# local G856; contiguous rule per t02_b3_judgement_2026-07-15.md sec.3.2 --
# 12b is non-monotone under 85.6 -> censored <4K)
G856 = {'gemma-3-270m': 2048, 'gemma-3-1b': 2048, 'gemma-3-4b': 4096,
        'gemma-3-12b': 2048, 'gemma-3-27b': 4096}


def corr(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = [dict(r) for r in
            json.load(open(os.path.join(B5, 'b5_results.json')))['dataset']]
    assert len(rows) == 19
    data = b4.add_derived([dict(r) for r in rows])
    out = {'note': ('post-hoc supplementary analyses per '
                    'the internal crosscheck review (2026-07-20) sec.3.2; '
                    'B4/B5 verdicts immutable')}

    # ---- 1. N=19 correlations (paper source of record for the correlation
    #         paragraphs; replaces B4/N=15 transcriptions)
    lp = [r['log10_params'] for r in data]
    y = [r['y'] for r in data]
    cors = {}
    for f in ('A_mass', 'W_erank_all', 'R', 'A_gini'):
        v = [r[f] for r in data]
        cors[f'params_vs_{f}'] = corr(lp, v)
        cors[f'y_vs_{f}'] = corr(y, v)
    cors['params_vs_y'] = corr(lp, y)
    out['correlations_N19'] = cors
    print('== 1. correlations (N=19) ==')
    for k, v in cors.items():
        print(f'  {k:24s} {v:+.3f}')

    # ---- 2. joint size+factor LOFO (direct incremental test; post hoc)
    base_mae = b4.lofo_mae(data, ['log10_params'])
    joint = {k: b4.lofo_mae(data, f) for k, f in JOINT_SPECS.items()}
    best_j = min(joint.items(), key=lambda kv: kv[1])
    diff_obs = best_j[1] - base_mae
    p_best = b4.permutation_p_diff(rows, JOINT_SPECS[best_j[0]],
                                   ['log10_params'], diff_obs)
    out['joint_N19'] = {
        'base_params': base_mae, **joint,
        'best_joint': best_j[0], 'best_joint_minus_base': diff_obs,
        'perm_p_best_joint_vs_base': p_best,
        'multiplicity_note': ('5 joint specs tried post hoc; only size+R is in '
                              'the improving direction; p is selection-uncorrected'),
    }
    print('\n== 2. joint size+factor LOFO (N=19) ==')
    print(f'  base_params          {base_mae:.4f}')
    for k, v in sorted(joint.items(), key=lambda kv: kv[1]):
        print(f'  {k:20s} {v:.4f}')
    print(f'  best={best_j[0]} diff={diff_obs:+.4f} perm p={p_best:.4f}')

    # selection-corrected permutation (added 2026-07-20 after the exposed
    # crosscheck pointed out that p_best above is selection-uncorrected):
    # within each permutation, reselect the best of the five joint specs, so
    # the null distribution reflects the same selection freedom used on the
    # observed data. Statistic = min over specs of (joint MAE - size MAE).
    import random as _random
    obs_min = min(joint.values()) - base_mae
    rng = _random.Random(20260716)
    ys = [r['y'] for r in rows]
    dperm = b4.add_derived([dict(r) for r in rows])
    count = 0
    for _ in range(1000):
        perm = ys[:]
        rng.shuffle(perm)
        for r, yy in zip(dperm, perm):
            r['y_perm'] = yy
        mb = b4.lofo_mae(dperm, ['log10_params'], y_key='y_perm')
        mmin = min(b4.lofo_mae(dperm, f, y_key='y_perm')
                   for f in JOINT_SPECS.values()) - mb
        if mmin <= obs_min:
            count += 1
    p_corrected = (count + 1) / 1001
    out['joint_N19']['obs_min_joint_minus_base'] = obs_min
    out['joint_N19']['selection_corrected_perm_p'] = p_corrected
    out['joint_N19']['selection_corrected_note'] = (
        'statistic = min over the 5 specs of (joint - size) LOFO MAE, best spec '
        'reselected within each of 1000 permutations, seed 20260716; the '
        'per-spec p above is selection-uncorrected (fixed-spec diagnostic)')
    print(f'  selection-corrected perm p (best-of-5 reselected) = {p_corrected:.4f}')

    # censored-excluded joint check (N=16)
    sub = [r for r in rows if not r['censored']]
    dsub = b4.add_derived([dict(r) for r in sub])
    mj16 = b4.lofo_mae(dsub, JOINT_SPECS['joint_params_R'])
    mb16 = b4.lofo_mae(dsub, ['log10_params'])
    p16 = b4.permutation_p_diff(sub, JOINT_SPECS['joint_params_R'],
                                ['log10_params'], mj16 - mb16)
    out['joint_N16_censor_excluded'] = {
        'joint_params_R': mj16, 'base_params': mb16,
        'diff': mj16 - mb16, 'perm_p': p16}
    print(f'  N=16: size+R {mj16:.4f} vs size {mb16:.4f} '
          f'(diff {mj16-mb16:+.4f}, p={p16:.4f})')

    # ---- 3. incremental paired bootstrap (size+R vs size)
    e_joint = lofo_abs_errors(data, JOINT_SPECS['joint_params_R'])
    e_base = lofo_abs_errors(data, ['log10_params'])
    names = [r['name'] for r in data]
    fam_of = {r['name']: r['family'] for r in data}
    pairs = [(e_joint[n], e_base[n]) for n in names]

    rng = random.Random(SEED_MODEL)
    d_model = []
    for _ in range(B_BOOT):
        s = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        d_model.append(sum(a for a, _ in s) / len(s)
                       - sum(b for _, b in s) / len(s))
    fams = sorted(set(fam_of.values()))
    by_fam = {f: [(e_joint[n], e_base[n]) for n in names if fam_of[n] == f]
              for f in fams}
    rng = random.Random(SEED_FAM)
    d_fam = []
    for _ in range(B_BOOT):
        s = []
        for _ in range(len(fams)):
            s.extend(by_fam[fams[rng.randrange(len(fams))]])
        d_fam.append(sum(a for a, _ in s) / len(s)
                     - sum(b for _, b in s) / len(s))
    out['incremental_bootstrap'] = {
        'quantity': 'MAE(size+R) - MAE(size), per-model paired LOFO errors',
        'B': B_BOOT, 'seeds': {'model': SEED_MODEL, 'family': SEED_FAM},
        'observed': sum(a for a, _ in pairs) / 19 - sum(b for _, b in pairs) / 19,
        'model_level': ci(d_model), 'family_cluster': ci(d_fam),
    }
    print('\n== 3. incremental bootstrap (size+R - size) ==')
    for k in ('model_level', 'family_cluster'):
        c = out['incremental_bootstrap'][k]
        print(f'  {k:14s} mean {c["mean"]:+.4f} 95% CI '
              f'[{c["p2.5"]:+.4f}, {c["p97.5"]:+.4f}] P(<=0)={c["frac_le_0"]:.3f}')

    # ---- 4. uniform-85.6 Gemma sensitivity at N=19 (b4.G856 mapping)
    rows856 = []
    for r in rows:
        r2 = dict(r)
        if r2["name"] in G856:
            r2["eff"] = G856[r2["name"]]
            r2['y'] = math.log2(r2['eff'])
        rows856.append(r2)
    res856 = b4.run_suite(rows856, 'b6_g856_n19')
    out['sens_gemma_thr85.6_N19'] = res856
    print('\n== 4. uniform-85.6 sensitivity (N=19), top 6 ==')
    for k, v in sorted(res856.items(), key=lambda kv: kv[1])[:6]:
        print(f'  {k:20s} {v:.4f}')

    # ---- 5. censored-imputation sensitivity (1024 / 2048 / 4096)
    imp = {}
    for val in (1024, 2048, 4096):
        rws = []
        for r in rows:
            r2 = dict(r)
            if r2['censored']:
                r2['eff'] = val
                r2['y'] = math.log2(val)
            rws.append(r2)
        res = b4.run_suite(rws, f'b6_imp{val}')
        top = sorted(res.items(), key=lambda kv: kv[1])[:3]
        imp[str(val)] = {'top3': top,
                         'base_params': res['base_params'],
                         'mult_R_Amass_W': res['mult_R_Amass_W'],
                         'size_still_best_overall': top[0][0] == 'base_params'}
        print(f'  imputed={val}: top3 = {[(k, round(v,4)) for k, v in top]}')
    out['sens_censored_imputation'] = imp

    with open(os.path.join(OUT_DIR, 'b6_results.json'), 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print('\nwritten:', os.path.join(OUT_DIR, 'b6_results.json'))


if __name__ == '__main__':
    main()
