#!/usr/bin/env python3
"""T02 Part B — B4 regression (researcher, 2026-07-15).

Head-to-head: do short-context mechanism factors (R, A, W) predict RULER
effective context length better than single factors and no-mechanism baselines?

Protocol per data/t02_partB_design_2026-07-11.md §4 (pre-registered):
  - y = log2(effective length), effective from threshold method (B3 judgement +
    public-row recalc, contiguous-exceedance rule).
  - censored "<4K" imputed at 2048 (main) / excluded (sensitivity). No ">128K" points.
  - predictors standardized (train-fold statistics only), <=3 simultaneous.
  - leave-one-family-out CV, metric = held-out MAE in log2 space.
  - permutation test: shuffle y 1000x, p = P(perm CV MAE <= observed).

Run from repo root:  python3 experiments/t02/scripts/b4_regression.py
Outputs: data/t02_b4_2026-07-15/b4_results.json
"""
import json, math, os, random, sys

DATA = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')
B1B2 = os.path.join(DATA, 't02_partB_b1b2_2026-07-11')
ERANK_LADDER = os.path.join(DATA, 't02_erank_pr_2026-07-11')
PARTA = os.path.join(DATA, 't02_partA_2026-07-10')
OUT_DIR = os.path.join(DATA, 't02_b4_2026-07-15')

# ---------------------------------------------------------------- dataset spec
# effective: B3 judgement (own, answer_prefix restored) / public recalc
# (contiguous >= threshold; Qwen3 public rows author-reported, threshold 85.6;
#  cross rows NVIDIA-measured, threshold 85.6; own Gemma-3 -pt rows 79.4).
# claimed: model cards / RULER table. params in B. d_model from factor JSONs.
MODELS = [
    # name            family    eff_tokens censored claimed  params
    ('qwen3-0.6b',    'qwen3',   2048,  True,   32768,   0.6),
    ('qwen3-1.7b',    'qwen3',   8192,  False,  32768,   1.7),
    ('qwen3-4b',      'qwen3',  32768,  False, 131072,   4.0),
    ('qwen3-8b',      'qwen3',  32768,  False, 131072,   8.0),
    ('qwen3-14b',     'qwen3',  65536,  False, 131072,  14.0),
    ('gemma-3-270m',  'gemma',   2048,  True,   32768,   0.27),
    ('gemma-3-1b',    'gemma',   2048,  True,   32768,   1.0),
    ('gemma-3-4b',    'gemma',  16384,  False, 131072,   4.0),
    ('gemma-3-12b',   'gemma',  65536,  False, 131072,  12.0),
    ('gemma-3-27b',   'gemma',  65536,  False, 131072,  27.0),
    ('llama-3.1-8b',  'llama',  32768,  False, 131072,   8.0),
    ('mistral-7b',    'mistral',16384,  False,  32768,   7.0),
    ('mistral-nemo',  'mistral',16384,  False, 131072,  12.0),
    ('yi-34b',        'yi',     32768,  False, 204800,  34.0),
    ('phi3-mini',     'phi',    32768,  False, 131072,   3.8),
]
LADDER = {'qwen3-0.6b','qwen3-1.7b','qwen3-4b','qwen3-8b','qwen3-14b',
          'gemma-3-270m','gemma-3-1b','gemma-3-4b','gemma-3-12b','gemma-3-27b'}

# rule bands for ladder (Part A band_final; 1.7b reused main band 14-22;
# qwen3-14b rule band broken -> fallback [18-30] used ONLY in band sensitivity, disclosed)
LADDER_BANDS = {
    'qwen3-0.6b': list(range(14,23)), 'qwen3-1.7b': list(range(14,23)),
    'qwen3-4b': list(range(16,27)),   'qwen3-8b': list(range(14,25)),
    'qwen3-14b': list(range(18,31)),  # fallback band (depth transfer), disclosed
    'gemma-3-270m': list(range(13,17)), 'gemma-3-1b': list(range(17,24)),
    'gemma-3-4b': list(range(17,28)), 'gemma-3-12b': list(range(23,38)),
    'gemma-3-27b': list(range(29,49)),
}
BAND_HEALTHY = {m for m in LADDER if m != 'qwen3-14b'} | {'yi-34b'}

def jload(p):
    with open(p) as f: return json.load(f)

def r_factor(name):
    """R over layers finite at every length. For full-attention models this is
    all layers (== recorded R). For Gemma-3 the B4-era (v1) recorded H is NaN on
    SWA layers once seq_len exceeds the sliding window. Root cause (final, B5
    round-3 diagnosis -- supersedes the round-2 note that shipped here): the
    CLIP_FLOOR clamp (1e-45) silently underflowed to exactly 0.0 under bfloat16,
    disabling the log(0) guard, so exact-zero attention entries outside the
    window produced 0*log(0)=NaN; clip_count==0 because the clamp was inactive,
    not because 0*log0 was ruled out. Fixed in the B5 v2 probe (CLIP_FLOOR=1e-38
    + dtype assert; see b5_judgement sec.6 and the paper's App on measurement
    history). Here, only the global-attention layers (every 6th) survive
    -> R_global, disclosed; B5 replaces Gemma R with the v2 measurement."""
    d = jload(os.path.join(B1B2, f'r_delta_{name}.json'))
    pls = d['neutral']['per_length']
    lengths = sorted(pls, key=int)
    ok = None
    for L in lengths:
        fin = {k for k, v in pls[L]['per_layer'].items()
               if math.isfinite(v['H_mean'])}
        ok = fin if ok is None else ok & fin
    H = [sum(pls[L]['per_layer'][k]['H_mean'] for k in ok) / len(ok)
         for L in lengths]
    x = [math.log(int(L)) for L in lengths]
    mx, my = sum(x) / len(x), sum(H) / len(H)
    slope = (sum((xi - mx) * (hi - my) for xi, hi in zip(x, H))
             / sum((xi - mx) ** 2 for xi in x))
    R = 1.0 - slope
    rec = d['neutral']['R']['R']
    if math.isfinite(rec):
        assert abs(R - rec) < 1e-9, (name, R, rec)  # full-attn: must reproduce
    return R, d['d_model'], len(ok), d['n_layers']

def a_factor(name):
    d = jload(os.path.join(B1B2, f'a_retrieval_{name}.json'))
    ag = d['standard']['aggregate']
    return ag['top_5pct_s_h_sum'], ag['gini_coefficient']

def erank_layers(name):
    p = os.path.join(ERANK_LADDER if name in LADDER else B1B2, f'erank_pr_{name}.json')
    d = jload(p)
    pl = d['neutral']['per_layer']
    return {int(k): v['value_erank'] for k, v in pl.items()}, \
           {int(k): v['value_ner'] for k, v in pl.items()}

def cross_band(name):
    d = jload(os.path.join(B1B2, f'cross_capacity_{name}.json'))
    return d['band']

def kstar64(cap_dict):
    """stationary mean (positions 21-80) of count at k=64, averaged over seeds."""
    vals = []
    for sv in cap_dict.values():
        mb = sv.get('main_band') or sv.get('bands', {}).get('main_band')
        cc = mb['counts_at_k_by_position']
        vals.append(sum(row[63] for row in cc[20:80]) / 60.0)
    return sum(vals) / len(vals), len(vals)

KJ_DOCUMENTED = {'qwen3-1.7b': 8.39}  # Part A final judgement table (E1 full
# grid file has a different schema; value cross-checked there)

def ladder_kJ(name):
    for pat in (f'partA_capacity_{name}.json', f'partA_supplement_{name}.json'):
        p = os.path.join(PARTA, pat)
        if os.path.exists(p):
            d = jload(p)
            if 'capacity' in d:
                return kstar64(d['capacity'])
    if name in KJ_DOCUMENTED:
        return KJ_DOCUMENTED[name], -1
    return None, 0

# ---------------------------------------------------------------- assemble
rows = []
for name, fam, eff, cens, claimed, params in MODELS:
    R, d_model, n_r_layers, n_layers = r_factor(name)
    A_mass, A_gini = a_factor(name)
    er, ner = erank_layers(name)
    W_erank_all = sum(er.values()) / len(er)
    W_ner_all = sum(ner.values()) / len(ner)
    band = LADDER_BANDS.get(name)
    if band is None and os.path.exists(os.path.join(B1B2, f'cross_capacity_{name}.json')):
        band = cross_band(name)
    W_erank_band = (sum(er[l] for l in band if l in er) /
                    len([l for l in band if l in er])) if band else None
    kJ = ladder_kJ(name)[0] if name in LADDER else None
    rows.append(dict(name=name, family=fam, eff=eff, censored=cens,
                     y=math.log2(eff), claimed=claimed, params=params, d_model=d_model,
                     R=R, R_n_layers_used=n_r_layers, n_layers=n_layers,
                     A_mass=A_mass, A_gini=A_gini,
                     W_erank_all=W_erank_all, W_ner_all=W_ner_all,
                     W_erank_band=W_erank_band, W_kJ=kJ,
                     band_healthy=name in BAND_HEALTHY))

# ---------------------------------------------------------------- OLS / CV
def ols_fit(X, y):
    """X list of rows (with 1s prepended here); returns beta via normal equations."""
    n, p = len(X), len(X[0])
    XtX = [[sum(X[i][a]*X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    Xty = [sum(X[i][a]*y[i] for i in range(n)) for a in range(p)]
    # gaussian elimination with ridge epsilon for near-singular safety
    for a in range(p): XtX[a][a] += 1e-9
    M = [row[:] + [Xty[a]] for a, row in enumerate(XtX)]
    for col in range(p):
        piv = max(range(col, p), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        d = M[col][col]
        M[col] = [v/d for v in M[col]]
        for r in range(p):
            if r != col and M[r][col] != 0:
                f = M[r][col]
                M[r] = [vr - f*vc for vr, vc in zip(M[r], M[col])]
    return [M[a][p] for a in range(p)]

def lofo_mae(data, feats, y_key='y'):
    """leave-one-family-out CV MAE; standardize features on train fold."""
    fams = sorted({r['family'] for r in data})
    abs_err, n_pred = 0.0, 0
    for hold in fams:
        tr = [r for r in data if r['family'] != hold]
        te = [r for r in data if r['family'] == hold]
        if not tr or not te: continue
        mu = {f: sum(r[f] for r in tr)/len(tr) for f in feats}
        sd = {f: (sum((r[f]-mu[f])**2 for r in tr)/len(tr))**0.5 or 1.0 for f in feats}
        X = [[1.0] + [(r[f]-mu[f])/sd[f] for f in feats] for r in tr]
        beta = ols_fit(X, [r[y_key] for r in tr])
        for r in te:
            pred = beta[0] + sum(beta[j+1]*(r[feats[j]]-mu[feats[j]])/sd[feats[j]]
                                 for j in range(len(feats)))
            abs_err += abs(pred - r[y_key]); n_pred += 1
    return abs_err / n_pred

def add_derived(data):
    for r in data:
        r['log2_claimed'] = math.log2(r['claimed'])
        r['log10_params'] = math.log10(r['params'])
        # multiplicative composite: log(R * A_mass * W_erank_all) (all > 0)
        r['mult_RAW'] = (math.log(max(r['R'], 1e-9)) + math.log(max(r['A_mass'], 1e-9))
                         + math.log(max(r['W_erank_all'], 1e-9)))
        r['mult_RAgW'] = (math.log(max(r['R'], 1e-9)) + math.log(max(r['A_gini'], 1e-9))
                          + math.log(max(r['W_erank_all'], 1e-9)))
    return data

MODEL_SPECS = {
    'single_R': ['R'], 'single_A_mass': ['A_mass'], 'single_A_gini': ['A_gini'],
    'single_W_erank': ['W_erank_all'],
    'add_R_Amass_W': ['R', 'A_mass', 'W_erank_all'],
    'add_R_Agini_W': ['R', 'A_gini', 'W_erank_all'],
    'mult_R_Amass_W': ['mult_RAW'],
    'mult_R_Agini_W': ['mult_RAgW'],
    'base_claimed': ['log2_claimed'], 'base_params': ['log10_params'],
    'base_dmodel': ['d_model'],
}

def run_suite(data, label):
    data = add_derived([dict(r) for r in data])
    res = {name: lofo_mae(data, feats) for name, feats in MODEL_SPECS.items()}
    return res

def permutation_p(data, feats_target, mae_obs, n_perm=1000, seed=20260715):
    data = add_derived([dict(r) for r in data])
    rng = random.Random(seed)
    ys = [r['y'] for r in data]
    count = 0
    for _ in range(n_perm):
        perm = ys[:]; rng.shuffle(perm)
        for r, yy in zip(data, perm): r['y_perm'] = yy
        if lofo_mae(data, feats_target, y_key='y_perm') <= mae_obs: count += 1
    return (count + 1) / (n_perm + 1)

def permutation_p_diff(data, feats_a, feats_b, diff_obs, n_perm=1000, seed=20260716):
    """paired permutation on MAE(a) - MAE(b) under shuffled y."""
    data = add_derived([dict(r) for r in data])
    rng = random.Random(seed)
    ys = [r['y'] for r in data]
    count = 0
    for _ in range(n_perm):
        perm = ys[:]; rng.shuffle(perm)
        for r, yy in zip(data, perm): r['y_perm'] = yy
        d = (lofo_mae(data, feats_a, y_key='y_perm')
             - lofo_mae(data, feats_b, y_key='y_perm'))
        if d <= diff_obs: count += 1
    return (count + 1) / (n_perm + 1)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {'dataset': rows}
    print(f"N = {len(rows)}, families = {sorted({r['family'] for r in rows})}")
    for r in rows:
        print(f"  {r['name']:14s} {r['family']:8s} eff={r['eff']:>6d}{'*' if r['censored'] else ' '}"
              f" R={r['R']:.3f} A_mass={r['A_mass']:.2f} A_gini={r['A_gini']:.3f}"
              f" W_er_all={r['W_erank_all']:.1f} kJ={r['W_kJ'] if r['W_kJ'] is None else round(r['W_kJ'],2)}")

    # main analysis (N=15, censored imputed 2K)
    main_res = run_suite(rows, 'main')
    out['main'] = main_res
    print('\n== main (N=15, LOFO-CV MAE in log2 units) ==')
    for k, v in sorted(main_res.items(), key=lambda kv: kv[1]):
        print(f"  {k:18s} {v:.4f}")

    best_single = min((v, k) for k, v in main_res.items() if k.startswith('single_'))
    best_comp = min((v, k) for k, v in main_res.items()
                    if k.startswith(('add_', 'mult_')))
    best_base = min((v, k) for k, v in main_res.items() if k.startswith('base_'))
    out['comparison'] = {
        'best_single': {'model': best_single[1], 'mae': best_single[0]},
        'best_composite': {'model': best_comp[1], 'mae': best_comp[0]},
        'best_baseline': {'model': best_base[1], 'mae': best_base[0]},
        'composite_beats_single': best_comp[0] < best_single[0],
        'mechanism_beats_baseline': min(best_comp[0], best_single[0]) < best_base[0],
    }

    # permutation tests on the pre-registered comparisons
    p_comp = permutation_p(rows, MODEL_SPECS[best_comp[1]], best_comp[0])
    p_single = permutation_p(rows, MODEL_SPECS[best_single[1]], best_single[0])
    p_diff = permutation_p_diff(rows, MODEL_SPECS[best_comp[1]],
                                MODEL_SPECS[best_single[1]],
                                best_comp[0] - best_single[0])
    p_vs_base = permutation_p_diff(rows, MODEL_SPECS[best_comp[1]],
                                   MODEL_SPECS[best_base[1]],
                                   best_comp[0] - best_base[0])
    out['permutation'] = {'best_composite_p': p_comp, 'best_single_p': p_single,
                          'diff_composite_minus_single_p': p_diff,
                          'diff_composite_minus_baseline_p': p_vs_base,
                          'n_perm': 1000}
    print(f"paired diff (comp-single) p={p_diff:.4f}; (comp-baseline) p={p_vs_base:.4f}")
    print(f"\nbest composite = {best_comp[1]} MAE {best_comp[0]:.4f} perm p={p_comp:.4f}")
    print(f"best single    = {best_single[1]} MAE {best_single[0]:.4f} perm p={p_single:.4f}")
    print(f"best baseline  = {best_base[1]} MAE {best_base[0]:.4f}")

    # sensitivity: censored excluded (N=12)
    sub = [r for r in rows if not r['censored']]
    out['sens_censor_excluded'] = run_suite(sub, 'nocens')
    print('\n== sensitivity: censored 3 pts excluded (N=12) ==')
    for k, v in sorted(out['sens_censor_excluded'].items(), key=lambda kv: kv[1])[:6]:
        print(f"  {k:18s} {v:.4f}")

    # sensitivity: W = NER (normalized erank)
    ner_specs = {'single_W_ner': ['W_ner_all'],
                 'add_R_Amass_Wner': ['R', 'A_mass', 'W_ner_all']}
    d2 = add_derived([dict(r) for r in rows])
    out['sens_W_ner'] = {k: lofo_mae(d2, f) for k, f in ner_specs.items()}

    # sensitivity: band-mean erank on healthy-band subset (N=10: ladder-14b +yi)
    subb = [r for r in rows if r['band_healthy'] and r['W_erank_band'] is not None]
    if len({r['family'] for r in subb}) >= 3:
        d3 = add_derived([dict(r) for r in subb])
        out['sens_W_band'] = {
            'N': len(subb),
            'single_W_erank_band': lofo_mae(d3, ['W_erank_band']),
            'single_W_erank_all': lofo_mae(d3, ['W_erank_all']),
        }

    # sensitivity: ladder-only with W = k*J (N=9, 14b excluded, 2 families)
    subk = [r for r in rows if r['W_kJ'] is not None and r['name'] != 'qwen3-14b']
    d4 = add_derived([dict(r) for r in subk])
    out['sens_ladder_kJ'] = {
        'N': len(subk), 'note': '2 families only — LOFO = leave-one-series-out',
        'single_W_kJ': lofo_mae(d4, ['W_kJ']),
        'single_W_erank_all': lofo_mae(d4, ['W_erank_all']),
        'single_R': lofo_mae(d4, ['R']),
    }

    # sensitivity: Gemma-3 judged at aligned threshold 85.6 (threshold-mix disclosure)
    G856 = {'gemma-3-270m': 2048, 'gemma-3-1b': 2048, 'gemma-3-4b': 4096,
            'gemma-3-12b': 2048, 'gemma-3-27b': 4096}  # contiguous rule; 12b non-monotone -> <4K
    rows856 = []
    for r in rows:
        r2 = dict(r)
        if r2['name'] in G856:
            r2['eff'] = G856[r2['name']]; r2['y'] = math.log2(r2['eff'])
        rows856.append(r2)
    out['sens_gemma_thr85.6'] = run_suite(rows856, 'g856')

    with open(os.path.join(OUT_DIR, 'b4_results.json'), 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\nwritten: {os.path.join(OUT_DIR, 'b4_results.json')}")

if __name__ == '__main__':
    main()
