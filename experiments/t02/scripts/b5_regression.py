"""T02 B5: pre-registered extension analysis (instructions sec.4 + Amendment 1).

Reuses b4_regression.py verbatim (MODEL_SPECS, LOFO-CV, permutation seeds) —
the B4 verdict is immutable; B5 reports robustness of the same comparisons at
N=19 (15 + megabeam/prolong/gradientai/command-r; internlm/lwm/film skipped
for structural tokenizer incompatibility, disclosed) with Gemma R replaced by
the v2 (CLIP_FLOOR-fixed, all-layer) measurement as primary and the B4 rescue
(global-layer-only) as sensitivity.

y for new models: researcher-recomputed from the NVIDIA-measured public rows
(t02_ruler_public_pack_2026-07-11.md, threshold 85.6 contiguous):
  prolong 32K / megabeam 32K / command-r 32K / gradientai-8b 16K.
Families for LOFO: long-context finetunes join their base architecture family
(prolong, gradientai -> llama; megabeam -> mistral); command-r -> cohere (new).

Run: python3 b5_regression.py   (deterministic; rewrites b5_results.json)
"""
import json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b4_regression as b4  # noqa: E402  (module import assembles B4 rows)

DATA = b4.DATA
B5 = os.path.join(DATA, 't02_b5_2026-07-15')
OUT_DIR = b5_out = os.path.join(DATA, 't02_b5_2026-07-15')

NEW_MODELS = [
    # name                   family     eff  cens  claimed   params  erank_file
    ('megabeam-mistral-7b',  'mistral', 32768, False,  524288,  7.0,
     'erank_pr_megabeam-mistral-7b.ckpt.json'),
    ('prolong-llama3-8b',    'llama',   32768, False,  524288,  8.0,
     'erank_pr_prolong-llama3-8b.json'),
    ('gradientai-llama3-8b', 'llama',   16384, False, 1048576,  8.0,
     'erank_pr_gradientai-llama3-8b.json'),
    ('command-r-v01',        'cohere',  32768, False,  131072, 35.0,
     'erank_pr_command-r-v01.json'),
]

GEMMA = ['gemma-3-270m', 'gemma-3-1b', 'gemma-3-4b', 'gemma-3-12b', 'gemma-3-27b']


def jload(p):
    with open(p) as f:
        return json.load(f)


def gemma_v2_R(name):
    """Primary R from the CLIP_FLOOR-fixed v2 file (all layers, recorded R),
    plus global-layer-only R (B4 rescue method re-applied to v2) for sensitivity."""
    d = jload(os.path.join(B5, f'r_delta_{name}_v2.json'))
    rec = d['neutral']['R']['R']
    assert math.isfinite(rec), name
    pls = d['neutral']['per_length']
    lengths = sorted(pls, key=int)
    # verify all-layer finiteness (gate G4 already checked on 4b; re-verify all)
    for L in lengths:
        assert all(math.isfinite(v['H_mean']) for v in pls[L]['per_layer'].values()), (name, L)
    # global layers = every 6th (0-indexed 5,11,...) as in B4 rescue
    n_layers = d['n_layers']
    glob = [l for l in range(n_layers) if (l + 1) % 6 == 0]
    H = [sum(pls[L]['per_layer'][str(l)]['H_mean'] for l in glob) / len(glob)
         for L in lengths]
    x = [math.log(int(L)) for L in lengths]
    mx, my = sum(x) / len(x), sum(H) / len(H)
    slope = (sum((xi - mx) * (hi - my) for xi, hi in zip(x, H))
             / sum((xi - mx) ** 2 for xi in x))
    return rec, 1.0 - slope


def new_model_row(name, fam, eff, cens, claimed, params, erank_file):
    rd = jload(os.path.join(B5, f'r_delta_{name}.json'))
    R = rd['neutral']['R']['R']
    assert math.isfinite(R), name
    ar = jload(os.path.join(B5, f'a_retrieval_{name}.json'))
    ag = ar['standard']['aggregate']
    ek = jload(os.path.join(B5, erank_file))
    pl = ek['neutral']['per_layer']
    er = {int(k): v['value_erank'] for k, v in pl.items()}
    ner = {int(k): v['value_ner'] for k, v in pl.items()}
    return dict(name=name, family=fam, eff=eff, censored=cens,
                y=math.log2(eff), claimed=claimed, params=params,
                d_model=rd['d_model'],
                R=R, R_n_layers_used=rd['n_layers'], n_layers=rd['n_layers'],
                A_mass=ag['top_5pct_s_h_sum'], A_gini=ag['gini_coefficient'],
                W_erank_all=sum(er.values()) / len(er),
                W_ner_all=sum(ner.values()) / len(ner),
                W_erank_band=None, W_kJ=None, band_healthy=False)


def kendall_tau(xs, ys):
    """plain Kendall tau-a over pairs (ties counted as discordant-neutral: skipped)."""
    n = len(xs)
    conc = disc = used = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = xs[i] - xs[j], ys[i] - ys[j]
            if dx == 0 or dy == 0:
                continue
            used += 1
            if (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    return (conc - disc) / used if used else None, conc, disc, used


def main():
    # ---- assemble N=19 with Gemma R v2 primary
    rows = [dict(r) for r in b4.rows]
    gemma_v2 = {g: gemma_v2_R(g) for g in GEMMA}
    diff_table = {}
    for r in rows:
        if r['name'] in gemma_v2:
            v2_all, v2_glob = gemma_v2[r['name']]
            diff_table[r['name']] = {
                'b4_rescue_global_only_R': r['R'],
                'v2_all_layer_R': v2_all,
                'v2_global_only_R': v2_glob,
                'v2all_minus_b4rescue': v2_all - r['R'],
                'v2glob_minus_b4rescue': v2_glob - r['R'],
            }
            r['R'] = v2_all
    for spec in NEW_MODELS:
        rows.append(new_model_row(*spec))

    out = {'note': 'B5 pre-registered extension; B4 verdict immutable',
           'gemma_R_diff_table': diff_table,
           'dataset': rows}
    print(f"N = {len(rows)}, families = {sorted({r['family'] for r in rows})}")
    for r in rows:
        print(f"  {r['name']:22s} {r['family']:8s} eff={r['eff']:>6d}"
              f"{'*' if r['censored'] else ' '} R={r['R']:.3f}"
              f" A_mass={r['A_mass']:.2f} A_gini={r['A_gini']:.3f}"
              f" W_er_all={r['W_erank_all']:.1f} params={r['params']}")

    # ---- main suite (identical MODEL_SPECS / CV / seeds)
    main_res = b4.run_suite(rows, 'b5_main')
    out['main'] = main_res
    print('\n== B5 main (N=19, LOFO-CV MAE in log2 units) ==')
    for k, v in sorted(main_res.items(), key=lambda kv: kv[1]):
        print(f"  {k:18s} {v:.4f}")

    best_single = min((v, k) for k, v in main_res.items() if k.startswith('single_'))
    best_comp = min((v, k) for k, v in main_res.items() if k.startswith(('add_', 'mult_')))
    best_base = min((v, k) for k, v in main_res.items() if k.startswith('base_'))
    best_mech = min(best_single, best_comp)
    all_bases = {k: v for k, v in main_res.items() if k.startswith('base_')}
    mech_beats_all_bases = all(best_mech[0] < v for v in all_bases.values())

    p_diff_mech_vs_base = b4.permutation_p_diff(
        rows, b4.MODEL_SPECS[best_mech[1]], b4.MODEL_SPECS[best_base[1]],
        best_mech[0] - best_base[0])
    p_comp = b4.permutation_p(rows, b4.MODEL_SPECS[best_comp[1]], best_comp[0])
    p_diff_comp_single = b4.permutation_p_diff(
        rows, b4.MODEL_SPECS[best_comp[1]], b4.MODEL_SPECS[best_single[1]],
        best_comp[0] - best_single[0])

    out['comparison'] = {
        'best_single': {'model': best_single[1], 'mae': best_single[0]},
        'best_composite': {'model': best_comp[1], 'mae': best_comp[0]},
        'best_baseline': {'model': best_base[1], 'mae': best_base[0]},
        'all_baselines': all_bases,
        'composite_beats_single': best_comp[0] < best_single[0],
        'mechanism_beats_ALL_baselines': mech_beats_all_bases,
        'reversal_p_mech_minus_bestbase': p_diff_mech_vs_base,
        'reversal_declared': bool(mech_beats_all_bases and p_diff_mech_vs_base < 0.05),
        'best_composite_p': p_comp,
        'diff_composite_minus_single_p': p_diff_comp_single,
    }
    print(f"\nbest mech = {best_mech[1]} MAE {best_mech[0]:.4f}"
          f" | best base = {best_base[1]} MAE {best_base[0]:.4f}"
          f" | mech beats ALL bases: {mech_beats_all_bases}"
          f" | paired p (mech-base) = {p_diff_mech_vs_base:.4f}"
          f" | reversal: {out['comparison']['reversal_declared']}")

    # ---- sensitivity: censored excluded (N=16)
    sub = [r for r in rows if not r['censored']]
    out['sens_censor_excluded'] = b4.run_suite(sub, 'b5_nocens')
    print('\n== sensitivity: censored excluded (N=16), top 6 ==')
    for k, v in sorted(out['sens_censor_excluded'].items(), key=lambda kv: kv[1])[:6]:
        print(f"  {k:18s} {v:.4f}")

    # ---- sensitivity: Gemma R = v2 global-only (continuity with B4 rescue)
    rows_g = [dict(r) for r in rows]
    for r in rows_g:
        if r['name'] in gemma_v2:
            r['R'] = gemma_v2[r['name']][1]
    out['sens_gemma_R_global_only'] = b4.run_suite(rows_g, 'b5_gglob')

    # ---- same-size control groups (descriptive, pre-registered sec.4-2)
    groups = {
        'G1_mistral7B': ['mistral-7b', 'megabeam-mistral-7b'],
        'G2_llama8B': ['llama-3.1-8b', 'prolong-llama3-8b', 'gradientai-llama3-8b'],
    }
    factors = ['R', 'A_mass', 'A_gini', 'W_erank_all', 'log10_params']
    data = b4.add_derived([dict(r) for r in rows])
    byname = {r['name']: r for r in data}
    ss = {}
    print('\n== same-size control groups (Kendall tau vs effective, descriptive) ==')
    for gname, members in groups.items():
        g = [byname[m] for m in members]
        effs = [r['eff'] for r in g]
        entry = {'members': {r['name']: {'eff': r['eff'], 'params': r['params'],
                                         **{f: r[f] for f in factors}} for r in g}}
        for f in factors:
            tau, c, d, used = kendall_tau([r[f] for r in g], effs)
            entry[f'tau_{f}'] = {'tau': tau, 'concordant': c, 'discordant': d,
                                 'pairs_used': used}
            print(f"  {gname} {f:12s} tau={tau if tau is None else round(tau,3)}"
                  f" (C={c} D={d} of {used})")
        ss[gname] = entry
    out['same_size_groups'] = ss

    with open(os.path.join(OUT_DIR, 'b5_results.json'), 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\nwritten: {os.path.join(OUT_DIR, 'b5_results.json')}")


if __name__ == '__main__':
    main()
