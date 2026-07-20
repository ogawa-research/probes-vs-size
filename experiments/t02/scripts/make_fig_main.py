"""T02 main figure: effective length vs model size, colored by family, with
the same-size counterexample annotated. Data = frozen b5_results.json dataset.
Deterministic; writes papers/t02/fig_main.pdf.

Run: python3 make_fig_main.py
"""
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DATA = os.path.join(ROOT, 'data', 't02_b5_2026-07-15', 'b5_results.json')
OUT = os.path.join(ROOT, 'papers', 't02', 'fig_main.pdf')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FAM_COLOR = {'qwen3': '#1b6ca8', 'gemma': '#c0392b', 'llama': '#27ae60',
             'mistral': '#8e44ad', 'yi': '#d68910', 'phi': '#16a085',
             'cohere': '#7f8c8d'}


def main():
    rows = json.load(open(DATA))['dataset']
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.9))

    # -- left: y=log2(eff) vs log10(params), colored by family
    seen = set()
    for r in rows:
        x = math.log10(r['params'])
        y = r['y']
        fam = r['family']
        lab = fam if fam not in seen else None
        seen.add(fam)
        m = '^' if r['censored'] else 'o'
        axL.scatter(x, y, c=FAM_COLOR[fam], marker=m, s=46,
                    edgecolor='k', linewidth=0.4, zorder=3)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                      markeredgecolor='k', markersize=7, label=f)
               for f, c in FAM_COLOR.items()]
    axL.legend(handles=handles, fontsize=6.5, ncol=2, frameon=False, loc='lower right')
    axL.set_xlabel(r'$\log_{10}$(parameters)')
    axL.set_ylabel(r'$\log_2$(effective length)')
    axL.set_title('Effective length tracks size\n(triangles: left-censored $<$4K)', fontsize=9)
    axL.grid(alpha=0.25, zorder=0)

    # -- right: within-8B-class same-size contrast (R vs eff), the counterexample
    g2 = [r for r in rows if r['name'] in
          ('llama-3.1-8b', 'prolong-llama3-8b', 'gradientai-llama3-8b')]
    names = {'llama-3.1-8b': 'llama-3.1', 'prolong-llama3-8b': 'prolong',
             'gradientai-llama3-8b': 'gradientai'}
    for r in g2:
        axR.scatter(r['R'], r['eff'] / 1000.0, c=FAM_COLOR['llama'], s=70,
                    edgecolor='k', linewidth=0.5, zorder=3)
        # offsets chosen to keep labels inside the axes (llama-3.1 sits near the
        # left edge: below-right, prolong above, gradientai below)
        off = {'llama-3.1-8b': (6, -24), 'prolong-llama3-8b': (0, 12),
               'gradientai-llama3-8b': (-6, -24)}[r['name']]
        axR.annotate(f"{names[r['name']]}\n({r['eff']//1024}K)",
                     (r['R'], r['eff'] / 1000.0), fontsize=7.5, ha='center',
                     xytext=off, textcoords='offset points')
    axR.set_xlabel(r'$R$ (attention-entropy resolution)')
    axR.set_ylabel('effective length (K tokens)')
    axR.set_title('Same-size (8B) contrast:\nhighest $R$, lowest effective length', fontsize=9)
    axR.grid(alpha=0.25, zorder=0)
    axR.set_xlim(0.898, 0.946)
    axR.set_ylim(8, 40)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches='tight')
    print('written:', OUT)
    print('G2 points:', [(r['name'], round(r['R'], 3), r['eff']) for r in g2])


if __name__ == '__main__':
    main()
