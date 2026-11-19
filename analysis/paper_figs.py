"""Step 7. The final figures used in the paper (written to paper/)."""
import os, sys, shutil, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from scipy import stats

DEST = sys.argv[1] if len(sys.argv) > 1 else "paper"
FIG = os.path.join(DEST, 'figures'); APP = os.path.join(DEST, 'appendix')
for d in (FIG, APP): os.makedirs(d, exist_ok=True)
BOARDS = ['3x3', '3x4', '4x3', '3x5', '4x4', '4x5', '5x5']
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
FAM = {'expectimax': '#2a78d6', 'minimax': '#eb6834', 'mcts': '#1baf7a', 'baseline': '#8a8984'}
MK = {'expectimax': 'o', 'minimax': 's', 'mcts': '^', 'baseline': 'D'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'legend.frameon': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                     'axes.titlesize': 9.5, 'axes.titleweight': 'bold', 'axes.titlecolor': INK})
famof = lambda s: 'expectimax' if s.startswith('expectimax') else 'minimax' if s.startswith('minimax') else 'mcts' if s.startswith('mcts') else 'baseline'
pretty = lambda s: s.replace('expectimax_d', 'expectimax d').replace('minimax_d', 'minimax d').replace('mcts_', 'MCTS-')
xfmt = FuncFormatter(lambda v, _: f'{v:g}×')
S = pd.read_csv('out/summary.csv')
COMMON = pd.read_csv('out/kendall_w.csv').iloc[0].solvers.split(';')  # 16 solvers run on all 7 boards in both objectives

def kendall_w(R):
    m, n = R.shape; Rj = R.sum(0)
    W = 12 * ((Rj - Rj.mean()) ** 2).sum() / (m ** 2 * (n ** 3 - n)); chi2 = m * (n - 1) * W
    return W, stats.chi2.sf(chi2, n - 1)

def save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches='tight'); plt.close(fig); print('saved', path)

# ---- Fig 1: depth ladder (already final) ----
shutil.copy('fig/fig1_depth_pairing.png', os.path.join(FIG, 'fig1_depth_pairing.png'))

# ---- Fig 2: ranks of the 16 common solvers, same row order in both objectives ----
cmap = LinearSegmentedColormap.from_list('b', ['#0d366b', '#2a78d6', '#86b6ef', '#e8f0fb'])
s = S[S.solver.isin(COMMON)].copy()
s['r16'] = s.groupby(['objective', 'board'])['mean'].rank(ascending=False)
s.loc[s.objective == 2, 'r16'] = s[s.objective == 2].groupby('board')['mean'].rank(ascending=True)
order = s[s.objective == 1].pivot_table(index='solver', columns='board', values='r16').mean(axis=1).sort_values().index
fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6), sharey=True)
for ax, obj in zip(axes, (1, 2)):
    piv = s[s.objective == obj].pivot_table(index='solver', columns='board', values='r16')[BOARDS].loc[order]
    W, p = kendall_w(piv.T.values)
    ax.imshow(piv.values, cmap=cmap, aspect='auto', vmin=1, vmax=16)
    for (r, c), v in np.ndenumerate(piv.values):
        ax.text(c, r, f'{int(v)}', ha='center', va='center', fontsize=6.6, color='white' if v < 7 else INK)
    for _, q in s[(s.objective == obj) & (s.n < 5000)].iterrows():
        ax.add_patch(plt.Rectangle((BOARDS.index(q.board) - .5, list(order).index(q.solver) - .5), 1, 1, fill=False, ec='#eda100', lw=1.4))
    ax.set_xticks(range(7)); ax.set_xticklabels([b.replace('x', '×') for b in BOARDS]); ax.set_yticks(range(len(order)))
    ax.set_yticklabels([pretty(o) for o in order], fontsize=7.2); ax.grid(False); ax.tick_params(length=0)
    for sp_ in ax.spines.values(): sp_.set_visible(False)
    ax.set_title(f"{'Objective 1 (maximise)' if obj == 1 else 'Objective 2 (minimise)'}\nKendall's W = {W:.3f}", loc='left', fontsize=8.8)
fig.text(0.01, -0.02, 'Rank 1 = best. Rows ordered by Objective 1 rank. Outlined: fewer than 5,000 games.', fontsize=7, color=INK2)
fig.tight_layout(); save(fig, os.path.join(FIG, 'fig2_rank_stability.png'))

# ---- Fig 3: strength at maximising vs minimising, every board ----
C = pd.read_csv('out/objective_contrast.csv')
fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.2)); axes = axes.ravel()
for ax, b in zip(axes, BOARDS):
    c = C[C.board == b].copy(); c['fam'] = c.solver.map(famof)
    for f, q in c.groupby('fam'):
        ax.scatter(q.max_gain, q.min_gain, s=18, color=FAM[f], marker=MK[f], edgecolor='white', lw=0.5, zorder=3)
    rho, p = stats.spearmanr(c.max_gain, c.min_gain)
    ax.set_xscale('log'); ax.set_yscale('log')
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, subs=(1, 3)) if axis is ax.xaxis else LogLocator(base=10, subs=(1, 2, 5)))
        axis.set_major_formatter(xfmt); axis.set_minor_formatter(NullFormatter())
    ax.tick_params(labelsize=6)
    ax.set_title(b.replace('x', '×'), loc='left', fontsize=8.5)
    ax.text(0.97, 0.05, f'ρ = {rho:.2f}' + ('' if p < 0.05 else ' (n.s.)'), transform=ax.transAxes, ha='right', fontsize=6.8, color=INK2)
ax = axes[-1]; ax.axis('off')
for i, f in enumerate(['expectimax', 'minimax', 'mcts', 'baseline']):
    ax.scatter(0.1, 0.85 - i * 0.14, s=26, color=FAM[f], marker=MK[f]); ax.text(0.2, 0.85 - i * 0.14, {'mcts': 'MCTS', 'baseline': 'greedy / random'}.get(f, f), va='center', fontsize=7.5)
ax.text(0.05, 0.25, 'x: × higher score than\nrandom (Obj. 1)\ny: × lower score than\nrandom (Obj. 2)', fontsize=6.8, color=INK2, va='top')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.tight_layout(); save(fig, os.path.join(FIG, 'fig3_objective_transfer.png'))

# ---- Appendix A1: 4x4 score distributions, both objectives ----
tcol = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#0d366b']
spec = {1: ([('greedy', 'greedy'), ('expectimax_d7', 'expectimax d7'), ('mcts_1000', 'MCTS-1000')], [256, 512, 1024, 2048, 4096, 8192], np.linspace(0, 160000, 110)),
        2: ([('greedy', 'greedy'), ('minimax_d7', 'minimax d7'), ('expectimax_d9', 'expectimax d9')], [8, 16, 32, 64, 128, 256], np.linspace(0, 700, 110))}
fig, axes = plt.subplots(3, 2, figsize=(7.2, 4.8), sharex='col')
for col, obj in enumerate((1, 2)):
    d = pd.read_pickle(f'data/obj{obj}.pkl'); d = d[d.board == '4x4']
    sols, tiles, bins = spec[obj]
    for row, (sname, lab) in enumerate(sols):
        ax = axes[row, col]; g = d[d.solver == sname]
        labs = [('≤' if i == 0 else '≥' if i == len(tiles) - 1 else '') + f'{t:,}' for i, t in enumerate(tiles)]
        ax.hist([g[g.max_tile.clip(tiles[0], tiles[-1]) == t].score for t in tiles], bins=bins, stacked=True, color=tcol, edgecolor='white', linewidth=0.2, label=labs)
        ax.text(0.98, 0.82, lab, transform=ax.transAxes, ha='right', fontsize=7.5, color=INK); ax.tick_params(labelsize=6.5)
        if obj == 1: ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v/1000:g}k' if v else '0'))
        if col == 0: ax.set_ylabel('games', fontsize=7)
    axes[0, col].set_title('Objective 1 (maximise)' if obj == 1 else 'Objective 2 (minimise)', loc='left', fontsize=8.8)
    axes[0, col].legend(title='final max tile', ncol=3, fontsize=5.8, title_fontsize=6.2, loc='upper center', bbox_to_anchor=(0.5, 1.75))
    axes[-1, col].set_xlabel('final score, 4×4')
fig.tight_layout(); save(fig, os.path.join(APP, 'figA1_score_distributions_4x4.png'))

# ---- Appendix A2: expectimax vs minimax (already final) ----
shutil.copy('fig/fig7_expectimax_vs_minimax.png', os.path.join(APP, 'figA2_expectimax_vs_minimax.png'))
print('W obj1/obj2 on common solvers printed in fig2 titles')
