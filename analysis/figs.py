"""Step 5. Figures with both objectives side by side (written to fig/)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap

OUT = 'fig/'
import os; os.makedirs(OUT, exist_ok=True)
BOARDS = ['3x3', '3x4', '4x3', '3x5', '4x4', '4x5', '5x5']
CELLS = {b: int(b[0]) * int(b[2]) for b in BOARDS}
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
FAM = {'expectimax': '#2a78d6', 'minimax': '#eb6834', 'mcts': '#1baf7a', 'baseline': '#8a8984'}
SEQ = ['#b7d3f6', '#86b6ef', '#5598e7', '#3987e5', '#256abf', '#184f95', '#0d366b']  # 7 boards, light->dark by size
BOARD_ORDER_BY_SIZE = ['3x3', '3x4', '4x3', '3x5', '4x4', '4x5', '5x5']
BCOL = dict(zip(BOARD_ORDER_BY_SIZE, SEQ))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'legend.frameon': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                     'axes.titlesize': 9.5, 'axes.titleweight': 'bold', 'axes.titlecolor': INK})
S = pd.read_csv('out/summary.csv')
S.loc[S.solver == 'greedy', 'depth'] = 1
OBJ_LABEL = {1: 'Objective 1 (maximise score)', 2: 'Objective 2 (minimise score)'}

def save(fig, name):
    fig.savefig(OUT + name, dpi=300, bbox_inches='tight'); plt.close(fig); print('saved', name)

# ---------- Fig 1: depth ladder, normalised to greedy (=depth 1) ----------
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
for i, obj in enumerate((1, 2)):
    for j, fam in enumerate(('expectimax', 'minimax')):
        ax = axes[i, j]
        s = S[(S.objective == obj) & ((S.family == fam) | (S.solver == 'greedy')) & (S.solver != 'minimax_d5_full')]
        maxd = int(s.depth.max())
        for k in range(1, maxd + 1, 2):
            if (k // 2) % 2 == 0:
                ax.axvspan(k - 0.4, k + 1.4, color='#f3f2ef', zorder=0, lw=0)
        for b in BOARD_ORDER_BY_SIZE:
            sb = s[s.board == b].sort_values('depth')
            g = sb[sb.depth == 1]['mean'].iloc[0]
            y = sb['mean'] / g
            ax.plot(sb.depth, y, '-o', color=BCOL[b], lw=1.6, ms=3.2, label=b, zorder=3,
                    markeredgecolor='white', markeredgewidth=0.5)
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}×'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{fam.capitalize()} — {'max' if obj == 1 else 'min'}imise", loc='left')
        if j == 0:
            ax.set_ylabel('mean score ÷ greedy (log)')
        ax.set_xlabel('search depth  (1 = greedy)')
        ax.set_xticks(range(1, maxd + 1))
        for k in range(1, maxd + 1, 2):
            ax.text(k + 0.5, 0.99, f'{(k + 1) // 2} move{"s" if k > 1 else ""}', transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=6.5, color=MUTED)
        ax.axhline(1, color=MUTED, lw=0.8, ls=':')
        if obj == 2:
            ax.text(0.98, 0.06, 'lower is better', transform=ax.transAxes, ha='right', fontsize=7, color=MUTED)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, title='board', ncol=7, loc='upper center', bbox_to_anchor=(0.5, 1.03), fontsize=7.5,
           title_fontsize=7.5)
fig.tight_layout(rect=(0, 0, 1, 0.96))
save(fig, 'fig1_depth_pairing.png')

# ---------- Fig 2: effect size of each depth step ----------
P = pd.read_csv('out/depth_pairs.csv')
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
rng = np.random.default_rng(1)
for ax, obj in zip(axes, (1, 2)):
    p = P[P.objective == obj]
    for xi, step in enumerate(['odd->even', 'even->odd']):
        for fam, off, mk in (('expectimax', -0.12, 'o'), ('minimax', 0.12, 's')):
            q = p[(p.step == step) & (p.family == fam)]
            x = xi + off + rng.uniform(-0.07, 0.07, len(q))
            ax.scatter(x, q.r_rankbiserial.abs(), s=16, color=FAM[fam], marker=mk, edgecolor='white', lw=0.5,
                       label=fam if xi == 0 else None, zorder=3)
        med = p[p.step == step].r_rankbiserial.abs().median()
        ax.hlines(med, xi - 0.3, xi + 0.3, color=INK, lw=1.2, zorder=4)
        ax.text(xi + 0.32, med, f'median {med:.2f}', va='center', fontsize=7, color=INK2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['odd → even depth\n(adds a spawn layer)', 'even → odd depth\n(adds a player move)'])
    ax.set_xlim(-0.5, 1.8)
    ax.set_title(OBJ_LABEL[obj], loc='left')
axes[0].set_ylabel('|rank-biserial r| of the step')
axes[0].legend(loc='upper left', fontsize=7.5)
fig.tight_layout()
save(fig, 'fig2_step_effects.png')

# ---------- Fig 3: advantage over random vs board size ----------
SEL = [('greedy', 'baseline', 'o', 'greedy'), ('minimax_d6', 'minimax', 's', 'minimax d6'),
       ('expectimax_d6', 'expectimax', 'o', 'expectimax d6'), ('mcts_1000', 'mcts', '^', 'MCTS-1000'),
       ('expectimax_d9', 'expectimax', 'D', 'expectimax d9')]
XB = {9: 0, 12: 1, 15: 2, 16: 3, 20: 4, 25: 5}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, obj in zip(axes, (1, 2)):
    s = S[S.objective == obj]
    ends = []
    for solver, fam, mk, lab in SEL:
        q = s[s.solver == solver].copy()
        if q.empty:
            continue
        q['gain'] = q.ratio_vs_random if obj == 1 else 1 / q.ratio_vs_random
        q = q.groupby('cells').gain.mean().reset_index()  # 3x4 & 4x3 averaged (statistically equivalent)
        x = q.cells.map(XB)
        ls = '--' if solver == 'expectimax_d9' else '-'
        ax.plot(x, q.gain, ls, marker=mk, color=FAM[fam], lw=1.6, ms=4, markeredgecolor='white', mew=0.5)
        ends.append([np.log10(q.gain.iloc[-1]), lab])
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}×'))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # simple label repel in log space
    ends.sort(); lo, hi = ax.get_ylim(); L, Hh = np.log10(lo), np.log10(hi); gap = 0.075 * (Hh - L)
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + gap)
    over = ends[-1][0] - (Hh - 0.02 * (Hh - L))
    if over > 0:
        for k in range(len(ends) - 1, -1, -1):
            ends[k][0] -= over
            if k > 0 and ends[k - 1][0] > ends[k][0] - gap:
                over = ends[k - 1][0] - (ends[k][0] - gap)
            else:
                break
    for yv, lab in ends:
        ax.text(5.15, 10 ** yv, lab, fontsize=7, va='center', color=INK2)
    ax.set_xticks(range(6)); ax.set_xticklabels(['3×3', '3×4/4×3', '3×5', '4×4', '4×5', '5×5'], fontsize=7)
    ax.set_xlim(-0.3, 6.6)
    ax.set_xlabel('board (ordered by cell count: 9, 12, 15, 16, 20, 25)')
    ax.set_ylabel('× better than random (log)')
    ax.set_title(OBJ_LABEL[obj], loc='left')
fig.tight_layout()
save(fig, 'fig3_advantage_vs_size.png')

# ---------- Fig 4: rank heatmaps ----------
W = pd.read_csv('out/kendall_w.csv')
fig, axes = plt.subplots(1, 2, figsize=(7.4, 5.2))
cmap = LinearSegmentedColormap.from_list('b', ['#0d366b', '#2a78d6', '#86b6ef', '#e8f0fb'])
for ax, obj in zip(axes, (1, 2)):
    s = S[S.objective == obj]
    piv = s.pivot_table(index='solver', columns='board', values='rank')[BOARDS]
    order = piv.mean(1).sort_values().index
    piv = piv.loc[order]
    ax.imshow(piv.values, cmap=cmap, aspect='auto', vmin=1, vmax=21)
    for (r, c), v in np.ndenumerate(piv.values):
        if not np.isnan(v):
            ax.text(c, r, f'{int(v)}', ha='center', va='center', fontsize=6.3, color='white' if v < 9 else INK)
        else:
            ax.text(c, r, '–', ha='center', va='center', fontsize=6.3, color=MUTED)
    nsmall = s[s.n < 5000]
    for _, q in nsmall.iterrows():
        r = list(order).index(q.solver); c = BOARDS.index(q.board)
        ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False, ec='#eda100', lw=1.4))
    ax.set_xticks(range(7)); ax.set_xticklabels(BOARDS); ax.set_yticks(range(len(order)))
    ax.set_yticklabels([o.replace('_', ' ') for o in order], fontsize=7)
    ax.grid(False); ax.tick_params(length=0)
    for sp_ in ax.spines.values(): sp_.set_visible(False)
    w = W[W.objective == obj].iloc[0]
    ax.set_title(f"{'Obj. 1' if obj == 1 else 'Obj. 2'} rank (1 = best)\nKendall's W = {w.W:.3f} over {int(w.n_solvers)} common solvers", loc='left', fontsize=8.5)
fig.text(0.01, -0.01, 'Outlined cells: fewer than 5,000 games (kept: costly runs). Dashes: not run.', fontsize=7, color=INK2)
fig.tight_layout()
save(fig, 'fig4_rank_heatmaps.png')

# ---------- Fig 5: does skill transfer between objectives? (4x4) ----------
C = pd.read_csv('out/objective_contrast.csv')
fig, ax = plt.subplots(figsize=(4.4, 3.6))
c = C[C.board == '4x4'].copy()
def famof(s):
    return 'expectimax' if s.startswith('expectimax') else 'minimax' if s.startswith('minimax') else 'mcts' if s.startswith('mcts') else 'baseline'
c['fam'] = c.solver.map(famof)
mk = {'expectimax': 'o', 'minimax': 's', 'mcts': '^', 'baseline': 'D'}
for f, q in c.groupby('fam'):
    ax.scatter(q.max_gain, q.min_gain, s=34, color=FAM[f], marker=mk[f], edgecolor='white', lw=0.6, label=f, zorder=3)
for _, q in c.iterrows():
    if q.solver in ('mcts_1000', 'mcts_60', 'expectimax_d7', 'expectimax_d2', 'greedy', 'minimax_d7', 'random', 'expectimax_d5'):
        off = {'greedy': (-6, -9), 'expectimax_d2': (5, 4)}.get(q.solver, (4, 3))
        ax.annotate(q.solver.replace('_', ' '), (q.max_gain, q.min_gain), xytext=off, textcoords='offset points', fontsize=6.5, color=INK2,
                    ha='right' if q.solver == 'greedy' else 'left')
ax.set_xscale('log'); ax.set_yscale('log')
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}×'))
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel('Obj. 1: × higher score than random')
ax.set_ylabel('Obj. 2: × lower score than random')
ax.set_title('4×4: strength at maximising vs minimising', loc='left')
ax.legend(fontsize=7, loc='upper left')
fig.tight_layout()
save(fig, 'fig5_objective_transfer_4x4.png')

# ---------- Fig 6: score distributions are clustered by final max tile ----------
d1 = pd.read_pickle('data/obj1.pkl')
sols = [('greedy', 'greedy'), ('expectimax_d7', 'expectimax d7'), ('mcts_1000', 'MCTS-1000')]
tiles = [256, 512, 1024, 2048, 4096, 8192]
tcol = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#0d366b']
fig, axes = plt.subplots(3, 1, figsize=(5.2, 4.6), sharex=True)
bins = np.linspace(0, 160000, 120)
for ax, (s, lab) in zip(axes, sols):
    g = d1[(d1.solver == s) & (d1.board == '4x4')]
    data = [g[g.max_tile.clip(256, 8192) == t].score for t in tiles]
    ax.hist(data, bins=bins, stacked=True, color=tcol, edgecolor='white', linewidth=0.2,
            label=[str(t) for t in tiles])
    ax.set_ylabel('games', fontsize=7.5)
    ax.text(0.99, 0.85, f'{lab}  (n = {len(g):,})', transform=ax.transAxes, ha='right', fontsize=8, color=INK)
axes[0].legend(title='final max tile', ncol=6, fontsize=6.5, title_fontsize=7, loc='upper center', bbox_to_anchor=(0.5, 1.55))
axes[-1].set_xlabel('final score, 4×4, Objective 1')
fig.tight_layout()
save(fig, 'fig6_score_distribution_4x4.png')

# ---------- Fig 7: expectimax vs minimax at matched depth ----------
E = pd.read_csv('out/expectimax_vs_minimax.csv')
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=False)
div = LinearSegmentedColormap.from_list('d', ['#eb6834', '#f7f6f3', '#2a78d6'])
for ax, obj in zip(axes, (1, 2)):
    e = E[E.objective == obj].pivot_table(index='depth', columns='board', values='pct')[BOARDS]
    val = e.values if obj == 1 else -e.values  # positive = expectimax better
    im = ax.imshow(val, cmap=div, vmin=-60, vmax=60, aspect='auto')
    for (r, cc), v in np.ndenumerate(val):
        if not np.isnan(v):
            ax.text(cc, r, f'{v:+.0f}', ha='center', va='center', fontsize=6.5, color=INK)
    ax.set_xticks(range(7)); ax.set_xticklabels(BOARDS); ax.set_yticks(range(len(e))); ax.set_yticklabels([f'd{int(x)}' for x in e.index])
    ax.grid(False); ax.tick_params(length=0)
    for sp_ in ax.spines.values(): sp_.set_visible(False)
    ax.set_title(f"{'Objective 1 (maximise)' if obj == 1 else 'Objective 2 (minimise)'}", loc='left', fontsize=8.5)
cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
cb.set_label('% expectimax advantage\n(blue: expectimax better; orange: minimax better)', fontsize=6.5)
fig.suptitle('Expectimax vs minimax at the same depth', x=0.02, ha='left', fontsize=9.5, fontweight='bold', y=1.02)
save(fig, 'fig7_expectimax_vs_minimax.png')
