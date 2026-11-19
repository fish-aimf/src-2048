"""Step 2. Descriptive statistics (bootstrap CIs), Kruskal-Wallis, Dunn post-hoc,
neighbouring-rank comparisons and Kendall's W across boards.
"""
import numpy as np, pandas as pd
from scipy import stats
import scikit_posthocs as sp

D = 'data/'
rng = np.random.default_rng(2048)
OUT = 'out/'
import os; os.makedirs(OUT, exist_ok=True)

def boot_ci(x, B=10000):
    x = np.asarray(x, float); n = len(x)
    means = np.empty(B)
    for i in range(0, B, 500):
        idx = rng.integers(0, n, size=(min(500, B - i), n))
        means[i:i + idx.shape[0]] = x[idx].mean(1)
    return np.percentile(means, [2.5, 97.5])

BOARDS = ['3x3', '3x4', '4x3', '3x5', '4x4', '4x5', '5x5']
rows = []
for obj in (1, 2):
    d = pd.read_pickle(f'{D}obj{obj}.pkl')
    for (b, s), g in d.groupby(['board', 'solver']):
        x = g.score.values
        lo, hi = boot_ci(x)
        rows.append(dict(objective=obj, board=b, cells=g.cells.iloc[0], solver=s, family=g.family.iloc[0],
                         depth=g.depth.iloc[0], n=len(x), mean=x.mean(), ci_lo=lo, ci_hi=hi,
                         median=np.median(x), q1=np.percentile(x, 25), q3=np.percentile(x, 75),
                         sd=x.std(ddof=1), se=x.std(ddof=1) / np.sqrt(len(x)), cv=x.std(ddof=1) / x.mean(),
                         skew=stats.skew(x), min=x.min(), max=x.max(),
                         mean_maxtile=g.max_tile.mean(), modal_maxtile=g.max_tile.mode().iloc[0],
                         best_maxtile=g.max_tile.max(),
                         p2048=(g.max_tile >= 2048).mean(), p1024=(g.max_tile >= 1024).mean(),
                         mean_moves=g.moves.mean(), median_sec_per_move=g.sec_per_move.median(),
                         score_per_move=(g.score / g.moves.clip(lower=1)).mean()))
S = pd.DataFrame(rows)
S['board'] = pd.Categorical(S.board, BOARDS, ordered=True)
S = S.sort_values(['objective', 'board', 'mean'], ascending=[True, True, False])
# rank: obj1 higher better, obj2 lower better
S['rank'] = S.groupby(['objective', 'board'], observed=True)['mean'].rank(ascending=False)
m2 = S.objective == 2
S.loc[m2, 'rank'] = S[m2].groupby('board', observed=True)['mean'].rank(ascending=True)
# ratio vs random on same board
rnd = S[S.solver == 'random'].set_index(['objective', 'board'])['mean']
S['ratio_vs_random'] = [r['mean'] / rnd[(r.objective, r.board)] for _, r in S.iterrows()]
S.to_csv(OUT + 'summary.csv', index=False)
print('summary rows', len(S))

# ---------- omnibus per board ----------
om = []
for obj in (1, 2):
    d = pd.read_pickle(f'{D}obj{obj}.pkl')
    for b in BOARDS:
        g = d[d.board == b]
        groups = [x.score.values for _, x in g.groupby('solver')]
        H, p = stats.kruskal(*groups)
        k, n = len(groups), sum(map(len, groups))
        eps2 = (H - k + 1) / (n - k)  # epsilon-squared (Tomczak & Tomczak 2014)
        F, pF = stats.f_oneway(*groups)
        lev = stats.levene(*groups, center='median')
        om.append(dict(objective=obj, board=b, k=k, N=n, kruskal_H=H, p=p, epsilon_sq=eps2,
                       anova_F=F, anova_p=pF, levene_W=lev.statistic, levene_p=lev.pvalue))
        # full Dunn matrix (Holm)
        dm = sp.posthoc_dunn(g, val_col='score', group_col='solver', p_adjust='holm')
        dm.to_csv(f'{OUT}dunn_obj{obj}_{b}.csv')
pd.DataFrame(om).to_csv(OUT + 'omnibus.csv', index=False)
print(pd.DataFrame(om).round(4).to_string())

# ---------- adjacent-rank pairwise (Mann-Whitney + rank-biserial, Holm within board) ----------
from statsmodels.stats.multitest import multipletests
pw = []
for obj in (1, 2):
    d = pd.read_pickle(f'{D}obj{obj}.pkl')
    for b in BOARDS:
        order = S[(S.objective == obj) & (S.board == b)].sort_values('rank').solver.tolist()
        tmp = []
        for a, c in zip(order[:-1], order[1:]):
            x = d[(d.board == b) & (d.solver == a)].score.values
            y = d[(d.board == b) & (d.solver == c)].score.values
            U, p = stats.mannwhitneyu(x, y, alternative='two-sided')
            r = 1 - 2 * U / (len(x) * len(y))  # rank-biserial: >0 means y tends larger
            tmp.append(dict(objective=obj, board=b, better=a, next=c, U=U, p=p, rank_biserial=-r))
        ps = multipletests([t['p'] for t in tmp], method='holm')[1]
        for t, q in zip(tmp, ps):
            t['p_holm'] = q; t['sig'] = q < 0.05
        pw += tmp
pd.DataFrame(pw).to_csv(OUT + 'adjacent_pairs.csv', index=False)

# ---------- Kendall's W across boards (common solvers) ----------
def kendall_w(R):  # R: boards x solvers rank matrix
    m, n = R.shape
    Rj = R.sum(0)
    Ssum = ((Rj - Rj.mean()) ** 2).sum()
    W = 12 * Ssum / (m ** 2 * (n ** 3 - n))
    chi2 = m * (n - 1) * W
    return W, chi2, stats.chi2.sf(chi2, n - 1)

kw = []
for obj in (1, 2):
    s = S[S.objective == obj]
    piv = s.pivot_table(index='board', columns='solver', values='mean', observed=True)
    common = piv.columns[piv.notna().all()]
    piv = piv[common]
    R = piv.rank(axis=1, ascending=(obj == 2)).values
    W, chi2, p = kendall_w(R)
    kw.append(dict(objective=obj, n_solvers=len(common), m_boards=len(piv), W=W, chi2=chi2, p=p,
                   solvers=';'.join(common)))
    sp_mat = piv.T.corr(method='spearman')
    sp_mat.to_csv(f'{OUT}board_rank_spearman_obj{obj}.csv')
    print(f'obj{obj} min pairwise Spearman between boards', sp_mat.values[np.triu_indices(len(sp_mat), 1)].min().round(3))
    # best per board (all solvers incl. partial)
pd.DataFrame(kw).to_csv(OUT + 'kendall_w.csv', index=False)
print(pd.DataFrame(kw)[['objective', 'n_solvers', 'm_boards', 'W', 'chi2', 'p']])
best = S[S['rank'] <= 3][['objective', 'board', 'rank', 'solver', 'n', 'mean', 'ci_lo', 'ci_hi', 'median']]
print(best.sort_values(['objective', 'board', 'rank']).to_string())
