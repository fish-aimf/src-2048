"""Step 3. Depth pairing: seed-matched comparison of each depth with the next,
effect sizes, and the parity regression.
"""
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

OUT = 'out/'
BOARDS = ['3x3', '3x4', '4x3', '3x5', '4x4', '4x5', '5x5']
rows = []
regs = []
for obj in (1, 2):
    d = pd.read_pickle(f'data/obj{obj}.pkl')
    d.loc[d.solver == 'greedy', 'depth'] = 1
    for fam in ('expectimax', 'minimax'):
        f = d[((d.family == fam) | (d.solver == 'greedy')) & (d.solver != 'minimax_d5_full')]
        for b in BOARDS:
            fb = f[f.board == b]
            depths = sorted(fb.depth.unique())
            for a, c in zip(depths[:-1], depths[1:]):
                x = fb[fb.depth == a].set_index('seed')
                y = fb[fb.depth == c].set_index('seed')
                j = x.join(y, lsuffix='_a', rsuffix='_b', how='inner')
                if len(j) < 100:
                    continue
                same = ((j.score_a == j.score_b) & (j.moves_a == j.moves_b) & (j.board_sum_a == j.board_sum_b))
                diff = j.score_b - j.score_a
                nz = diff[diff != 0]
                if len(nz) > 0:
                    W, pw = stats.wilcoxon(j.score_b, j.score_a, zero_method='wilcox')
                    rk = stats.rankdata(np.abs(nz))
                    r_mp = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()  # matched-pairs rank-biserial
                else:
                    pw, r_mp = 1.0, 0.0
                U, pu = stats.mannwhitneyu(y.score, x.score)
                r_ind = 2 * U / (len(x) * len(y)) - 1
                rows.append(dict(objective=obj, family=fam, board=b, d_from=int(a), d_to=int(c),
                                 step='odd->even' if int(a) % 2 == 1 else 'even->odd',
                                 n_paired=len(j), identical_game_rate=same.mean(),
                                 mean_from=x.score.mean(), mean_to=y.score.mean(),
                                 pct_change=100 * (y.score.mean() / x.score.mean() - 1),
                                 wilcoxon_p=pw, r_matched=r_mp, mw_p=pu, r_rankbiserial=r_ind))
        # parity regression on log score, per family/objective:
        # log(score) ~ board FE + C(lookahead player moves) + even ; even-coef = what the extra chance/min layer adds
        f = f.copy()
        f['plies'] = np.ceil(f.depth / 2).astype(int)
        f['even'] = (f.depth % 2 == 0).astype(int)
        f['ls'] = np.log(f.score.clip(lower=1))
        m_full = smf.ols('ls ~ C(board) * C(depth)', f).fit()
        m_pair = smf.ols('ls ~ C(board) * C(plies)', f).fit()
        m_par = smf.ols('ls ~ C(board) * C(plies) + even', f).fit(cov_type='HC3')
        regs.append(dict(objective=obj, family=fam, n=len(f), R2_depth=m_full.rsquared, R2_pairs=m_pair.rsquared,
                         share_explained_by_pairs=m_pair.rsquared / m_full.rsquared,
                         even_coef=m_par.params['even'], even_pct=100 * (np.exp(m_par.params['even']) - 1),
                         even_ci_lo=100 * (np.exp(m_par.conf_int().loc['even', 0]) - 1),
                         even_ci_hi=100 * (np.exp(m_par.conf_int().loc['even', 1]) - 1),
                         even_p=m_par.pvalues['even']))
P = pd.DataFrame(rows)
P.to_csv(OUT + 'depth_pairs.csv', index=False)
R = pd.DataFrame(regs)
R.to_csv(OUT + 'parity_regression.csv', index=False)
pd.set_option('display.width', 250)
print(P.groupby(['objective', 'family', 'step']).agg(identical=('identical_game_rate', 'mean'),
      abs_r=('r_rankbiserial', lambda s: s.abs().mean()), med_pct=('pct_change', 'median'),
      n=('board', 'size')).round(3))
print(P[P.step == 'odd->even'][['objective', 'family', 'board', 'd_from', 'd_to', 'identical_game_rate', 'pct_change', 'r_rankbiserial', 'wilcoxon_p']].round(3).to_string())
print(R.round(4).to_string())
