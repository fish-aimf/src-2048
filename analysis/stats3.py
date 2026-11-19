"""Step 4. Board-size scaling, 3x4 vs 4x3 check, random-play control, design
regressions, objective comparison and outcome correlations.
"""
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
pd.set_option('display.width', 250)
OUT = 'out/'
BOARDS = ['3x3', '3x4', '4x3', '3x5', '4x4', '4x5', '5x5']
d1 = pd.read_pickle('data/obj1.pkl'); d2 = pd.read_pickle('data/obj2.pkl')
S = pd.read_csv(OUT + 'summary.csv')

# 1) symmetry 3x4 vs 4x3 (transposed boards; game + heuristic are rotation/transpose invariant)
sym = []
for obj, d in ((1, d1), (2, d2)):
    for s in sorted(d.solver.unique()):
        x = d[(d.solver == s) & (d.board == '3x4')].score; y = d[(d.solver == s) & (d.board == '4x3')].score
        if len(x) and len(y):
            U, p = stats.mannwhitneyu(x, y)
            sym.append(dict(objective=obj, solver=s, n34=len(x), n43=len(y), mean34=x.mean(), mean43=y.mean(),
                            pct_diff=100 * (x.mean() / y.mean() - 1), p=p, r=2 * U / (len(x) * len(y)) - 1))
sym = pd.DataFrame(sym)
from statsmodels.stats.multitest import multipletests
sym['p_holm'] = multipletests(sym.p, method='holm')[1]
sym.to_csv(OUT + 'symmetry_3x4_4x3.csv', index=False)
print('SYMMETRY: n solvers', len(sym), ' sig after Holm:', (sym.p_holm < .05).sum(),
      ' max |r|', sym.r.abs().max().round(3), ' max |pct|', sym.pct_diff.abs().max().round(2))
print(sym[sym.p_holm < .05].round(4).to_string())

# 2) random control: identical under both objectives?
rc = []
for b in BOARDS:
    x = d1[(d1.solver == 'random') & (d1.board == b)].score; y = d2[(d2.solver == 'random') & (d2.board == b)].score
    U, p = stats.mannwhitneyu(x, y)
    rc.append(dict(board=b, mean_obj1=x.mean(), mean_obj2=y.mean(), p=p, r=2 * U / (len(x) * len(y)) - 1))
rc = pd.DataFrame(rc); rc.to_csv(OUT + 'random_control.csv', index=False); print(rc.round(4).to_string())

# 3) scaling exponents: log(score) ~ log(cells) per solver (boards with full coverage), per-game ols
sc = []
for obj, d in ((1, d1), (2, d2)):
    for s in sorted(d.solver.unique()):
        g = d[d.solver == s]
        if g.board.nunique() < 4:
            continue
        m = smf.ols('np.log(score.clip(lower=1)) ~ np.log(cells)', g).fit(cov_type='HC3')
        sc.append(dict(objective=obj, solver=s, boards=g.board.nunique(), exponent=m.params.iloc[1],
                       ci_lo=m.conf_int().iloc[1, 0], ci_hi=m.conf_int().iloc[1, 1], R2=m.rsquared))
sc = pd.DataFrame(sc); sc.to_csv(OUT + 'scaling_exponents.csv', index=False)
print(sc.round(3).to_string())

# 4) design-variable regressions (reframed RQ regression)
reg_rows = []
def report(name, m, obj):
    t = pd.DataFrame({'coef': m.params, 'se': m.bse, 't': m.tvalues, 'p': m.pvalues,
                      'ci_lo': m.conf_int()[0], 'ci_hi': m.conf_int()[1]})
    t['model'] = name; t['objective'] = obj; t['R2'] = m.rsquared; t['adjR2'] = m.rsquared_adj
    t['RMSE_log'] = np.sqrt(m.mse_resid); t['n'] = int(m.nobs)
    reg_rows.append(t.reset_index().rename(columns={'index': 'term'}))
    print(f'--- {name} obj{obj}: R2={m.rsquared:.3f} n={int(m.nobs)}')
    print(t[['coef', 'se', 't', 'p']].round(4).to_string())

for obj, d in ((1, d1), (2, d2)):
    d = d.copy(); d['ls'] = np.log(d.score.clip(lower=1)); d['lc'] = np.log(d.cells)
    # tree search: expectimax/minimax (+greedy as 1-ply of both)
    t = d[d.family.isin(['expectimax', 'minimax']) & (d.solver != 'minimax_d5_full')].copy()
    t['plies'] = np.ceil(t.depth / 2); t['even'] = (t.depth % 2 == 0).astype(int)
    t['expecti'] = (t.family == 'expectimax').astype(int)
    m = smf.ols('ls ~ lc * plies + even + expecti + expecti:plies', t).fit(cov_type='HC3')
    report('tree_search', m, obj)
    # MCTS: rollout budget
    mc = d[d.solver.isin(['mcts_60', 'mcts_300', 'mcts_1000'])].copy()
    mc['lr'] = np.log(mc.rollouts)
    m = smf.ols('ls ~ lc * lr', mc).fit(cov_type='HC3')
    report('mcts', m, obj)
    # all solvers: how much of the variance does the solver identity explain vs board?
    def r2_groups(keys):
        fitted = d.groupby(keys)['ls'].transform('mean')
        return 1 - ((d.ls - fitted) ** 2).sum() / ((d.ls - d.ls.mean()) ** 2).sum()
    rb = r2_groups(['board']); rbs = r2_groups(['board', 'solver'])
    ma = smf.ols('ls ~ C(board) + C(solver)', d[['ls', 'board', 'solver']]).fit()
    print(f'obj{obj} variance explained (log score): board only {rb:.3f}, board+solver additive {ma.rsquared:.3f}, board x solver cells {rbs:.3f}')
    reg_rows.append(pd.DataFrame([dict(model='variance_decomp', objective=obj, term='board_only', R2=rb),
                                  dict(model='variance_decomp', objective=obj, term='board+solver', R2=ma.rsquared),
                                  dict(model='variance_decomp', objective=obj, term='board*solver', R2=rbs)]))
pd.concat(reg_rows).to_csv(OUT + 'design_regressions.csv', index=False)

# 5) objective contrast: does a solver's strength transfer? improvement factor vs random
c = S.pivot_table(index=['board', 'solver'], columns='objective', values='ratio_vs_random').dropna().reset_index()
c.columns = ['board', 'solver', 'max_ratio', 'min_ratio']
c['max_gain'] = c.max_ratio  # x random (higher better)
c['min_gain'] = 1 / c.min_ratio  # x random reduction (higher better)
c.to_csv(OUT + 'objective_contrast.csv', index=False)
ct = []
for b in BOARDS:
    cb = c[c.board == b]
    rho, p = stats.spearmanr(cb.max_gain, cb.min_gain)
    ct.append(dict(board=b, n=len(cb), spearman_rho=rho, p=p,
                   best_obj1=cb.loc[cb.max_gain.idxmax(), 'solver'], best_obj2=cb.loc[cb.min_gain.idxmax(), 'solver'],
                   top_gain_obj1=cb.max_gain.max(), top_gain_obj2=cb.min_gain.max()))
ct = pd.DataFrame(ct); ct.to_csv(OUT + 'objective_contrast_by_board.csv', index=False); print(ct.round(3).to_string())
# family ranks under each objective
fam = S.assign(fam=S.solver.str.replace(r'_d\d+.*|_\d+.*', '', regex=True))
print(fam.groupby(['objective', 'fam'])['rank'].mean().unstack(0).round(2))

# 6) minimax 2-only adversary vs full (2 and 4) at d5, obj1
mf = []
for b in ['3x3', '3x4', '4x3', '4x4']:
    x = d1[(d1.solver == 'minimax_d5') & (d1.board == b)].score; y = d1[(d1.solver == 'minimax_d5_full') & (d1.board == b)].score
    U, p = stats.mannwhitneyu(y, x)
    ts = [d1[(d1.solver == s) & (d1.board == b)].sec_per_move.median() for s in ('minimax_d5', 'minimax_d5_full')]
    mf.append(dict(board=b, mean_2only=x.mean(), mean_full=y.mean(), pct=100 * (y.mean() / x.mean() - 1), p=p,
                   r=2 * U / (len(x) * len(y)) - 1, sec_move_2only=ts[0], sec_move_full=ts[1]))
mf = pd.DataFrame(mf); mf.to_csv(OUT + 'minimax_full_vs_2only.csv', index=False); print(mf.round(4).to_string())

# 7) expectimax vs minimax at matched depth
em = []
for obj, d in ((1, d1), (2, d2)):
    for b in BOARDS:
        for dep in range(2, 9):
            x = d[(d.solver == f'expectimax_d{dep}') & (d.board == b)].score
            y = d[(d.solver == f'minimax_d{dep}') & (d.board == b)].score
            if len(x) and len(y):
                U, p = stats.mannwhitneyu(x, y)
                em.append(dict(objective=obj, board=b, depth=dep, mean_expecti=x.mean(), mean_minimax=y.mean(),
                               pct=100 * (x.mean() / y.mean() - 1), p=p, r=2 * U / (len(x) * len(y)) - 1))
em = pd.DataFrame(em); em.to_csv(OUT + 'expectimax_vs_minimax.csv', index=False)
print(em.pivot_table(index=['objective', 'depth'], columns='board', values='pct')[BOARDS].round(1).to_string())

# 8) outcome-variable relationships within games (benchmark variables we do have)
oc = []
for obj, d in ((1, d1), (2, d2)):
    for b in BOARDS:
        g = d[d.board == b]
        r = dict(objective=obj, board=b)
        for v in ('moves', 'board_sum', 'max_tile'):
            r['spearman_' + v] = stats.spearmanr(g.score, g[v])[0]
            r['pearson_' + v] = stats.pearsonr(g.score, g[v])[0]
        # within-solver (removes between-solver differences)
        r['within_solver_spearman_moves'] = g.groupby('solver').apply(lambda x: stats.spearmanr(x.score, x.moves)[0], include_groups=False).median()
        m = smf.ols('np.log(score.clip(lower=1)) ~ np.log(moves.clip(lower=1)) + np.log(max_tile)', g).fit()
        r['R2_moves_maxtile'] = m.rsquared
        oc.append(r)
oc = pd.DataFrame(oc); oc.to_csv(OUT + 'outcome_correlations.csv', index=False); print(oc.round(3).to_string())

# 9) cost-performance (median seconds per move). note: hardware was heterogeneous.
cost = S[['objective', 'board', 'solver', 'mean', 'median_sec_per_move', 'rank']]
cost.to_csv(OUT + 'cost.csv', index=False)
print(cost[(cost.board == '4x4')].sort_values(['objective', 'rank']).round(5).to_string())
