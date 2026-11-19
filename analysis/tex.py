"""Step 6. LaTeX tables (written to tex/). Needs booktabs, multirow and longtable."""
import numpy as np, pandas as pd
O = 'out/'; T = 'tex/'
import os; os.makedirs(T, exist_ok=True)
BOARDS = ['3x3', '3x4', '4x3', '3x5', '4x4', '4x5', '5x5']
S = pd.read_csv(O + 'summary.csv')
esc = lambda s: s.replace('_', r'\_')

def fmt(v):
    if v >= 1e6: return f'{v/1e6:.2f}M'
    if v >= 1e4: return f'{v/1e3:.1f}k'
    if v >= 100: return f'{v:,.0f}'.replace(',', '{,}')
    return f'{v:.1f}'

# T1/T2: mean score matrix, best per board in bold, low-n marked with dagger
for obj in (1, 2):
    s = S[S.objective == obj]
    piv = s.pivot_table(index='solver', columns='board', values='mean')[BOARDS]
    npv = s.pivot_table(index='solver', columns='board', values='n')[BOARDS]
    rk = s.pivot_table(index='solver', columns='board', values='rank')[BOARDS]
    order = rk.mean(axis=1).sort_values().index
    lines = [r'\begin{table*}[t]', r'\centering\small',
             rf'\caption{{Mean final score by solver and board, Objective {obj} ({"higher" if obj == 1 else "lower"} is better). '
             r'Bold: best on that board. $^\dagger$: fewer than 5{,}000 games (n given in Appendix). --: not run. '
             r'k = $\times10^3$, M = $\times10^6$.}',
             rf'\label{{tab:mean_obj{obj}}}', r'\begin{tabular}{l' + 'r' * 7 + '}', r'\toprule',
             'Solver & ' + ' & '.join(b.replace('x', r'$\times$') for b in BOARDS) + r' \\', r'\midrule']
    for sol in order:
        cells = []
        for b in BOARDS:
            v = piv.loc[sol, b]
            if np.isnan(v):
                cells.append('--'); continue
            c = fmt(v)
            if rk.loc[sol, b] == 1: c = r'\textbf{' + c + '}'
            if npv.loc[sol, b] < 5000: c += r'$^\dagger$'
            cells.append(c)
        lines.append(esc(sol) + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    open(f'{T}tab_mean_obj{obj}.tex', 'w').write('\n'.join(lines))

# T3: omnibus + Kendall
om = pd.read_csv(O + 'omnibus.csv'); kw = pd.read_csv(O + 'kendall_w.csv')
L = [r'\begin{table}[t]', r'\centering\small',
     r'\caption{Kruskal--Wallis test across all solvers on each board ($p<10^{-300}$ in every case). '
     r'$\varepsilon^2$: share of rank variance explained by solver identity. Levene (median-centred) rejects equal variances on every board, '
     r'so ANOVA assumptions do not hold.}', r'\label{tab:omnibus}',
     r'\resizebox{\columnwidth}{!}{\begin{tabular}{lrrrr}', r'\toprule', r'Board & \multicolumn{2}{c}{Obj.\ 1} & \multicolumn{2}{c}{Obj.\ 2} \\',
     r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}', r' & $H$ (k) & $\varepsilon^2$ & $H$ (k) & $\varepsilon^2$ \\', r'\midrule']
for b in BOARDS:
    a = om[(om.objective == 1) & (om.board == b)].iloc[0]; c = om[(om.objective == 2) & (om.board == b)].iloc[0]
    L.append(f"{b.replace('x', r'$\times$')} & {a.kruskal_H:,.0f} ({a.k}) & {a.epsilon_sq:.2f} & {c.kruskal_H:,.0f} ({c.k}) & {c.epsilon_sq:.2f} \\\\".replace(',', '{,}'))
k1, k2 = kw.iloc[0], kw.iloc[1]
L += [r'\midrule', rf"Kendall's $W$ & \multicolumn{{2}}{{c}}{{{k1.W:.3f} ($n$={int(k1.n_solvers)}, $p<10^{{-13}}$)}} & \multicolumn{{2}}{{c}}{{{k2.W:.3f} ($n$={int(k2.n_solvers)}, $p<10^{{-17}}$)}} \\",
      r'\bottomrule', r'\end{tabular}}', r'\end{table}']
open(T + 'tab_omnibus.tex', 'w').write('\n'.join(L))

# T4: depth pairing
P = pd.read_csv(O + 'depth_pairs.csv'); R = pd.read_csv(O + 'parity_regression.csv')
L = [r'\begin{table}[t]', r'\centering\small',
     r'\caption{Effect of one extra ply of search, split by whether the step adds a player move (even$\to$odd depth) '
     r'or only a spawn layer (odd$\to$even). $|r|$: mean absolute rank-biserial correlation over boards; '
     r'$\Delta$: median \% change in mean score; last column: \% change attributable to the extra spawn layer in '
     r'$\log(\text{score}) \sim \text{board}\times\text{player moves} + \text{even}$ (95\% CI, HC3 errors).}',
     r'\label{tab:pairing}', r'\resizebox{\columnwidth}{!}{\begin{tabular}{llrrrrl}', r'\toprule',
     r'Obj. & Family & \multicolumn{2}{c}{odd$\to$even} & \multicolumn{2}{c}{even$\to$odd} & Even-depth effect \\',
     r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}', r' & & $|r|$ & $\Delta$ & $|r|$ & $\Delta$ & \\', r'\midrule']
for obj in (1, 2):
    for fam in ('expectimax', 'minimax'):
        p = P[(P.objective == obj) & (P.family == fam)]
        oe = p[p.step == 'odd->even']; eo = p[p.step == 'even->odd']
        r = R[(R.objective == obj) & (R.family == fam)].iloc[0]
        L.append(f"{obj} & {fam} & {oe.r_rankbiserial.abs().mean():.3f} & {oe['pct_change'].median():+.1f}\\% & "
                 f"{eo.r_rankbiserial.abs().mean():.3f} & {eo['pct_change'].median():+.1f}\\% & "
                 f"{r.even_pct:+.1f}\\% [{r.even_ci_lo:+.1f}, {r.even_ci_hi:+.1f}] \\\\")
L += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
open(T + 'tab_pairing.tex', 'w').write('\n'.join(L))

# T5: scaling exponents (7-board solvers)
sc = pd.read_csv(O + 'scaling_exponents.csv'); sc = sc[sc.boards == 7]
w = sc.pivot_table(index='solver', columns='objective', values='exponent')
L = [r'\begin{table}[t]', r'\centering\small',
     r'\caption{Board-size scaling exponent $b$ in $\log \text{score} = a + b\log(\text{cells})$, fitted per solver on all games over the seven boards '
     r'(95\% CI half-width $\le 0.03$ for all).}', r'\label{tab:scaling}', r'\begin{tabular}{lrr}', r'\toprule',
     r'Solver & $b$ (Obj.\ 1) & $b$ (Obj.\ 2) \\', r'\midrule']
for sol in w.sort_values(1, ascending=False, na_position='last').index:
    a = w.loc[sol, 1] if 1 in w.columns else np.nan; c = w.loc[sol, 2]
    L.append(f"{esc(sol)} & {'--' if np.isnan(a) else f'{a:.2f}'} & {'--' if np.isnan(c) else f'{c:.2f}'} \\\\")
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open(T + 'tab_scaling.tex', 'w').write('\n'.join(L))

# Appendix: full per-board descriptives
L = [r'% requires \usepackage{longtable,booktabs}', r'{\footnotesize']
for obj in (1, 2):
    L += [r'\begin{longtable}{llrrrrrrr}',
          rf'\caption{{Descriptive statistics, Objective {obj}. CI: 95\% bootstrap (10{{,}}000 resamples).}}\label{{tab:desc_obj{obj}}}\\',
          r'\toprule', r'Board & Solver & $n$ & Mean & 95\% CI & Median & SD & SE & Skew \\', r'\midrule', r'\endhead']
    for b in BOARDS:
        s = S[(S.objective == obj) & (S.board == b)].sort_values('rank')
        for _, q in s.iterrows():
            L.append(f"{b.replace('x', r'$\times$')} & {esc(q.solver)} & {int(q.n)} & {q['mean']:.1f} & [{q.ci_lo:.1f}, {q.ci_hi:.1f}] & {q['median']:.0f} & {q.sd:.1f} & {q.se:.2f} & {q['skew']:.2f} \\\\")
        L.append(r'\midrule')
    L += [r'\bottomrule', r'\end{longtable}', '']
L.append('}')
open(T + 'tab_appendix_descriptives.tex', 'w').write('\n'.join(L))
print('ok')
