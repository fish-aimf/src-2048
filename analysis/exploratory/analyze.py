"""First-pass statistics we ran while the data was still coming in.

Per solver and board: mean, median, 95% CI, win rate, skewness, Shapiro-Wilk,
then Kruskal-Wallis per board and pairwise tests, for both score and board_sum.
Everything is written to one file, master_results.csv.

    python analyze.py --file results.csv --exclude_timeouts
"""
import argparse
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings(
    "ignore", message="scipy.stats.shapiro: For N > 5000.*", category=UserWarning
)

MASTER_COLUMNS = [
    "table", "metric", "board", "solver", "solver_a", "solver_b",
    "n_games_scored", "n_games_attempted", "true_timeout_rate",
    "mean", "median", "ci_low", "ci_high", "mean_max_tile", "best_max_tile",
    "skewness", "shapiro_stat", "shapiro_p", "likely_normal",
    "kruskal_H", "anova_F", "p_value",
    "mannwhitney_U", "welch_t", "z",
    "mean_rank_a", "mean_rank_b", "rank",
    "raw_p", "holm_adjusted_p", "bonferroni_adjusted_p",
    "rank_biserial_r", "cohens_d", "kendalls_w",
    "significant_at_0.05",
]


def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    if len(values) < 2:
        return (float(values[0]), float(values[0])) if len(values) == 1 else (np.nan, np.nan)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_means = values[idx].mean(axis=1)
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return lo, hi


def _normality_check(scores):
    """Skewness + Shapiro-Wilk test for one (solver, board) group's scores."""
    scores = np.asarray(scores)
    n = len(scores)
    skewness = stats.skew(scores) if n >= 2 else np.nan
    if n < 3:
        return skewness, np.nan, np.nan, None
    try:
        if n > 5000:
            # Shapiro-Wilk's p-value is unreliable at this sample size; report
            # the statistic but not a p-value/verdict rather than mislead.
            w_stat, _ = stats.shapiro(scores)
            return skewness, w_stat, np.nan, None
        w_stat, p_val = stats.shapiro(scores)
        return skewness, w_stat, p_val, bool(p_val >= 0.05)
    except ValueError:
        return skewness, np.nan, np.nan, None


def summarize(scored_df, full_df, metric="score"):
    """scored_df: rows used for statistics (may have timeouts excluded)."""
    true_timeout_rate = full_df.groupby(["board", "solver"])["timed_out"].mean()
    attempted_n = full_df.groupby(["board", "solver"]).size()

    rows = []
    for (board, solver), g in scored_df.groupby(["board", "solver"]):
        values = g[metric].values
        lo, hi = bootstrap_ci(values)
        skewness, shapiro_stat, shapiro_p, likely_normal = _normality_check(values)
        rows.append({
            "board": board,
            "solver": solver,
            "n_games_scored": len(g),
            "n_games_attempted": attempted_n.get((board, solver), len(g)),
            "true_timeout_rate": true_timeout_rate.get((board, solver), 0.0),
            "mean": values.mean(),
            "median": np.median(values),
            "ci_low": lo,
            "ci_high": hi,
            "mean_max_tile": g["max_tile"].mean(),
            "best_max_tile": g["max_tile"].max(),
            "skewness": skewness,
            "shapiro_stat": shapiro_stat,
            "shapiro_p": shapiro_p,
            "likely_normal": likely_normal,
        })
    result = pd.DataFrame(rows).sort_values(["board", "mean"], ascending=[True, False])
    non_normal = result[result["likely_normal"] == False]
    if len(non_normal):
        print(f"!! NOTE: the following (solver, board) combos failed the Shapiro-Wilk")
        print(f"!! normality test (p < 0.05) on {metric} — likely skewed (common in 2048:")
        print(f"!! jumps only happen on merges, so values cluster with long right tails).")
        print(f"!! For these, trust Kruskal-Wallis/Mann-Whitney over ANOVA/Welch's t-test:")
        print(non_normal[["board", "solver", "skewness", "shapiro_p"]].to_string(index=False))
        print()
    high_timeout = result[result["true_timeout_rate"] > 0.15]
    if len(high_timeout):
        print(f"!! WARNING: the following (solver, board) combos had >15% of games time out.")
        print(f"!! Their {metric} stats are based only on the fastest-finishing games and are")
        print(f"!! likely BIASED — treat with caution, don't rank them at face value:")
        print(high_timeout[["board", "solver", "true_timeout_rate", "n_games_attempted"]]
              .to_string(index=False))
        print()
    return result


def _holm_bonferroni(raw_pvals):
    """Shared Holm-Bonferroni step-down correction."""
    order = np.argsort([p if not np.isnan(p) else 1.0 for p in raw_pvals])
    m = len(order)
    adjusted = [None] * m
    for rank, idx in enumerate(order):
        adjusted[idx] = min(1.0, raw_pvals[idx] * (m - rank)) if not np.isnan(raw_pvals[idx]) else np.nan
    return adjusted


def _cohens_d(a, b):
    """Cohen's d using the pooled SD (Welch's t-test's natural effect-size companion)."""
    a, b = np.asarray(a), np.asarray(b)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled_sd = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled_sd


def _rank_biserial(u_stat, n1, n2):
    """Rank-biserial correlation r = 1 - 2U/(n1*n2) — the non-parametric effect size
    that pairs with Mann-Whitney U (and, by extension, Dunn's test).
    """
    if n1 == 0 or n2 == 0:
        return np.nan
    return 1 - (2 * u_stat) / (n1 * n2)


def significance_tests(df, metric="score"):
    """Non-parametric (rank-based) tests: Kruskal-Wallis + pairwise Mann-Whitney U."""
    kw_rows = []
    pairwise_rows = []
    for board, g in df.groupby("board"):
        groups = {s: sub[metric].values for s, sub in g.groupby("solver")}
        solvers = list(groups.keys())
        if len(solvers) < 2:
            continue

        # Kruskal-Wallis: does solver choice matter at all on this board?
        try:
            h_stat, p_val = stats.kruskal(*groups.values())
        except ValueError:
            h_stat, p_val = np.nan, np.nan
        kw_rows.append({"board": board, "kruskal_H": h_stat, "p_value": p_val,
                         "significant_at_0.05": p_val < 0.05 if not np.isnan(p_val) else None})

        # Pairwise Mann-Whitney U with Holm-Bonferroni correction
        pairs = list(itertools.combinations(solvers, 2))
        raw_pvals = []
        u_stats = []
        for a, b in pairs:
            try:
                u_stat, p = stats.mannwhitneyu(groups[a], groups[b], alternative="two-sided")
            except ValueError:
                u_stat, p = np.nan, np.nan
            u_stats.append(u_stat)
            raw_pvals.append(p)

        adjusted = _holm_bonferroni(raw_pvals)

        for (a, b), u_stat, raw_p, adj_p in zip(pairs, u_stats, raw_pvals, adjusted):
            r = _rank_biserial(u_stat, len(groups[a]), len(groups[b])) if not np.isnan(u_stat) else np.nan
            pairwise_rows.append({
                "board": board, "solver_a": a, "solver_b": b,
                "mannwhitney_U": u_stat, "raw_p": raw_p, "holm_adjusted_p": adj_p,
                "rank_biserial_r": r,
                "significant_at_0.05": adj_p < 0.05 if adj_p is not None and not np.isnan(adj_p) else None,
            })

    return pd.DataFrame(kw_rows), pd.DataFrame(pairwise_rows)


def parametric_tests(df, metric="score"):
    """Parametric tests: one-way anova (overall, per board) + pairwise Welch's t-test
    (unequal variance) with Holm-Bonferroni correction.
    """
    anova_rows = []
    welch_rows = []
    for board, g in df.groupby("board"):
        groups = {s: sub[metric].values for s, sub in g.groupby("solver")}
        solvers = list(groups.keys())
        if len(solvers) < 2:
            continue

        # One-way anova: does solver choice matter at all on this board?
        try:
            f_stat, p_val = stats.f_oneway(*groups.values())
        except ValueError:
            f_stat, p_val = np.nan, np.nan
        anova_rows.append({"board": board, "anova_F": f_stat, "p_value": p_val,
                            "significant_at_0.05": p_val < 0.05 if not np.isnan(p_val) else None})

        # Pairwise Welch's t-test (equal_var=False) with Holm-Bonferroni correction
        pairs = list(itertools.combinations(solvers, 2))
        raw_pvals = []
        t_stats = []
        for a, b in pairs:
            try:
                t_stat, p = stats.ttest_ind(groups[a], groups[b], equal_var=False)
            except ValueError:
                t_stat, p = np.nan, np.nan
            t_stats.append(t_stat)
            raw_pvals.append(p)

        adjusted = _holm_bonferroni(raw_pvals)

        for (a, b), t_stat, raw_p, adj_p in zip(pairs, t_stats, raw_pvals, adjusted):
            d = _cohens_d(groups[a], groups[b])
            welch_rows.append({
                "board": board, "solver_a": a, "solver_b": b,
                "welch_t": t_stat, "raw_p": raw_p, "holm_adjusted_p": adj_p,
                "cohens_d": d,
                "significant_at_0.05": adj_p < 0.05 if adj_p is not None and not np.isnan(adj_p) else None,
            })

    return pd.DataFrame(anova_rows), pd.DataFrame(welch_rows)


def dunn_test(df, metric="score"):
    """Dunn's post-hoc test (Dunn, 1964) with Bonferroni correction."""
    rows = []
    for board, g in df.groupby("board"):
        solvers = sorted(g["solver"].unique())
        if len(solvers) < 2:
            continue

        N = len(g)
        g = g.copy()
        g["_rank"] = stats.rankdata(g[metric].values)
        mean_rank = g.groupby("solver")["_rank"].mean()
        n_per = g.groupby("solver").size()

        # Tie correction term shared across all pairs on this board
        _, tie_counts = np.unique(g[metric].values, return_counts=True)
        sum_t3_t = np.sum(tie_counts.astype(float) ** 3 - tie_counts)
        tie_term = sum_t3_t / (12 * (N - 1)) if N > 1 else 0.0

        pairs = list(itertools.combinations(solvers, 2))
        m = len(pairs)
        for a, b in pairs:
            n_a, n_b = n_per[a], n_per[b]
            se = np.sqrt(max(0.0, (N * (N + 1) / 12) - tie_term) * (1 / n_a + 1 / n_b))
            if se == 0:
                z, p_raw = np.nan, np.nan
            else:
                z = (mean_rank[a] - mean_rank[b]) / se
                p_raw = 2 * (1 - stats.norm.cdf(abs(z)))
            p_bonf = min(1.0, p_raw * m) if not np.isnan(p_raw) else np.nan
            rows.append({
                "board": board, "solver_a": a, "solver_b": b,
                "mean_rank_a": mean_rank[a], "mean_rank_b": mean_rank[b],
                "z": z, "raw_p": p_raw, "bonferroni_adjusted_p": p_bonf,
                "significant_at_0.05": p_bonf < 0.05 if not np.isnan(p_bonf) else None,
            })

    return pd.DataFrame(rows)


def rank_stability(summary_df, metric="score"):
    pivot = summary_df.pivot(index="solver", columns="board", values="mean")

    complete = pivot.dropna()
    dropped = set(pivot.index) - set(complete.index)
    if dropped:
        print(f"(Note: excluding {sorted(dropped)} from rank-stability analysis — "
              f"not yet run on all board sizes. Re-run once data collection finishes.)")

    ranks = complete.rank(ascending=False, axis=0)  # 1 = best, per board column, now consistent

    if ranks.shape[1] >= 2 and ranks.shape[0] >= 2:
        n, k = ranks.shape  # n solvers, k boards
        rank_sums = ranks.sum(axis=1).values
        S = np.sum((rank_sums - rank_sums.mean()) ** 2)
        W = 12 * S / (k ** 2 * (n ** 3 - n))
    else:
        W = np.nan

    return ranks, W


def _plot_by_board(summary, metric):
    try:
        import matplotlib.pyplot as plt
        boards = sorted(summary["board"].unique())
        fig, axes = plt.subplots(1, len(boards), figsize=(6 * len(boards), 9), squeeze=False)
        for ax, board in zip(axes[0], boards):
            sub = summary[summary["board"] == board].sort_values("mean", ascending=False)
            yerr = [sub["mean"] - sub["ci_low"], sub["ci_high"] - sub["mean"]]
            ax.bar(sub["solver"], sub["mean"], yerr=yerr, capsize=4)
            ax.set_title(f"Board {board}", pad=16, fontsize=13)
            ax.set_ylabel(f"Mean {metric} (95% CI)", fontsize=11)
            ax.tick_params(axis="x", rotation=45, labelsize=10)
            for label in ax.get_xticklabels():
                label.set_ha("right")
                label.set_rotation_mode("anchor")
        # Extra breathing room: bigger inter-panel gap (wspace) and more
        # vertical margin at top/bottom for titles and rotated labels.
        fig.subplots_adjust(wspace=0.55, bottom=0.22, top=0.90)
        out_path = f"{metric}_by_board.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved chart to {out_path}")
    except ImportError:
        print("(matplotlib not installed — skipping chart. pip install matplotlib to get one.)")


def _tag(df, table, metric):
    """Attach table/metric identifier columns and reindex to the shared MASTER_COLUMNS
    layout so every table's rows stack cleanly into one CSV.
    """
    if df.empty:
        df = pd.DataFrame(columns=MASTER_COLUMNS)
    df = df.copy()
    df["table"] = table
    df["metric"] = metric
    return df.reindex(columns=MASTER_COLUMNS)


def analyze_metric(df, df_raw, metric, master_rows):
    """Run the full summary/significance/rank-stability/plot pipeline for one metric
    column ('score' or 'board_sum').
    """
    print(f"\n{'=' * 70}")
    print(f"METRIC: {metric}")
    print(f"{'=' * 70}\n")

    summary = summarize(df, df_raw, metric=metric)
    print(f"=== Summary ({metric}, top of table) ===")
    print(summary.to_string(index=False))
    print()
    master_rows.append(_tag(summary, "summary", metric))

    kw_df, pairwise_df = significance_tests(df, metric=metric)
    print(f"=== Kruskal-Wallis on {metric} (does solver matter, per board?) ===")
    print(kw_df.to_string(index=False))
    print()
    master_rows.append(_tag(kw_df, "significance_overall", metric))
    master_rows.append(_tag(pairwise_df, "significance_pairwise", metric))

    dunn_df = dunn_test(df, metric=metric)
    print(f"=== Dunn's post-hoc test on {metric} (Bonferroni-corrected, formal KW follow-up) ===")
    print(dunn_df.to_string(index=False))
    print()
    master_rows.append(_tag(dunn_df, "significance_dunn_posthoc", metric))

    anova_df, welch_df = parametric_tests(df, metric=metric)
    print(f"=== One-way ANOVA on {metric} (does solver matter, per board? parametric) ===")
    print(anova_df.to_string(index=False))
    print()
    master_rows.append(_tag(anova_df, "significance_anova", metric))
    master_rows.append(_tag(welch_df, "significance_welch", metric))

    ranks, W = rank_stability(summary, metric=metric)
    print(f"=== Solver rank per board on {metric} (1 = best mean {metric}) ===")
    print(ranks.to_string())
    print(f"\nKendall's W (rank concordance across board sizes): {W:.3f}")
    print("  (1.0 = same ranking on every board size, 0 = no relationship at all)\n")
    # ranks is a solver x board pivot table — melt to long rows (one per
    # solver/board) so it stacks into the master CSV like everything else.
    ranks_long = ranks.reset_index().melt(id_vars="solver", var_name="board", value_name="rank")
    master_rows.append(_tag(ranks_long, "rank_stability", metric))
    master_rows.append(_tag(pd.DataFrame([{"kendalls_w": W}]), "rank_stability_kendalls_w", metric))

    _plot_by_board(summary, metric)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default="results.csv")
    ap.add_argument("--exclude_timeouts", action="store_true",
                     help="drop timed_out=1 rows before analysis (recommended, since a "
                          "cut-off game's score/board_sum isn't a real final outcome)")
    args = ap.parse_args()

    df = pd.read_csv(args.file)
    print(f"Loaded {len(df)} games from {args.file}")
    print(f"Solvers: {sorted(df['solver'].unique())}")
    print(f"Boards: {sorted(df['board'].unique())}")
    print(f"Overall timeout rate: {df['timed_out'].mean():.1%}\n")

    if "board_sum" not in df.columns:
        print("!! 'board_sum' column not found in this CSV — it was added to benchmark.py")
        print("!! after this file was generated. Re-run benchmark.py to get board_sum, or")
        print("!! this run will only analyze 'score'.\n")
        metrics = ["score"]
    else:
        metrics = ["score", "board_sum"]

    df_raw = df.copy()
    if args.exclude_timeouts:
        before = len(df)
        df = df[df["timed_out"] == 0]
        print(f"Excluded timed-out games from stats: {before} -> {len(df)} rows. "
              f"(True timeout rates are still reported per combo below.)\n")

    master_rows = []
    for metric in metrics:
        analyze_metric(df, df_raw, metric, master_rows)

    master = pd.concat(master_rows, ignore_index=True, sort=False)
    master = master.reindex(columns=MASTER_COLUMNS)
    master.to_csv("master_results.csv", index=False)
    print(f"\nSaved all tables ({', '.join(sorted(master['table'].unique()))}) "
          f"to master_results.csv ({len(master)} rows total)")


if __name__ == "__main__":
    main()
