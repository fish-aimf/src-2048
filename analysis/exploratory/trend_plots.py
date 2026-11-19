"""Mean, max and min of each solver against board size, for score and board_sum.

    python trend_plots.py --file results.csv --exclude_timeouts --out_dir plots/
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

# ---- Global look & feel (matches scatter_plots.py) ---------------------
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 220,
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.edgecolor": "0.3",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "font.family": "DejaVu Sans",
})

SIDEBAR_BG = "#f7f7f8"
_COMMA_FMT = FuncFormatter(lambda v, _: f"{v:,.0f}")


def _board_sort_key(b):
    """Natural sort for board labels: works whether `board` is a plain int/float (3, 4,
    5...) or a string like '3x3', '4x5' — sorts by the embedded numbers first,
    falling back to plain string order for anything with no digits in it.
    """
    s = str(b)
    nums = [int(n) for n in re.findall(r"\d+", s)]
    return (0, nums, s) if nums else (1, [], s)


def _solver_color_map(solvers):
    cmap = plt.get_cmap("tab10")
    return {s: cmap(i % 10) for i, s in enumerate(sorted(solvers))}


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def _apply_autoscale_y(ax, values):
    """Same reasoning as scatter_plots.py's version: a board with one outlier-heavy run
    can otherwise stretch a linear axis so far that every other board's line gets
    squashed flat.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return
    vmax = values.max()
    vmin = min(0.0, values.min())
    if vmax <= 0:
        ax.set_ylim(vmin, 1)
        return
    p90 = np.percentile(values, 90)
    if p90 > 0 and vmax > p90 * 4 and vmax > 200:
        linthresh = max(10.0, p90 * 0.1)
        ax.set_yscale("symlog", linthresh=linthresh, linscale=0.9)
        ax.yaxis.set_minor_formatter(NullFormatter())
        if vmin >= 0:
            ax.set_ylim(bottom=0)
    else:
        ax.set_ylim(vmin, vmax * 1.08)
    ax.yaxis.set_major_formatter(_COMMA_FMT)


def _trend_series(df, metric, boards, solvers):
    """Returns {solver: {'mean': [...], 'max': [...], 'min': [...], 'n': [...]}}, one
    entry per board in `boards` order (NaN where a solver has no games on that
    board).
    """
    out = {s: {"mean": [], "max": [], "min": [], "n": []} for s in solvers}
    for b in boards:
        sub = df[df["board"] == b]
        for s in solvers:
            v = sub.loc[sub["solver"] == s, metric].values
            if len(v):
                out[s]["mean"].append(v.mean())
                out[s]["max"].append(v.max())
                out[s]["min"].append(v.min())
                out[s]["n"].append(len(v))
            else:
                out[s]["mean"].append(np.nan)
                out[s]["max"].append(np.nan)
                out[s]["min"].append(np.nan)
                out[s]["n"].append(0)
    return out


def _draw_legend_sidebar(side_ax, solvers, color_map, n_total):
    side_ax.axis("off")
    side_ax.set_facecolor(SIDEBAR_BG)
    side_ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=side_ax.transAxes,
                                     facecolor=SIDEBAR_BG, edgecolor="0.8",
                                     linewidth=1.0, zorder=0))
    side_ax.text(0.08, 0.985, f"n = {n_total:,}", transform=side_ax.transAxes,
                 va="top", ha="left", fontsize=13, fontweight="bold")
    handles = [plt.Line2D([0], [0], color=color_map[s], marker="o", linestyle="-",
                           markersize=6, linewidth=2) for s in solvers]
    side_ax.legend(handles, solvers, loc="upper left", bbox_to_anchor=(0.02, 0.94),
                    frameon=False, fontsize=11, handletextpad=0.6, labelspacing=0.8)


def plot_trend(df, metric, color_map, out_dir):
    boards = sorted(df["board"].unique(), key=_board_sort_key)
    solvers = sorted(df["solver"].unique())
    series = _trend_series(df, metric, boards, solvers)
    x = np.arange(len(boards))
    n_total = len(df)

    fig_width = max(12, 1.6 * len(boards) + 8)
    fig_height = max(11, 2.6 * len(solvers) * 0.35 + 9)  # room for a tall legend

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(3, 2, width_ratios=[4.3, 1.2], height_ratios=[1, 1, 1],
                           hspace=0.32, wspace=0.05)

    panel_specs = [("mean", "Mean"), ("max", "Max (best game)"), ("min", "Min (worst game)")]
    axes = []
    for i, (stat_key, stat_label) in enumerate(panel_specs):
        ax = fig.add_subplot(gs[i, 0])
        axes.append(ax)
        all_vals = []
        for s in solvers:
            y = series[s][stat_key]
            ax.plot(x, y, color=color_map[s], marker="o", markersize=5,
                    linewidth=1.8, alpha=0.9, label=s)
            all_vals.extend([v for v in y if np.isfinite(v)])
        _apply_autoscale_y(ax, np.array(all_vals) if all_vals else np.array([0.0]))
        _style_axes(ax)
        ax.set_ylabel(f"{stat_label}\n{metric}", fontsize=12)
        if i == 0:
            ax.set_title(f"{metric} vs board size — trend across boards", fontsize=16, pad=12)
        ax.set_xticks(x)
        if i == len(panel_specs) - 1:
            ax.set_xticklabels([str(b) for b in boards], rotation=20, ha="right")
            ax.set_xlabel("board size", fontsize=13)
        else:
            ax.set_xticklabels([])
        ax.set_xlim(-0.4, len(boards) - 0.6)

    side_ax = fig.add_subplot(gs[:, 1])
    _draw_legend_sidebar(side_ax, solvers, color_map, n_total)

    path = os.path.join(out_dir, f"trend_{metric}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default="results.csv")
    ap.add_argument("--out_dir", type=str, default="trend_plots",
                     help="directory to write PNGs into (created if missing)")
    ap.add_argument("--exclude_timeouts", action="store_true",
                     help="drop timed_out=1 rows before plotting (same flag/behavior as analyze.py/scatter_plots.py)")
    args = ap.parse_args()

    df = pd.read_csv(args.file)
    print(f"Loaded {len(df)} games from {args.file}")
    print(f"Solvers: {sorted(df['solver'].unique())}")
    print(f"Boards: {sorted(df['board'].unique(), key=_board_sort_key)}")

    if args.exclude_timeouts:
        before = len(df)
        df = df[df["timed_out"] == 0]
        print(f"Excluded timed-out games: {before} -> {len(df)} rows")

    metrics = ["score"]
    if "board_sum" in df.columns:
        metrics.append("board_sum")
    else:
        print("!! 'board_sum' column not found — only plotting 'score'.")

    os.makedirs(args.out_dir, exist_ok=True)
    color_map = _solver_color_map(df["solver"].unique())

    for metric in metrics:
        path = plot_trend(df, metric, color_map, args.out_dir)
        print(f"Saved {path}")

    print(f"\nAll PNGs written to {os.path.abspath(args.out_dir)}/")


if __name__ == "__main__":
    main()
