"""One point per game, per board and solver, for score and board_sum.

    python scatter_plots.py --file results.csv --exclude_timeouts --out_dir plots/
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

# ---- Global look & feel -----------------------------------------------
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 220,  # high-res output, no graininess
    "font.size": 13,
    "axes.titlesize": 16,
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


def _marker_size(n):
    """Smaller markers as point count grows, so dense plots stay legible instead of
    turning into a solid smear.
    """
    return float(np.clip(7000.0 / max(n, 1), 10, 220))


def _solver_color_map(solvers):
    cmap = plt.get_cmap("tab10")
    return {s: cmap(i % 10) for i, s in enumerate(sorted(solvers))}


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def _apply_autoscale_y(ax, values):
    """Scale the y-axis to how the data is actually shaped instead of plain linear
    min/max.
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
    p99 = np.percentile(values, 99)
    if p99 > 0 and vmax > p99 * 4 and vmax > 200:
        linthresh = max(10.0, p99 * 0.1)
        ax.set_yscale("symlog", linthresh=linthresh, linscale=0.9)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        if vmin >= 0:
            ax.set_ylim(bottom=0)
    else:
        ax.set_ylim(vmin, vmax * 1.05)


def _stat_lines(groups):
    """groups: dict solver -> array of metric values for that solver."""
    lines = []
    active = [s for s in sorted(groups) if len(groups[s]) > 0]
    name_w = max((len(s) for s in active), default=6)
    if active:
        lines.append(
            f"{'solver':<{name_w}}  {'mean':>9}  {'min':>8}  {'max':>9}  {'n':>8}"
        )
    for s in active:
        v = np.asarray(groups[s])
        lines.append(
            f"{s:<{name_w}}  {v.mean():>9,.1f}  {v.min():>8,.0f}  {v.max():>9,.0f}  {len(v):>8,}"
        )
    solvers = active
    if len(solvers) == 2:
        a, b = np.asarray(groups[solvers[0]]), np.asarray(groups[solvers[1]])
        try:
            u_stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            n1, n2 = len(a), len(b)
            mu = n1 * n2 / 2
            sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            z = (u_stat - mu) / sigma if sigma > 0 else np.nan
            lines.append(f"Mann-Whitney\n  Z: {z:.2f}\n  p: {p:.3g}")
        except ValueError:
            pass
    elif len(solvers) > 2:
        try:
            h, p = stats.kruskal(*[groups[s] for s in solvers])
            lines.append(f"Kruskal-Wallis\n  H: {h:.2f}\n  p: {p:.3g}")
        except ValueError:
            pass
    return lines


SIDEBAR_AXES_FRAC = 0.80
SIDEBAR_TOP_PAD_IN = 0.9  # inches of slack above/below the pure content estimate
_LINE_MULT = 1.55  # line height = fontsize_pt * _LINE_MULT / 72, in inches

_SIZES = {"header": 13.0, "legend": 11.0, "title": 12.5, "text": 10.5}


def _line_height_in(pt, scale=1.0):
    return pt * scale * _LINE_MULT / 72.0


def _sidebar_content_height_in(blocks, n_legend_items=0, header=True, scale=1.0):
    """Total physical height (inches) the sidebar content needs at the given font scale
    — used both to size the figure and, inverted, to size the scale factor once the
    figure height is capped.
    """
    total = 0.0
    if header:
        total += _line_height_in(_SIZES["header"], scale) + 0.05
    if n_legend_items:
        total += n_legend_items * _line_height_in(_SIZES["legend"], scale)
        total += _line_height_in(_SIZES["legend"], scale) * 0.6 + 0.10
    for title, lines in blocks:
        if title:
            total += _line_height_in(_SIZES["title"], scale) + 0.04
        for text in lines:
            n_sub = text.count("\n") + 1
            total += n_sub * _line_height_in(_SIZES["text"], scale) + 0.05
        total += 0.10
    return total


def _sidebar_height_and_scale(blocks, fig_width, n_legend_items=0, header=True,
                               min_height=9.5, max_aspect=0.85, min_scale=0.20):
    """Pick a figure height tall enough for the sidebar content to fit."""
    max_height = max(min_height, fig_width * max_aspect)
    natural_content_in = _sidebar_content_height_in(blocks, n_legend_items, header, scale=1.0)
    fig_height = natural_content_in / SIDEBAR_AXES_FRAC + SIDEBAR_TOP_PAD_IN
    fig_height = max(fig_height, min_height)
    if fig_height <= max_height:
        return fig_height, 1.0
    avail_content_in = (max_height - SIDEBAR_TOP_PAD_IN) * SIDEBAR_AXES_FRAC
    scale = max(min_scale, avail_content_in / natural_content_in)
    return max_height, scale


def _draw_sidebar(side_ax, blocks, fig_height_in, handles=None, labels=None, header=None, scale=1.0):
    """blocks: list of (title_or_None, list_of_text_lines) sections to stack top-to-
    bottom in the sidebar.
    """
    side_ax.axis("off")
    side_ax.set_facecolor(SIDEBAR_BG)
    side_ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=side_ax.transAxes,
                                     facecolor=SIDEBAR_BG, edgecolor="0.8",
                                     linewidth=1.0, zorder=0))
    axes_height_in = fig_height_in * SIDEBAR_AXES_FRAC

    def frac(pt):
        return _line_height_in(pt, scale) / axes_height_in

    f = lambda pts: max(6.5, pts * scale)
    y = 0.99
    if header:
        side_ax.text(0.06, y, header, transform=side_ax.transAxes, va="top",
                     ha="left", fontsize=f(_SIZES["header"]), fontweight="bold")
        y -= frac(_SIZES["header"]) + 0.05 / axes_height_in
    if handles:
        leg = side_ax.legend(handles, labels, loc="upper left",
                              bbox_to_anchor=(0.02, y), frameon=False,
                              fontsize=f(_SIZES["legend"]), handletextpad=0.5,
                              labelspacing=0.6 * scale)
        side_ax.add_artist(leg)
        y -= len(labels) * frac(_SIZES["legend"]) + frac(_SIZES["legend"]) * 0.6 + 0.10 / axes_height_in

    for title, lines in blocks:
        if title:
            side_ax.text(0.06, y, title, transform=side_ax.transAxes, va="top",
                         ha="left", fontsize=f(_SIZES["title"]), fontweight="bold", color="0.15")
            y -= frac(_SIZES["title"]) + 0.04 / axes_height_in
        for text in lines:
            n_sub = text.count("\n") + 1
            side_ax.text(0.09, y, text, transform=side_ax.transAxes, va="top",
                         ha="left", fontsize=f(_SIZES["text"]), family="monospace", color="0.2")
            y -= n_sub * frac(_SIZES["text"]) + 0.05 / axes_height_in
        y -= 0.10 / axes_height_in


def _scatter_solvers(ax, sub, metric, color_map, rng, size, edge=True):
    """Draw one jittered scatter of every game in `sub` onto `ax`, grouped by solver
    along the x-axis.
    """
    solvers = sorted(sub["solver"].unique())
    groups = {}
    n_total = len(sub)
    edge_kw = dict(edgecolors="white", linewidths=0.4) if (edge and n_total < 400) else dict(edgecolors="none")
    for i, s in enumerate(solvers):
        v = sub.loc[sub["solver"] == s, metric].values
        groups[s] = v
        if len(v) == 0:
            continue
        x = i + rng.uniform(-0.30, 0.30, size=len(v))
        ax.scatter(x, v, s=size, alpha=0.65, color=color_map[s], label=s, **edge_kw)
    ax.set_xticks(range(len(solvers)))
    ax.set_xticklabels(solvers, rotation=20, ha="right")
    ax.set_xlim(-0.6, len(solvers) - 0.4)
    return groups


def plot_scatter_single_board(df, board, metric, color_map, out_dir):
    sub = df[df["board"] == board]
    n_total = len(sub)
    size = _marker_size(n_total)
    rng = np.random.default_rng(0)
    n_solvers = sub["solver"].nunique()

    groups_preview = {s: sub.loc[sub["solver"] == s, metric].values for s in sorted(sub["solver"].unique())}
    lines = _stat_lines(groups_preview)
    fig_width = max(11, 2.0 * n_solvers + 8)
    height, scale = _sidebar_height_and_scale([(None, lines)], fig_width, n_legend_items=n_solvers)

    fig = plt.figure(figsize=(fig_width, height))
    fig.subplots_adjust(top=0.92, bottom=0.92 - SIDEBAR_AXES_FRAC)
    gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 1.3], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    side_ax = fig.add_subplot(gs[0, 1])

    groups = _scatter_solvers(ax, sub, metric, color_map, rng, size)
    _apply_autoscale_y(ax, sub[metric].values)
    _style_axes(ax)
    ax.set_ylabel(metric, fontsize=14)
    ax.set_title(f"{metric} — every game — board {board}", fontsize=17, pad=14)

    handles, labels = ax.get_legend_handles_labels()
    _draw_sidebar(side_ax, blocks=[(None, lines)], fig_height_in=height,
                  handles=handles, labels=labels, header=f"n = {n_total:,}", scale=scale)

    fig.suptitle(f"Marker size {size:.0f}px  ·  jittered by solver", fontsize=9.5, y=0.01, color="0.5")
    path = os.path.join(out_dir, f"scatter_{metric}_board_{board}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scatter_all_boards(df, metric, color_map, out_dir):
    boards = sorted(df["board"].unique())
    all_solvers = sorted(df["solver"].unique())
    rng = np.random.default_rng(0)

    # Pre-compute stats/blocks first so we know how tall the figure needs to
    # be before we build it — otherwise sidebar content overflows the canvas.
    blocks = []
    for board in boards:
        sub = df[df["board"] == board]
        groups = {s: sub.loc[sub["solver"] == s, metric].values for s in sorted(sub["solver"].unique())}
        blocks.append((f"Board {board}", _stat_lines(groups)))
    fig_width = 6.2 * len(boards) + 4.2
    height, scale = _sidebar_height_and_scale(blocks, fig_width, n_legend_items=len(all_solvers), header=True)

    fig = plt.figure(figsize=(fig_width, height))
    fig.subplots_adjust(top=0.92, bottom=0.92 - SIDEBAR_AXES_FRAC)
    width_ratios = [4] * len(boards) + [1.6]
    gs = fig.add_gridspec(1, len(boards) + 1, width_ratios=width_ratios, wspace=0.10)

    handles, labels = None, None
    for i, board in enumerate(boards):
        ax = fig.add_subplot(gs[0, i])
        sub = df[df["board"] == board]
        n_total = len(sub)
        size = _marker_size(n_total)
        _scatter_solvers(ax, sub, metric, color_map, rng, size)
        _apply_autoscale_y(ax, sub[metric].values)
        _style_axes(ax)
        ax.set_title(f"Board {board}  (n={n_total:,})", fontsize=14.5, pad=10)
        if i == 0:
            ax.set_ylabel(metric, fontsize=13.5)
            handles, labels = ax.get_legend_handles_labels()

    side_ax = fig.add_subplot(gs[0, len(boards)])
    _draw_sidebar(side_ax, blocks=blocks, fig_height_in=height,
                  handles=handles, labels=labels, header="Stats by board", scale=scale)

    fig.suptitle(f"{metric} — every game, every board size", fontsize=18, y=0.995)
    path = os.path.join(out_dir, f"scatter_{metric}_all_boards.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scatter_everything(df, metric, color_map, out_dir):
    boards = sorted(df["board"].unique())
    solvers = sorted(df["solver"].unique())
    n_total = len(df)
    size = _marker_size(n_total)
    rng = np.random.default_rng(0)

    groups_preview = {s: df.loc[df["solver"] == s, metric].values for s in solvers}
    lines_preview = _stat_lines(groups_preview)
    fig_width = max(13, 3.0 * len(boards) + 8)
    height, scale = _sidebar_height_and_scale([("All boards combined", lines_preview)], fig_width,
                                               n_legend_items=len(solvers))

    fig = plt.figure(figsize=(fig_width, height))
    fig.subplots_adjust(top=0.92, bottom=0.92 - SIDEBAR_AXES_FRAC)
    gs = fig.add_gridspec(1, 2, width_ratios=[4.4, 1.3], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    side_ax = fig.add_subplot(gs[0, 1])

    offset = {s: (i - (len(solvers) - 1) / 2) * 0.18 for i, s in enumerate(solvers)}
    edge_kw = dict(edgecolors="white", linewidths=0.3) if n_total < 400 else dict(edgecolors="none")
    for bi, board in enumerate(boards):
        sub = df[df["board"] == board]
        for s in solvers:
            v = sub.loc[sub["solver"] == s, metric].values
            if len(v) == 0:
                continue
            x = bi + offset[s] + rng.uniform(-0.06, 0.06, size=len(v))
            ax.scatter(x, v, s=size, alpha=0.55, color=color_map[s],
                       label=s if bi == 0 else None, **edge_kw)

    _apply_autoscale_y(ax, df[metric].values)
    _style_axes(ax)
    ax.set_xticks(range(len(boards)))
    ax.set_xticklabels([str(b) for b in boards], fontsize=13)
    ax.set_xlim(-0.6, len(boards) - 0.4)
    ax.set_xlabel("board size", fontsize=14)
    ax.set_ylabel(metric, fontsize=14)
    ax.set_title(f"{metric} — absolutely every game, every board, every solver", fontsize=17, pad=14)

    handles, labels = ax.get_legend_handles_labels()
    _draw_sidebar(side_ax, blocks=[("All boards combined", lines_preview)], fig_height_in=height,
                  handles=handles, labels=labels, header=f"n = {n_total:,}", scale=scale)

    fig.suptitle(f"Marker size {size:.0f}px", fontsize=9.5, y=0.01, color="0.5")
    path = os.path.join(out_dir, f"scatter_{metric}_everything.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_for_metric(df, metric, color_map, out_dir):
    print(f"\n{'=' * 70}\nMETRIC: {metric}\n{'=' * 70}")
    for board in sorted(df["board"].unique()):
        path = plot_scatter_single_board(df, board, metric, color_map, out_dir)
        print(f"Saved {path}")
    path = plot_scatter_all_boards(df, metric, color_map, out_dir)
    print(f"Saved {path}")
    path = plot_scatter_everything(df, metric, color_map, out_dir)
    print(f"Saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default="results.csv")
    ap.add_argument("--out_dir", type=str, default="scatter_plots",
                     help="directory to write PNGs into (created if missing)")
    ap.add_argument("--exclude_timeouts", action="store_true",
                     help="drop timed_out=1 rows before plotting (same flag/behavior as analyze.py)")
    args = ap.parse_args()

    df = pd.read_csv(args.file)
    print(f"Loaded {len(df)} games from {args.file}")
    print(f"Solvers: {sorted(df['solver'].unique())}")
    print(f"Boards: {sorted(df['board'].unique())}")

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
        generate_for_metric(df, metric, color_map, args.out_dir)

    print(f"\nAll PNGs written to {os.path.abspath(args.out_dir)}/")


if __name__ == "__main__":
    main()
