# 2048 solver benchmark across board sizes

Code and data for a benchmark of search-based and heuristic 2048 players on seven
board sizes (3x3 to 5x5) under two objectives:

- **Objective 1** - maximise the final score
- **Objective 2** - minimise the final score (same algorithms, heuristic and merge reward negated)

In total 1,407,428 games were recorded, normally 5,000 per solver and board
(1,367,403 remain after removing duplicate `_tt` runs and one 25-game cell). Every game is seeded, so the starting board and the stream of
random numbers used for tile spawns are the same for every solver.

## Repository layout

```
src/
  common/                 game engine and the function that plays one game
    engine.py             board, moves, spawns (NumPy + Numba)
    core.py               run_one_game(), job list, results columns
  objective1_maximise/
    solvers.py            all solvers for Objective 1
    mcts_tpu.py           TPU version of Monte Carlo search (MCTS-3000)
    demo_visualizer.py    watch one solver play, move by move
  objective2_minimise/
    solvers.py            all solvers for Objective 2 (sign-flipped)
    demo_visualizer.py    same, for Objective 2
  run/                    how the games were actually run
    paths.py              picks the objective (OBJECTIVE=1 or 2)
    benchmark.py          single machine, process pool
    server.py             job server for running on several machines
    worker.py             worker that pulls games from server.py
    tpu_worker.py         batched TPU worker for MCTS-3000
    check_progress.py     progress per solver and board
    kaggle/               workers on Kaggle CPU notebooks
  neural_network/         Objective 3 prototype (DQN that minimises score)
analysis/                 statistics, figures and tables used in the paper
  data_cleaning/          one-off clean-up scripts used during data collection
  exploratory/            quick checks we ran while the data was coming in
data/                     raw results for both objectives (gzipped CSV)
results/                  final figures and LaTeX tables from the paper
```

## Solvers

| Family | Names | Notes |
| --- | --- | --- |
| Random | `random` | uniform over legal moves |
| Greedy | `greedy` | best heuristic one move ahead |
| Expectimax | `expectimax_d2` ... `expectimax_d8` (Obj. 1), `expectimax_d2` ... `expectimax_d10_tt` (Obj. 2) | spawns averaged with their real probabilities |
| Minimax | `minimax_d2` ... `minimax_d8`, plus `minimax_d5_full` (Obj. 1) | alpha-beta; spawn treated as an adversary placing 2s (`_full` also tries 4s) |
| Monte Carlo | `mcts_60`, `mcts_300`, `mcts_1000`, plus `mcts_3000_tpu` (Obj. 1) | flat Monte Carlo: greedy rollouts of up to 40 moves, best average wins |

Search depth counts plies (player move, spawn, player move, ...), so depth *d*
plans ceil(*d*/2) player moves. Names ending in `_tt` use a transposition table
(a cache); they play exactly the same moves as the plain version.

The heuristic used by every non-random solver is

```
h = 270*empty + mono + 0.1*smooth + 3*corner + maxtile
```

computed on log2 tile values (mono and smooth are penalties, corner is the max
tile's exponent if it sits in a corner).

## Running games

Install the requirements, then choose the objective with an environment variable:

```
pip install -r requirements.txt

# Objective 1, one machine
OBJECTIVE=1 python src/run/benchmark.py --games 5000 --workers 16 \
    --boards 3x3,3x4,4x3,3x5,4x4,4x5,5x5 --solvers greedy,expectimax_d3 --out results.csv

# Objective 2
OBJECTIVE=2 python src/run/benchmark.py --games 5000 --workers 16 --out results_min.csv
```

On Windows use `set OBJECTIVE=2` first. Re-running with the same `--out` file
skips games that are already done.

### How we actually ran it

For the full runs, one laptop acted only as a coordinator. It ran the job server
and played no games itself:

```
OBJECTIVE=1 python src/run/server.py --games 5000 --out results.csv \
    --duty_cycle 0.4 --job_timeout 80000 --priority idle --threads 16 \
    --boards 3x3,3x4,4x3,4x4,3x5,4x5,5x5 \
    --solvers mcts_60,mcts_300,mcts_1000,expectimax_d6_tt,expectimax_d7_tt,expectimax_d8_tt,minimax_d2,minimax_d3,minimax_d4,minimax_d5,minimax_d6,minimax_d7,minimax_d8,minimax_d5_full,mcts_3000_tpu
```

- `--threads` is the number of HTTP threads, not game processes.
- `--duty_cycle 0.4` throttles the workers so the machines stay usable.
- `--priority idle` keeps the server process out of the way.

The server listens on port 8765. To let workers outside the local network reach
it, we exposed that port with `ngrok http 8765` (a Cloudflare tunnel also works).
Other machines then ran `src/run/worker.py --server <address>`.

Workers take games in batches and send heartbeats. If a worker stops responding,
the server puts its games back in the queue.

The long Objective 1 games ran on Kaggle CPU notebooks through `src/run/kaggle/`.
MCTS-3000 ran on a TPU through `tpu_worker.py`.

Objective 2 was run the same way, with `OBJECTIVE=2` and its own solver list. That
list included some TPU Monte Carlo variants that were later dropped from the
study. `analysis/data_cleaning/purge_results.py results.csv` removed them, along
with plain `expectimax_d8` on 5x5, which was replaced by `expectimax_d8_tt`.

Machines running for days must not go to sleep. On Windows we used these
commands from an administrator prompt:

```
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 0
powercfg /change hibernate-timeout-ac 0
powercfg /change hibernate-timeout-dc 0
powercfg /h off
powercfg /change monitor-timeout-ac 20
powercfg /change monitor-timeout-dc 20
```

We also turned off the screensaver with `control desk.cpl,,@screensaver`.

Because the games ran on many different machines, the `seconds` column should
not be used to compare solvers.

### Watching a solver play

```
python src/objective1_maximise/demo_visualizer.py
python src/objective2_minimise/demo_visualizer.py
```

This opens a window that plays one game slowly and logs the values the solver
compares at each move. It needs tkinter.

## Reproducibility

Games are deterministic given the seed for every solver except `random` and the
Monte Carlo solvers, whose move choices use unseeded random numbers. As a check,
96 games (both objectives, several solvers and boards, including the `_tt`
versions) were replayed with the code in this repository; all 96 match the
recorded score and number of moves exactly.

## Data

See [data/README.md](data/README.md) for the columns and cleaning notes.

## Analysis

```
cd analysis
python clean.py
python stats1.py
python stats2.py
python stats3.py
python figs.py
python tex.py
python paper_figs.py
```

Outputs go to `analysis/out/` (statistics as CSV), `analysis/fig/`,
`analysis/tex/` and `analysis/paper/`. The full run takes about 3-4 minutes.
`results/` holds the figures and tables exactly as used in the paper.

`analysis/exploratory/` holds the quicker scripts we used to check results while
the data was still coming in. They also accept the `.csv.gz` files:

```
python analysis/exploratory/analyze.py --file results.csv --exclude_timeouts
python analysis/exploratory/scatter_plots.py --file results.csv --exclude_timeouts --out_dir plots/
python analysis/exploratory/trend_plots.py --file results.csv --exclude_timeouts --out_dir plots/
```

## Neural network (Objective 3)

`src/neural_network/` contains the environment, the dueling CNN and the scripts
used to evaluate saved checkpoints. It is a prototype and not part of the main
results. The training script and checkpoints are not included.
