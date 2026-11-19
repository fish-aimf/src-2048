# Data

One row per game. Read directly with `pandas.read_csv("objective1_results.csv.gz")`.

| Column | Meaning |
| --- | --- |
| `solver` | solver name (see the main README) |
| `board` | width x height, e.g. `4x3` |
| `seed` | game number; the same seed gives every solver the same start and spawn stream |
| `score` | final score (sum of merged tile values) |
| `board_sum` | sum of tile values on the final board |
| `max_tile` | largest tile reached |
| `moves` | number of moves played |
| `seconds` | wall-clock time of the game (hardware varied, see main README) |
| `timed_out` | 1 if the game hit a time limit; 0 for every row here |

## objective1_results.csv.gz

672,428 games, exactly as recorded.

- `expectimax_d6`/`expectimax_d6_tt` and `expectimax_d7`/`expectimax_d7_tt` were both
  run on some boards. They replay the same games seed for seed, so the analysis keeps
  one row per (solver, board, seed).
- `expectimax_d8` on 4x4 has only 25 games and is dropped in the analysis.
- A few expensive combinations have fewer than 5,000 games: on 5x5 `expectimax_d7`
  (500), `mcts_1000` (604) and `mcts_300` (4,301); `mcts_3000_tpu` on 3x4 (1,998).

## objective2_results.csv.gz

735,000 games: 21 solvers x 7 boards x 5,000 seeds.

This is the clean copy of the Objective 2 results. The working file was later
appended to by mistake with 31,651 Objective 1 (maximising) rows; this copy is the
backup taken before that and was checked row by row against the working file.
`expectimax_d8_tt` appears on 5x5 only, where it replaced plain `expectimax_d8`.
