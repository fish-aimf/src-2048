"""Watch one game of any Objective 2 solver, slowed down, with each move explained.

Same as the Objective 1 demo but with the flipped objective (-gained and the
negated heuristic). Not used to collect data.

    python src/objective2_minimise/demo_visualizer.py
"""
import math
import queue
import random
import threading
import time
import tkinter as tk
from tkinter import ttk

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from engine import Game, apply_move, empty_cells, DIRECTIONS
from solvers import heuristic_eval, expectimax_value, minimax_value, random_rollout, SOLVERS

# ---------------------------------------------------------------------------

def demo_random(game, log, think_pause):
    moves = game.legal_moves()
    if not moves:
        return None
    log(f"random: legal moves = {moves}")
    think_pause()
    d = random.choice(moves)
    log(f"random: rolled -> {d}")
    return d


def demo_greedy(game, log, think_pause):
    moves = game.legal_moves()
    if not moves:
        return None
    best, best_val = None, -math.inf
    for d in moves:
        new_grid, _, _ = apply_move(game.grid, d)
        val = heuristic_eval(new_grid)
        log(f"greedy: {d:>5} -> heuristic {val:9.1f}")
        think_pause()
        if val > best_val:
            best_val, best = val, d
    log(f"greedy: chose {best} (heuristic {best_val:.1f})")
    return best


def make_demo_expectimax(depth):
    def fn(game, log, think_pause):
        moves = game.legal_moves()
        if not moves:
            return None
        log(f"expectimax(d={depth}): evaluating {len(moves)} candidates...")
        best, best_val = None, -math.inf
        for d in moves:
            new_grid, gained, _ = apply_move(game.grid, d)
            v = -gained + expectimax_value(new_grid, depth - 1, False)
            log(f"expectimax: {d:>5} -> value {v:9.1f}  (immediate gain {gained})")
            think_pause()
            if v > best_val:
                best_val, best = v, d
        log(f"expectimax: chose {best} (value {best_val:.1f})")
        return best
    return fn


def make_demo_minimax(depth, consider_four_spawn=False):
    def fn(game, log, think_pause):
        moves = game.legal_moves()
        if not moves:
            return None
        tag = f"minimax(d={depth}{', +4-spawn' if consider_four_spawn else ''})"
        log(f"{tag}: evaluating {len(moves)} candidates (alpha-beta, "
            f"adversarial spawn)...")
        best, best_val = None, -math.inf
        alpha, beta = -math.inf, math.inf
        for d in moves:
            new_grid, gained, _ = apply_move(game.grid, d)
            v = -gained + minimax_value(new_grid, depth - 1, False, alpha, beta, consider_four_spawn)
            log(f"minimax: {d:>5} -> value {v:9.1f}  (immediate gain {gained}, alpha={alpha:.1f})")
            think_pause()
            if v > best_val:
                best_val, best = v, d
            if best_val > alpha:
                alpha = best_val
        log(f"minimax: chose {best} (value {best_val:.1f})")
        return best
    return fn


def make_demo_mcts(iterations):
    tag = "mcts"

    def fn(game, log, think_pause):
        moves = game.legal_moves()
        if not moves:
            return None
        per_move = max(1, iterations // len(moves))
        log(f"{tag}: {per_move} rollouts/move x {len(moves)} moves = "
            f"{per_move * len(moves)} total simulations")
        scores = {}
        report_every = max(1, per_move // 4)
        for d in moves:
            new_grid, gained, _ = apply_move(game.grid, d)
            total = 0.0
            for it in range(per_move):
                cells = empty_cells(new_grid)
                if cells:
                    r, c = random.choice(cells)
                    g = new_grid.copy()
                    g[r, c] = 1 if random.random() < 0.9 else 2
                else:
                    g = new_grid
                total += -gained + random_rollout(g)
                if (it + 1) % report_every == 0 or it == per_move - 1:
                    log(f"{tag}: {d:>5} rollout {it + 1:>4}/{per_move}  "
                        f"running avg {total / (it + 1):8.1f}")
            scores[d] = total / per_move
            think_pause()
        best = max(scores, key=scores.get)
        log(f"{tag}: chose {best} (avg score {scores[best]:.1f})")
        return best
    return fn


DEMO_SOLVERS = {
    "random": lambda: demo_random,
    "greedy": lambda: demo_greedy,

    "expectimax_d2": lambda: make_demo_expectimax(2),
    "expectimax_d3": lambda: make_demo_expectimax(3),
    "expectimax_d4": lambda: make_demo_expectimax(4),
    "expectimax_d5": lambda: make_demo_expectimax(5),
    "expectimax_d6": lambda: make_demo_expectimax(6),
    "expectimax_d7": lambda: make_demo_expectimax(7),
    "expectimax_d8": lambda: make_demo_expectimax(8),
    "expectimax_d8_tt": lambda: make_demo_expectimax(8),
    "expectimax_d9_tt": lambda: make_demo_expectimax(9),
    "expectimax_d10_tt": lambda: make_demo_expectimax(10),

    "minimax_d2": lambda: make_demo_minimax(2),
    "minimax_d3": lambda: make_demo_minimax(3),
    "minimax_d4": lambda: make_demo_minimax(4),
    "minimax_d5": lambda: make_demo_minimax(5),
    "minimax_d6": lambda: make_demo_minimax(6),
    "minimax_d7": lambda: make_demo_minimax(7),
    "minimax_d8": lambda: make_demo_minimax(8),
    "minimax_d5_full": lambda: make_demo_minimax(5, consider_four_spawn=True),

    "mcts_60": lambda: make_demo_mcts(60),
    "mcts_300": lambda: make_demo_mcts(300),
    "mcts_1000": lambda: make_demo_mcts(1000),


}

assert set(DEMO_SOLVERS.keys()) == set(SOLVERS.keys()), \
    "DEMO_SOLVERS must mirror SOLVERS in solvers.py"

# ---------------------------------------------------------------------------
TILE_COLORS = {
    0: ("#cdc1b4", "#cdc1b4"),
    1: ("#eee4da", "#776e65"),
    2: ("#ede0c8", "#776e65"),
    3: ("#f2b179", "#f9f6f2"),
    4: ("#f59563", "#f9f6f2"),
    5: ("#f67c5f", "#f9f6f2"),
    6: ("#f65e3b", "#f9f6f2"),
    7: ("#edcf72", "#f9f6f2"),
    8: ("#edcc61", "#f9f6f2"),
    9: ("#edc850", "#f9f6f2"),
    10: ("#edc53f", "#f9f6f2"),
    11: ("#edc22e", "#f9f6f2"),
}
DEFAULT_TILE_COLOR = ("#3c3a32", "#f9f6f2")
BOARD_BG = "#bbada0"
CELL_SIZE = 90
CELL_PAD = 8


class Demo2048App:
    def __init__(self, root):
        self.root = root
        self.root.title("2048 Solver Demo")
        self.root.configure(bg="#faf8ef")

        self.game = None
        self.worker_thread = None
        self.running = False  # auto-play loop active
        self.stop_requested = False
        self.msg_queue = queue.Queue()
        self.history = []  # snapshots for Undo: (grid, score, moves_made)
        self.history_lock = threading.Lock()

        self._build_ui()
        self._new_game()
        self.root.after(50, self._poll_queue)

    # ---------------------------------------------------------- UI layout
    def _build_ui(self):
        controls = ttk.Frame(self.root, padding=10)
        controls.grid(row=0, column=0, sticky="nsew")

        ttk.Label(controls, text="Board width").grid(row=0, column=0, sticky="w")
        self.width_var = tk.IntVar(value=4)
        ttk.Spinbox(controls, from_=2, to=8, width=4, textvariable=self.width_var)\
            .grid(row=0, column=1, padx=4)

        ttk.Label(controls, text="Board height").grid(row=0, column=2, sticky="w")
        self.height_var = tk.IntVar(value=4)
        ttk.Spinbox(controls, from_=2, to=8, width=4, textvariable=self.height_var)\
            .grid(row=0, column=3, padx=4)

        ttk.Label(controls, text="Solver").grid(row=0, column=4, sticky="w", padx=(16, 0))
        self.solver_var = tk.StringVar(value="expectimax_d3")
        solver_menu = ttk.Combobox(controls, textvariable=self.solver_var,
                                    values=list(SOLVERS.keys()), state="readonly", width=16)
        solver_menu.grid(row=0, column=5, padx=4)

        ttk.Label(controls, text="Speed (moves/sec)").grid(row=0, column=6, sticky="w", padx=(16, 0))
        self.speed_var = tk.DoubleVar(value=2.0)
        speed_scale = ttk.Scale(controls, from_=0.2, to=60.0, orient="horizontal",
                                 variable=self.speed_var, length=140,
                                 command=lambda _v: self._update_speed_label())
        speed_scale.grid(row=0, column=7, padx=4)
        self.speed_lbl = ttk.Label(controls, text="2.0", width=5)
        self.speed_lbl.grid(row=0, column=8)

        self.turbo_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="Turbo (no pauses)", variable=self.turbo_var)\
            .grid(row=0, column=9, padx=(12, 0))

        self.autoreset_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="Auto-restart on game over", variable=self.autoreset_var)\
            .grid(row=0, column=10, padx=(12, 0))

        btns = ttk.Frame(self.root, padding=(10, 0))
        btns.grid(row=1, column=0, sticky="nsew")
        self.start_btn = ttk.Button(btns, text="Start", command=self._on_start)
        self.start_btn.grid(row=0, column=0, padx=4)
        self.pause_btn = ttk.Button(btns, text="Pause", command=self._on_pause, state="disabled")
        self.pause_btn.grid(row=0, column=1, padx=4)
        self.step_btn = ttk.Button(btns, text="Step once", command=self._on_step)
        self.step_btn.grid(row=0, column=2, padx=4)
        self.undo_btn = ttk.Button(btns, text="Undo", command=self._on_undo, state="disabled")
        self.undo_btn.grid(row=0, column=3, padx=4)
        self.reset_btn = ttk.Button(btns, text="New game", command=self._on_reset)
        self.reset_btn.grid(row=0, column=4, padx=4)

        # Body: board on the left, console on the right
        body = ttk.Frame(self.root, padding=10)
        body.grid(row=2, column=0, sticky="nsew")

        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="n")

        stats = ttk.Frame(left)
        stats.pack(pady=(0, 8))
        self.score_lbl = ttk.Label(stats, text="Score: 0", font=("Segoe UI", 14, "bold"))
        self.score_lbl.grid(row=0, column=0, padx=10)
        self.maxtile_lbl = ttk.Label(stats, text="Max tile: -", font=("Segoe UI", 14))
        self.maxtile_lbl.grid(row=0, column=1, padx=10)
        self.moves_lbl = ttk.Label(stats, text="Moves: 0", font=("Segoe UI", 14))
        self.moves_lbl.grid(row=0, column=2, padx=10)
        self.status_lbl = ttk.Label(stats, text="idle", font=("Segoe UI", 14), foreground="#888")
        self.status_lbl.grid(row=0, column=3, padx=10)

        self.canvas = tk.Canvas(left, bg=BOARD_BG, highlightthickness=0)
        self.canvas.pack()

        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="n", padx=(16, 0))
        ttk.Label(right, text="Solver thinking...", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        console_frame = ttk.Frame(right)
        console_frame.pack()
        self.console = tk.Text(console_frame, width=64, height=28, bg="#1e1e1e", fg="#d4d4d4",
                                insertbackground="#d4d4d4", font=("Consolas", 10), state="disabled")
        scrollbar = ttk.Scrollbar(console_frame, command=self.console.yview)
        self.console["yscrollcommand"] = scrollbar.set
        self.console.grid(row=0, column=0)
        scrollbar.grid(row=0, column=1, sticky="ns")

    # ------------------------------------------------------- game control
    def _new_game(self):
        self._stop_worker()
        w, h = self.width_var.get(), self.height_var.get()
        self.game = Game(width=w, height=h)
        with self.history_lock:
            self.history = []
        self.undo_btn.config(state="disabled")
        self._clear_console()
        self._log(f"New game: {w}x{h} board, solver={self.solver_var.get()}")
        self._resize_canvas(w, h)
        self._draw_board()
        self._update_stats()
        self.status_lbl.config(text="idle", foreground="#888")

    def _resize_canvas(self, w, h):
        self.canvas.config(width=w * CELL_SIZE + CELL_PAD,
                            height=h * CELL_SIZE + CELL_PAD)

    def _on_reset(self):
        self._new_game()

    def _on_start(self):
        if self.running:
            return
        self.running = True
        self.stop_requested = False
        self.start_btn.config(state="disabled")
        self.pause_btn.config(state="normal")
        self.undo_btn.config(state="disabled")
        self.status_lbl.config(text="running", foreground="#2a8f2a")
        self._launch_worker(loop=True)

    def _on_pause(self):
        self.stop_requested = True
        self.pause_btn.config(state="disabled")
        self.start_btn.config(state="normal")
        self.status_lbl.config(text="paused", foreground="#c98a00")

    def _on_step(self):
        if self.running:
            return
        self._launch_worker(loop=False)

    def _on_undo(self):
        if self.running:
            return
        with self.history_lock:
            if not self.history:
                return
            grid, score, moves_made = self.history.pop()
            remaining = len(self.history)
        self.game.grid = grid
        self.game.score = score
        self.game.moves_made = moves_made
        self._log(f"<< undo: back to move {moves_made}")
        self._draw_board()
        self._update_stats()
        self.status_lbl.config(text="idle", foreground="#888")
        if remaining == 0:
            self.undo_btn.config(state="disabled")

    def _update_speed_label(self):
        self.speed_lbl.config(text=f"{self.speed_var.get():.1f}")

    def _stop_worker(self):
        self.stop_requested = True
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        self.start_btn.config(state="normal")
        self.pause_btn.config(state="disabled")

    def _launch_worker(self, loop):
        if self.worker_thread and self.worker_thread.is_alive():
            return
        self.stop_requested = False
        self.worker_thread = threading.Thread(
            target=self._worker_run, args=(loop,), daemon=True)
        self.worker_thread.start()

    # ------------------------------------------------------- worker thread
    def _autoreset_game(self, log):
        """Called from the worker thread when the game just ended and 'Auto-restart on
        game over' is checked and we're in an auto-play loop.
        """
        final_score = self.game.score
        final_tile = self.game.max_tile_value()
        w, h = self.width_var.get(), self.height_var.get()
        log(f"=== GAME OVER (score={final_score}, max tile={final_tile}) — auto-restarting ===")
        with self.history_lock:
            self.history = []
        self.game = Game(width=w, height=h)
        self.msg_queue.put(("autoreset", None))

    def _worker_run(self, loop):
        solver_name = self.solver_var.get()
        demo_policy = DEMO_SOLVERS[solver_name]()

        def log(msg):
            self.msg_queue.put(("log", msg))

        def think_pause():
            if self.turbo_var.get():
                return
            speed = max(0.2, self.speed_var.get())
            pause = min(0.35, 1.0 / (speed * 3.0))
            time.sleep(pause)

        while True:
            if self.stop_requested:
                break

            if self.game.is_over():
                if self.autoreset_var.get() and loop:
                    self._autoreset_game(log)
                    continue
                self.msg_queue.put(("gameover", None))
                break

            d = demo_policy(self.game, log, think_pause)
            if d is None:
                if self.autoreset_var.get() and loop:
                    self._autoreset_game(log)
                    continue
                self.msg_queue.put(("gameover", None))
                break

            with self.history_lock:
                self.history.append((self.game.grid.copy(), self.game.score, self.game.moves_made))
            self.msg_queue.put(("history_grew", None))

            self.game.step(d)
            self.msg_queue.put(("board", None))

            if not self.turbo_var.get():
                speed = max(0.2, self.speed_var.get())
                time.sleep(1.0 / speed)

            if not loop:
                break

        self.msg_queue.put(("worker_done", None))

    # ------------------------------------------------------- queue polling
    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "board":
                    self._draw_board()
                    self._update_stats()
                elif kind == "history_grew":
                    if not self.running or self.stop_requested:
                        self.undo_btn.config(state="normal")
                elif kind == "gameover":
                    self._log("=== GAME OVER ===")
                    self._draw_board()
                    self._update_stats()
                    self.status_lbl.config(text="game over", foreground="#b00")
                elif kind == "autoreset":
                    self.undo_btn.config(state="disabled")
                    self._draw_board()
                    self._update_stats()
                    self.status_lbl.config(text="running (auto-restarted)", foreground="#2a8f2a")
                elif kind == "worker_done":
                    self.running = False
                    self.start_btn.config(state="normal")
                    self.pause_btn.config(state="disabled")
                    if self.status_lbl.cget("text") == "running":
                        self.status_lbl.config(text="idle", foreground="#888")
                    if self.history:
                        self.undo_btn.config(state="normal")
        except queue.Empty:
            pass
        self.root.after(50, self._poll_queue)

    # -------------------------------------------------------------- draw
    def _draw_board(self):
        self.canvas.delete("all")
        grid = self.game.grid
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                exp = int(grid[r, c])
                bg, fg = TILE_COLORS.get(exp, DEFAULT_TILE_COLOR)
                x0 = c * CELL_SIZE + CELL_PAD // 2
                y0 = r * CELL_SIZE + CELL_PAD // 2
                x1 = x0 + CELL_SIZE - CELL_PAD
                y1 = y0 + CELL_SIZE - CELL_PAD
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=bg, outline="")
                if exp > 0:
                    val = 1 << exp
                    font_size = 24 if val < 1000 else (18 if val < 100000 else 14)
                    self.canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=str(val),
                                             fill=fg, font=("Segoe UI", font_size, "bold"))

    def _update_stats(self):
        self.score_lbl.config(text=f"Score: {self.game.score}")
        self.maxtile_lbl.config(text=f"Max tile: {self.game.max_tile_value()}")
        self.moves_lbl.config(text=f"Moves: {self.game.moves_made}")

    # ----------------------------------------------------------- console
    def _clear_console(self):
        self.console.config(state="normal")
        self.console.delete("1.0", "end")
        self.console.config(state="disabled")

    def _log(self, msg):
        self.console.config(state="normal")
        self.console.insert("end", msg + "\n")
        self.console.see("end")
        self.console.config(state="disabled")


def main():
    root = tk.Tk()
    app = Demo2048App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
