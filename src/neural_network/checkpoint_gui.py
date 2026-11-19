"""Small Tkinter viewer that plays a game with a chosen checkpoint."""
import argparse
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from engine import Game
from checkpoint_utils import find_checkpoints, load_checkpoint, pick_action

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
CELL_SIZE = 90
CELL_PAD = 8


class CheckpointDemoApp:
    def __init__(self, root, ckpt_dir):
        self.root = root
        self.root.title("2048 Min-Score Agent — Checkpoint Demo")
        self.root.configure(bg="#faf8ef")

        self.ckpt_dir = ckpt_dir
        self.checkpoints = find_checkpoints(ckpt_dir)  # [(step, label, path), ...]
        self.net = None
        self.board_sizes = None
        self.canvas_h = self.canvas_w = None
        self.loaded_step = None

        self.game = None
        self.worker_thread = None
        self.running = False
        self.stop_requested = False
        self.msg_queue = queue.Queue()
        self.history = []
        self.history_lock = threading.Lock()

        self._build_ui()
        if self.checkpoints:
            self.ckpt_var.set(self.checkpoints[-1][1])  # default to most-trained
            self._on_load_checkpoint()
        else:
            self._log(f"No checkpoints found in {ckpt_dir} "
                       f"(looking for step_*.pt / latest.pt)")
        self.root.after(50, self._poll_queue)

    # --------------------------------------------------------------- UI
    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack()

        controls = ttk.Frame(outer)
        controls.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(controls, text="Checkpoint:").grid(row=0, column=0, sticky="w")
        self.ckpt_var = tk.StringVar()
        ckpt_labels = [c[1] for c in self.checkpoints]
        self.ckpt_combo = ttk.Combobox(controls, textvariable=self.ckpt_var,
                                        values=ckpt_labels, state="readonly", width=16)
        self.ckpt_combo.grid(row=0, column=1, padx=(4, 12))
        self.load_btn = ttk.Button(controls, text="Load", command=self._on_load_checkpoint)
        self.load_btn.grid(row=0, column=2, padx=(0, 16))

        ttk.Label(controls, text="Board:").grid(row=0, column=3, sticky="w")
        self.board_var = tk.StringVar()
        self.board_combo = ttk.Combobox(controls, textvariable=self.board_var,
                                         values=[], state="readonly", width=6)
        self.board_combo.grid(row=0, column=4, padx=(4, 16))

        self.new_btn = ttk.Button(controls, text="New Game", command=self._new_game)
        self.new_btn.grid(row=0, column=5, padx=(0, 6))
        self.start_btn = ttk.Button(controls, text="Start", command=self._on_start)
        self.start_btn.grid(row=0, column=6, padx=(0, 6))
        self.pause_btn = ttk.Button(controls, text="Pause", command=self._on_pause, state="disabled")
        self.pause_btn.grid(row=0, column=7, padx=(0, 6))
        self.step_btn = ttk.Button(controls, text="Step", command=self._on_step)
        self.step_btn.grid(row=0, column=8, padx=(0, 6))
        self.undo_btn = ttk.Button(controls, text="Undo", command=self._on_undo, state="disabled")
        self.undo_btn.grid(row=0, column=9)

        speed_row = ttk.Frame(outer)
        speed_row.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Label(speed_row, text="Speed (moves/sec):").pack(side="left")
        self.speed_var = tk.DoubleVar(value=2.0)
        self.speed_lbl = ttk.Label(speed_row, text="2.0", width=4)
        ttk.Scale(speed_row, from_=0.2, to=10.0, variable=self.speed_var, orient="horizontal",
                  length=160, command=lambda _e: self._update_speed_label()).pack(side="left", padx=6)
        self.speed_lbl.pack(side="left")
        self.turbo_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(speed_row, text="Turbo (no delay)", variable=self.turbo_var).pack(
            side="left", padx=(16, 0))

        left = ttk.Frame(outer)
        left.grid(row=2, column=0, sticky="n")
        self.canvas = tk.Canvas(left, bg="#bbada0", highlightthickness=0)
        self.canvas.pack()
        stats = ttk.Frame(left, padding=(0, 8))
        stats.pack(fill="x")
        self.step_lbl = ttk.Label(stats, text="Checkpoint step: —", font=("Segoe UI", 10, "bold"))
        self.step_lbl.pack(anchor="w")
        self.score_lbl = ttk.Label(stats, text="Score: 0", font=("Segoe UI", 11))
        self.score_lbl.pack(anchor="w")
        self.maxtile_lbl = ttk.Label(stats, text="Max tile: —", font=("Segoe UI", 11))
        self.maxtile_lbl.pack(anchor="w")
        self.moves_lbl = ttk.Label(stats, text="Moves: 0", font=("Segoe UI", 11))
        self.moves_lbl.pack(anchor="w")
        self.status_lbl = ttk.Label(stats, text="idle", foreground="#888", font=("Segoe UI", 10, "italic"))
        self.status_lbl.pack(anchor="w")

        right = ttk.Frame(outer)
        right.grid(row=2, column=1, sticky="n", padx=(16, 0))
        ttk.Label(right, text="Model thinking...", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        console_frame = ttk.Frame(right)
        console_frame.pack()
        self.console = tk.Text(console_frame, width=64, height=28, bg="#1e1e1e", fg="#d4d4d4",
                                insertbackground="#d4d4d4", font=("Consolas", 10), state="disabled")
        scrollbar = ttk.Scrollbar(console_frame, command=self.console.yview)
        self.console["yscrollcommand"] = scrollbar.set
        self.console.grid(row=0, column=0)
        scrollbar.grid(row=0, column=1, sticky="ns")

    # ------------------------------------------------------- checkpoint
    def _on_load_checkpoint(self):
        label = self.ckpt_var.get()
        match = next((c for c in self.checkpoints if c[1] == label), None)
        if match is None:
            return
        step, _, path = match
        self._log(f"Loading checkpoint: {label} ({path})...")
        self.net, self.board_sizes, self.canvas_h, self.canvas_w, self.loaded_step = \
            load_checkpoint(path)
        board_labels = [f"{w}x{h}" for w, h in self.board_sizes]
        self.board_combo["values"] = board_labels
        self.board_var.set(board_labels[0])
        self.step_lbl.config(text=f"Checkpoint step: {self.loaded_step:,}")
        self._log(f"Loaded. Trained on board sizes: {board_labels} "
                   f"(canvas {self.canvas_h}x{self.canvas_w})")
        self._new_game()

    # ------------------------------------------------------- game control
    def _new_game(self):
        if self.net is None:
            return
        self._stop_worker()
        w, h = (int(x) for x in self.board_var.get().split("x"))
        self.game = Game(width=w, height=h)
        with self.history_lock:
            self.history = []
        self.undo_btn.config(state="disabled")
        self._clear_console()
        self._log(f"New game: {w}x{h} board, checkpoint step {self.loaded_step:,}")
        self._resize_canvas(w, h)
        self._draw_board()
        self._update_stats()
        self.status_lbl.config(text="idle", foreground="#888")

    def _resize_canvas(self, w, h):
        self.canvas.config(width=w * CELL_SIZE + CELL_PAD, height=h * CELL_SIZE + CELL_PAD)

    def _on_start(self):
        if self.running or self.game is None:
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
        if self.running or self.game is None:
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
        self.worker_thread = threading.Thread(target=self._worker_run, args=(loop,), daemon=True)
        self.worker_thread.start()

    # ------------------------------------------------------- worker thread
    def _worker_run(self, loop):
        net, canvas_h, canvas_w = self.net, self.canvas_h, self.canvas_w

        def log(msg):
            self.msg_queue.put(("log", msg))

        def think_pause():
            if self.turbo_var.get():
                return
            speed = max(0.2, self.speed_var.get())
            time.sleep(min(0.35, 1.0 / (speed * 3.0)))

        while True:
            if self.stop_requested:
                break
            if self.game.is_over():
                self.msg_queue.put(("gameover", None))
                break

            d, q_by_dir = pick_action(net, self.game.grid, self.game.h, self.game.w,
                                       canvas_h, canvas_w)
            parts = [f"{k}={v:8.2f}" if v is not None else f"{k}=  illegal"
                     for k, v in q_by_dir.items()]
            log("model Q-values: " + "  ".join(parts))
            log(f"model: chose {d}")
            think_pause()

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
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=str, default="./checkpoints",
                    help="folder containing step_*.pt / latest.pt downloaded from Kaggle")
    args = p.parse_args()
    root = tk.Tk()
    app = CheckpointDemoApp(root, args.ckpt_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
