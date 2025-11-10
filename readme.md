
# PettingZoo Pong DQN — Self‑Play & Human Play

Train two DQN agents to play Atari **Pong** (PettingZoo) with SuperSuit preprocessing and optional reward shaping. Includes self‑play training/evaluation and a script to play **human vs. agent**.

---

## Requirements

* **Python**: 3.11.14
* OS: Linux / macOS / Windows (GUI rendering requires a display; headless servers should use `--render_mode rgb_array`)
* Optional GPU (CUDA) or Apple Silicon (MPS) supported by PyTorch

### Python libraries

Paste these **one line at a time** in your terminal:

```bash
# Core environment + ROM installer
pip install "pettingzoo[atari]" "autorom[accept-rom-license]" supersuit numpy
AutoROM --accept-license

# Deep learning + video I/O
pip install torch imageio

# Nice to have (faster/robust image ops for some systems)
pip install opencv-python-headless
```

> **Why AutoROM?** Atari games require ROMs. `AutoROM --accept-license` installs them under the appropriate directory so `pettingzoo[atari]` can load Pong.

---

## Project Structure

```
.
├── envs.py           # PettingZoo pong_v3 + SuperSuit preprocessing utilities
├── network.py        # NatureCNN backbone + QNetwork head
├── learner.py        # DQN learner (target network, Adam, Huber loss)
├── policy.py         # ε-greedy action selection + epsilon scheduler
├── replay.py         # Minimal circular replay buffer (uint8 frames)
├── shaping.py        # Optional reward shaping utilities
├── main.py           # Self-play training loop (two agents)
├── eval_selfplay.py  # Evaluate two checkpoints head-to-head
└── play_human_vs_right.py  # Play human vs. trained right-side agent
```

---

## Quick Start

### 1) Install dependencies & ROMs

Follow the **Python libraries** commands above.

### 2) Train in self‑play

Minimal example (CPU or GPU auto-detected):

```bash
python main.py --total_steps 500000 --render_mode none
```

To **watch** self‑play with a window:

```bash
python main.py --total_steps 200000 --render_mode human
```

```bash
click run or python main.py
```

> Tip: `--render_mode human` opens an interactive window and is slower. For speed, use `--render_mode none` or `rgb_array`.

### 3) Checkpoints

Checkpoints are saved to `checkpoints/` when enabled. Filenames look like `left_step2000000.pt` and `right_step2000000.pt`.

---

## Beginner‑Friendly Hyperparameters

These defaults live in `main.py` and are safe starters:

```python
parser.add_argument("--total_steps", type=int, default=1_000_000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

# Replay & learning
parser.add_argument("--replay_capacity", type=int, default=100_000)  # ~11 GB total for two buffers
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learn_start", type=int, default=50_000)
parser.add_argument("--train_freq", type=int, default=4)
parser.add_argument("--target_update", type=int, default=10_000)

# Optimization
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--gamma", type=float, default=0.99)

# Exploration schedule (shared)
parser.add_argument("--eps_start", type=float, default=1.0)
parser.add_argument("--eps_final", type=float, default=0.01)
parser.add_argument("--eps_decay", type=int, default=600_000)  # ~0.6–0.7 × total_steps works well

# Rendering
parser.add_argument("--render_mode", type=str, default="rgb_array", choices=["none", "rgb_array", "human"])  # use "human" to watch

# Checkpointing
parser.add_argument("--save_dir", type=str, default="checkpoints")
parser.add_argument("--save_every", type=int, default=0)       # keep 0 to avoid frequent I/O
parser.add_argument("--save_on_episode", type=int, default=0)  # keep 0 for speed

# Reward shaping (toggle + hyperparameters)
parser.add_argument("--use_shaping", type=int, default=1)
parser.add_argument("--shape_alpha", type=float, default=0.10)
parser.add_argument("--shape_beta_contact", type=float, default=0.05)
parser.add_argument("--shape_lambda", type=float, default=0.90)
parser.add_argument("--shape_step_cost", type=float, default=1e-4)
parser.add_argument("--shape_reach_margin", type=float, default=10.0)
parser.add_argument("--shape_miss_penalty", type=float, default=0.40)
```

### Memory note

* `replay_capacity=100_000` with 84×84×4 stacked frames (uint8) stores **both** current and next states per transition for **two agents** ⇒ roughly **~11 GB** total RAM.
* Reduce memory by lowering `--replay_capacity` (e.g., `50_000`), or run fewer workers.

---

## How to Run

### Self‑Play Training (headless)

```bash
python main.py --total_steps 1000000 --render_mode none
```

### Self‑Play Training (watch the agents)

```bash
python main.py --total_steps 200000 --render_mode human
```

### Save Periodically

```bash
python main.py --total_steps 500000 --save_every 10000 --render_mode none
```

---

## Evaluate Two Trained Agents (Non‑Human vs. Non‑Human)

Use `eval_selfplay.py` to pit checkpoints against each other:

```bash
python eval_selfplay.py \
  --left_ckpt checkpoints/left_step2000000.pt \
  --right_ckpt checkpoints/right_step2000000.pt \
  --episodes 10 \
  --swap_sides 1 \
  --render rgb_array \
  --scale 6 \
  --fps 60
```

* Replace checkpoint paths with your actual filenames.
* Use `--render human` to open a window and watch in real time.

---

## Play **Human vs. Agent**

After training, play against the **right‑side** agent (P2). Example:

```bash
python play_human_vs_right.py \
  --right_ckpt checkpoints/right_step2000000.pt \
  --episodes 5 \
  --human_side left \
  --scale 6
```

* For variants that use `pong_v3` left checkpoint naming, just point to the correct file, e.g.:

```bash
python play_human_vs_right.py \
  --right_ckpt checkpoints/left_step2000000.pt \
  --episodes 5 \
  --human_side left \
  --scale 6
```

**Controls** depend on your `play_human_vs_right.py` implementation (commonly arrow keys / WASD). Ensure the script maps keyboard input to paddle actions and that `--render human` is used internally.

---

## Tips & Practical Settings

* **Speed vs. visuals**: prefer `--render_mode none` for training speed. Use `human` only to inspect behavior.
* **Epsilon decay**: set `--eps_decay ≈ 0.6–0.7 × --total_steps` for smooth annealing.
* **Target updates**: `--target_update 10_000` is a standard DQN baseline; try `20_000` for extra stability.
* **Learning starts**: ensure `--learn_start` is well below `--replay_capacity` (e.g., 50k with 100k capacity).
* **Checkpoints**: `--save_every` introduces I/O; keep at `0` for pure throughput and use manual saves if you add them.

---

## Troubleshooting

* **Game ROM not found**: run `AutoROM --accept-license` again. Verify `pettingzoo[atari]` is installed.
* **No window appears** (headless server): use `--render_mode rgb_array` or `none`.
* **CUDA out of memory**: lower batch size or run on CPU; reduce `--replay_capacity`.
* **High RAM use**: see **Memory note** above; replay dominates memory.

---

## License & Credits

* Built on **PettingZoo** (Atari), **SuperSuit**, and **PyTorch**. Atari ROMs installed via **AutoROM**.
* NatureCNN architecture as popularized by the DQN Nature paper (2015).

---

## Citation

If you use this codebase in a project or class, consider citing PettingZoo and SuperSuit. See their docs for proper citations.
