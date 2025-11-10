"""
Evaluate two trained agents (left vs right) fighting each other in PettingZoo Pong.

- Loads left/right checkpoints saved by main.py
- Runs N evaluation episodes with epsilon≈0 (greedy)
- Auto-serve: press FIRE for K frames after reset AND after every scored point
- Optional side swap per episode
- Optional on-screen viewer (pygame) with clean scaling/letterboxing
- Prints winner, score, and returns

Examples
    # Scaled window (crisp)
    python eval_selfplay.py --left_ckpt checkpoints/left_step2000000.pt --right_ckpt checkpoints/right_step2000000.pt \
        --episodes 10 --swap_sides 1 --render rgb_array --scale 6 --fps 60

    # Fit into arbitrary window with letterbox
    python eval_selfplay.py --left_ckpt ckpts/left.pt --right_ckpt ckpts/right.pt \
        --render rgb_array --win_width 1280 --win_height 960
"""
from __future__ import annotations
import argparse
from typing import List
import numpy as np
import torch

# NEW: optional display
import pygame

from envs import make_pong_env, hwc_to_chw_uint8
from network import QNetwork

DEFAULT_MEANINGS = ["NOOP", "FIRE", "UP", "DOWN", "UPFIRE", "DOWNFIRE"]

def _get_action_meanings(env) -> List[str]:
    try:
        meanings = env.unwrapped.get_action_meanings()
        if isinstance(meanings, dict):
            meanings = list(meanings.values())[0]
        return meanings
    except Exception:
        return DEFAULT_MEANINGS

def _find_idx(meanings: List[str], label: str, default_idx: int) -> int:
    try:
        return meanings.index(label)
    except ValueError:
        return default_idx

def load_q_from_ckpt(ckpt_path: str, n_actions: int, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    q = QNetwork(n_actions=n_actions, in_channels=4).to(device)
    q.load_state_dict(ckpt["online_state_dict"])
    q.eval()
    return q

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left_ckpt", type=str, required=True)
    parser.add_argument("--right_ckpt", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Rendering
    parser.add_argument("--render", type=str, default="none", choices=["none", "rgb_array", "human"],
                        help="Use 'rgb_array' for the pygame window below. 'human' uses ALE's own viewer.")
    parser.add_argument("--swap_sides", type=int, default=1, help="Swap left/right assignment each episode (0/1)")
    parser.add_argument("--kickoff_frames", type=int, default=10, help="FIRE frames after reset/score to serve the ball")
    parser.add_argument("--max_steps", type=int, default=200_000, help="Safety cap per episode")

    # NEW: pygame window options (used when --render=rgb_array)
    parser.add_argument("--scale", type=int, default=0, help="Integer pixel scale from native frame size (if >0).")
    parser.add_argument("--win_width", type=int, default=960, help="Target window width when --scale=0.")
    parser.add_argument("--win_height", type=int, default=720, help="Target window height when --scale=0.")
    parser.add_argument("--nearest", type=int, default=1, help="Use nearest-neighbor scaling when --scale>0 (1=yes,0=no).")
    parser.add_argument("--fps", type=int, default=60, help="Display FPS cap for pygame viewer.")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Build env (ensure rgb_array for pygame viewer)
    render_mode = None if args.render == "none" else args.render
    env = make_pong_env(render_mode=render_mode)
    obs, infos = env.reset(seed=args.seed)
    agents = list(env.agents)
    assert len(agents) == 2, f"Expected 2 agents, got {agents}"
    left_name, right_name = agents[0], agents[1]
    n_actions = int(env.action_space(left_name).n)

    meanings = _get_action_meanings(env)
    fire_idx = _find_idx(meanings, "FIRE", 1)

    left_q  = load_q_from_ckpt(args.left_ckpt,  n_actions, device)
    right_q = load_q_from_ckpt(args.right_ckpt, n_actions, device)

    @torch.no_grad()
    def q_vals(qnet: torch.nn.Module, x_chw_float: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(x_chw_float).to(device)
        return qnet(x).cpu().numpy()

    # --- pygame display setup (only if rgb_array) ---
    use_pygame = (args.render == "rgb_array")
    if use_pygame:
        pygame.init()
        # Get one frame to know native size
        first_frame = env.render()
        if first_frame is None:
            # some envs return a frame only after a step; assume Atari 210x160 and correct later
            native_h, native_w = 210, 160
        else:
            native_h, native_w = int(first_frame.shape[0]), int(first_frame.shape[1])

        if args.scale and args.scale > 0:
            win_w, win_h = native_w * args.scale, native_h * args.scale
            letterbox = False
        else:
            win_w, win_h = int(args.win_width), int(args.win_height)
            letterbox = True

        screen = pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption("Self-play (LEFT vs RIGHT)")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 22)

        def scale_surface(surf, tw, th, nearest: bool):
            return pygame.transform.scale(surf, (tw, th)) if nearest else pygame.transform.smoothscale(surf, (tw, th))

        def blit_frame(frame: np.ndarray, ep: int, pL: int, pR: int):
            nonlocal native_w, native_h
            if frame.shape[0] != native_h or frame.shape[1] != native_w:
                native_h, native_w = int(frame.shape[0]), int(frame.shape[1])
            surf = pygame.surfarray.make_surface(np.rot90(frame))
            if args.scale and args.scale > 0:
                scaled = scale_surface(surf, win_w, win_h, bool(args.nearest))
                screen.blit(scaled, (0, 0))
            else:
                ar_src = native_w / native_h
                ar_dst = win_w / win_h
                if abs(ar_src - ar_dst) < 1e-6:
                    scaled = pygame.transform.smoothscale(surf, (win_w, win_h))
                    screen.blit(scaled, (0, 0))
                else:
                    if ar_src > ar_dst:
                        # wider → full width
                        sw, sh = win_w, int(win_w / ar_src)
                        ox, oy = 0, (win_h - sh) // 2
                    else:
                        # taller → full height
                        sh, sw = win_h, int(win_h * ar_src)
                        ox, oy = (win_w - sw) // 2, 0
                    screen.fill((0, 0, 0))
                    scaled = pygame.transform.smoothscale(surf, (sw, sh))
                    screen.blit(scaled, (ox, oy))
            hud = font.render(f"EP {ep} | Score L/R: {pL}-{pR}", True, (255, 255, 255))
            screen.blit(hud, (10, 10))
            pygame.display.flip()
            clock.tick(args.fps)
    # ---------------------------------------------------

    total = {"left": 0, "right": 0}
    for ep in range(1, args.episodes + 1):
        # Optional side swap each episode
        use_left_q = left_q if (ep % 2 == 1 or not args.swap_sides) else right_q
        use_right_q = right_q if (ep % 2 == 1 or not args.swap_sides) else left_q

        obs, infos = env.reset(seed=args.seed + ep)
        points = {left_name: 0, right_name: 0}
        returns = {left_name: 0.0, right_name: 0.0}

        serve_frames_left = args.kickoff_frames
        steps = 0

        while True:
            steps += 1
            if steps > args.max_steps:
                print(f"[warn] Episode {ep}: reached max_steps={args.max_steps}, forcing reset.")
                break

            # pygame events (window close)
            if use_pygame:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit(); env.close(); return

            # prepare CHW float in [0,1]
            oL = hwc_to_chw_uint8(obs[left_name]).astype(np.float32, copy=False)[None] / 255.0
            oR = hwc_to_chw_uint8(obs[right_name]).astype(np.float32, copy=False)[None] / 255.0

            if serve_frames_left > 0:
                aL = fire_idx
                aR = fire_idx
                serve_frames_left -= 1
            else:
                aL = int(np.argmax(q_vals(use_left_q, oL)))
                aR = int(np.argmax(q_vals(use_right_q, oR)))

            obs, rewards, terminations, truncations, infos = env.step({left_name: aL, right_name: aR})

            # scoring / returns
            rL = float(rewards.get(left_name, 0.0))
            rR = float(rewards.get(right_name, 0.0))
            returns[left_name] += rL
            returns[right_name] += rR
            if rL > 0:
                points[left_name] += 1
            if rR > 0:
                points[right_name] += 1

            # re-serve after any score
            if (rL != 0.0) or (rR != 0.0):
                serve_frames_left = args.kickoff_frames

            # draw
            if use_pygame:
                frame = env.render()
                if frame is not None:
                    blit_frame(frame, ep, points[left_name], points[right_name])

            if len(env.agents) == 0:
                break

        # episode summary
        if points[left_name] > points[right_name]:
            winner = left_name; total["left"] += 1
        elif points[right_name] > points[left_name]:
            winner = right_name; total["right"] += 1
        else:
            winner = "tie"
        print(f"EP {ep} | winner: {winner} | score L/R: {points[left_name]}-{points[right_name]} | "
              f"returns L/R: {returns[left_name]:.1f}/{returns[right_name]:.1f}")

    print(f"\nFinal tally over {args.episodes} episodes → LEFT wins: {total['left']}, RIGHT wins: {total['right']}")
    if use_pygame:
        pygame.quit()
    env.close()

if __name__ == "__main__":
    main()
