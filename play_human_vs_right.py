"""
Play human vs the trained RIGHT agent in PettingZoo Pong using a pygame window.

- Loads the RIGHT agent checkpoint (online network)
- You control the LEFT paddle by default (W/S or Up/Down arrows; Space to serve)
- Runs one or more episodes; prints the winner and score each game

Usage
    pip install pygame
    # Option A: integer pixel scale from native frame size (crisp)
    python play_human_vs_right.py --right_ckpt checkpoints/right_step2000000.pt --episodes 5 --human_side left --scale 6
    # Option B: fit into an arbitrary window with letterboxing
    python play_human_vs_right.py --right_ckpt checkpoints/right_step1000000.pt --episodes 5 --human_side left --win_width 1280 --win_height 960
"""
from __future__ import annotations
import argparse
import sys
from typing import Dict, List

import numpy as np
import pygame
import torch

from envs import make_pong_env, hwc_to_chw_uint8
from network import QNetwork

# ---------------- Action mapping helpers ----------------
DEFAULT_MEANINGS = ["NOOP", "FIRE", "UP", "DOWN", "UPFIRE", "DOWNFIRE"]

def get_action_meanings(env) -> List[str]:
    try:
        meanings = env.unwrapped.get_action_meanings()  # works for ALE
        if isinstance(meanings, dict):  # some wrappers might return per-player dict
            meanings = list(meanings.values())[0]
        return meanings
    except Exception:
        return DEFAULT_MEANINGS

def action_from_keys(keys, meanings: List[str]) -> int:
    # Prefer pure directions without FIRE when possible
    def find(label: str, default_idx: int) -> int:
        try:
            return meanings.index(label)
        except ValueError:
            return default_idx

    idx_noop = find("NOOP", 0)
    idx_fire = find("FIRE", 1)
    idx_up   = find("UP", 2)
    idx_down = find("DOWN", 3)

    fire = keys[pygame.K_SPACE]
    up   = keys[pygame.K_UP] or keys[pygame.K_w]
    down = keys[pygame.K_DOWN] or keys[pygame.K_s]

    if fire:
        return idx_fire
    if up and not down:
        return idx_up
    if down and not up:
        return idx_down
    return idx_noop

# ---------------- Loading model ----------------
def load_right_q(ckpt_path: str, n_actions: int, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    q = QNetwork(n_actions=n_actions, in_channels=4).to(device)
    q.load_state_dict(ckpt["online_state_dict"])  # use online weights
    q.eval()
    return q

# ---------------- Main game loop ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--right_ckpt", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--human_side", type=str, default="left", choices=["left", "right"],
                        help="Which side you control: left (first_0) or right (second_0)")
    # Display options
    parser.add_argument("--win_width", type=int, default=640, help="Target window width (used when --scale==0)")
    parser.add_argument("--win_height", type=int, default=480, help="Target window height (used when --scale==0)")
    parser.add_argument("--scale", type=int, default=0,
                        help="Integer scale factor from native frame size. If >0, overrides --win_width/--win_height")
    parser.add_argument("--nearest", type=int, default=1,
                        help="When --scale>0, use nearest-neighbor (1) or smooth (0) scaling")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Env: we need rgb_array to draw ourselves
    env = make_pong_env(render_mode="rgb_array")
    obs, infos = env.reset(seed=args.seed)
    agents = list(env.agents)
    left_name, right_name = agents[0], agents[1]
    n_actions = int(env.action_space(left_name).n)

    # Load RIGHT agent
    right_q = load_right_q(args.right_ckpt, n_actions, device)
    meanings = get_action_meanings(env)

    # Grab one frame to determine native size
    init_frame = env.render()
    if init_frame is None:
        # Some renderers only produce a frame after the first step. Fallback to assumed 210x160 Atari size,
        # but this will get corrected on first real frame.
        native_h, native_w = 210, 160
    else:
        native_h, native_w = int(init_frame.shape[0]), int(init_frame.shape[1])

    # Decide window size
    if args.scale and args.scale > 0:
        win_w = native_w * args.scale
        win_h = native_h * args.scale
        letterbox = False
    else:
        win_w, win_h = int(args.win_width), int(args.win_height)
        letterbox = True  # fit into window with aspect preserved

    # Pygame setup
    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Human vs RIGHT Agent (Pong)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # Choose scaler
    def scale_surface(surf, tw, th, nearest: bool):
        if nearest:
            return pygame.transform.scale(surf, (tw, th))
        else:
            return pygame.transform.smoothscale(surf, (tw, th))

    @torch.no_grad()
    def q_vals(qnet: torch.nn.Module, x_chw_float: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(x_chw_float).to(device)
        return qnet(x).cpu().numpy()

    def blit_frame(frame: np.ndarray):
        nonlocal native_w, native_h
        if frame.shape[0] != native_h or frame.shape[1] != native_w:
            native_h, native_w = int(frame.shape[0]), int(frame.shape[1])  # update if first was None

        # Convert to pygame surface (rotate 90° to match pygame's expected orientation)
        surf = pygame.surfarray.make_surface(np.rot90(frame))

        if args.scale and args.scale > 0:
            scaled = scale_surface(surf, win_w, win_h, nearest=bool(args.nearest))
            screen.blit(scaled, (0, 0))
            return

        # Letterbox fit into (win_w, win_h) while preserving aspect
        ar_src = native_w / native_h
        ar_dst = win_w / win_h

        if abs(ar_src - ar_dst) < 1e-6:
            scaled = pygame.transform.smoothscale(surf, (win_w, win_h))
            screen.blit(scaled, (0, 0))
        else:
            if ar_src > ar_dst:
                # Source wider → full width, reduce height
                scaled_w = win_w
                scaled_h = int(win_w / ar_src)
                ox, oy = 0, (win_h - scaled_h) // 2
            else:
                # Source taller → full height, reduce width
                scaled_h = win_h
                scaled_w = int(win_h * ar_src)
                ox, oy = (win_w - scaled_w) // 2, 0
            screen.fill((0, 0, 0))
            scaled = pygame.transform.smoothscale(surf, (scaled_w, scaled_h))
            screen.blit(scaled, (ox, oy))

    for ep in range(1, args.episodes + 1):
        obs, infos = env.reset(seed=args.seed + ep)
        points = {left_name: 0, right_name: 0}
        returns = {left_name: 0.0, right_name: 0.0}

        running = True
        while running:
            # Handle input/events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); env.close(); sys.exit(0)
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit(); env.close(); sys.exit(0)

            keys = pygame.key.get_pressed()

            # Build actions
            oL = hwc_to_chw_uint8(obs[left_name]).astype(np.float32, copy=False)[None] / 255.0
            oR = hwc_to_chw_uint8(obs[right_name]).astype(np.float32, copy=False)[None] / 255.0

            if args.human_side == "left":
                a_left = action_from_keys(keys, meanings)
                a_right = int(np.argmax(q_vals(right_q, oR)))
            else:
                a_left = int(np.argmax(q_vals(right_q, oL)))
                a_right = action_from_keys(keys, meanings)

            obs, rewards, terminations, truncations, infos = env.step({left_name: a_left, right_name: a_right})

            returns[left_name] += float(rewards.get(left_name, 0.0))
            returns[right_name] += float(rewards.get(right_name, 0.0))
            if rewards.get(left_name, 0.0) > 0:
                points[left_name] += 1
            if rewards.get(right_name, 0.0) > 0:
                points[right_name] += 1

            # Draw frame (scaled)
            frame = env.render()
            if frame is not None:
                blit_frame(frame)

            # HUD
            txt = f"EP {ep} | Score L/R: {points[left_name]}-{points[right_name]}  (You: {args.human_side.upper()})"
            hud = font.render(txt, True, (255, 255, 255))
            screen.blit(hud, (10, 10))

            pygame.display.flip()
            clock.tick(60)  # cap FPS

            if len(env.agents) == 0:
                running = False

        # Episode summary
        if points[left_name] > points[right_name]:
            winner = left_name
        elif points[right_name] > points[left_name]:
            winner = right_name
        else:
            winner = "tie"
        print(
            f"EP {ep} | winner: {winner} | score L/R: {points[left_name]}-{points[right_name]} | "
            f"returns L/R: {returns[left_name]:.1f}/{returns[right_name]:.1f}"
        )

    pygame.quit()
    env.close()

if __name__ == "__main__":
    main()
