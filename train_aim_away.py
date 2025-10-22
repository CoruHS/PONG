#!/usr/bin/env python3
# pong_dqn_uint8.py
# Python 3.11 — Train DQN on Atari Pong with uint8 replay + enjoy/video/ONNX.
#
# Usage:
#   pip install "stable-baselines3[extra]" gymnasium[atari,accept-rom-license] ale-py torch onnx onnxruntime
#   python pong_dqn_uint8.py train --total-timesteps 1500000 --record-after-train
#   python pong_dqn_uint8.py enjoy --model runs/<run>/final.zip
#   python pong_dqn_uint8.py export-onnx --model runs/<run>/final.zip --onnx-path exports/pong_dqn.onnx
#
# Notes:
# - Replay buffer stores observations as uint8 (saves lots of RAM).
# - optimize_memory_usage=True to avoid redundant next_obs storage.
# - Evaluation uses sticky actions=0.25 for robustness/comparability.
# - Videos go to runs/<run>/videos when --record-after-train is set.
#
import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import AtariWrapper


# -------------------- Env builders --------------------
def make_train_env(env_id: str, n_envs: int, seed: int):
    """
    Vectorized Atari env with preprocessing + frame stacking.
    Ensures observations are uint8 so replay stores uint8.
    """
    # make_atari_env applies AtariWrapper by default; we pass kwargs for behavior.
    vec = make_atari_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        wrapper_kwargs={
            "noop_max": 30,          # standard atari start
            # AtariWrapper -> grayscale, resize(84), frame-skip(4), reward clipping (True by default)
            # Observations remain uint8 in [0, 255]
        },
    )
    # Stack 4 frames along channel dimension; dtype remains uint8
    vec = VecFrameStack(vec, n_stack=4)

    # Sanity: print obs space once (shape, dtype should be uint8 (84,84,4))
    try:
        print("[dtype-check/train] obs space:", vec.observation_space)
    except Exception:
        pass
    return vec


def make_eval_env(env_id: str, seed: int, sticky_actions: float = 0.25, render_mode=None, record_dir=None):
    """
    Single-process env for eval/play. Sticky actions default 0.25 for comparability.
    """
    def _make():
        env = gym.make(env_id, repeat_action_probability=sticky_actions, render_mode=render_mode)
        env = Monitor(env)
        # Keep preprocessing identical to training but without reward clipping
        env = AtariWrapper(env, clip_reward=False)
        if record_dir:
            # RecordVideo needs render_mode="rgb_array" above
            env = gym.wrappers.RecordVideo(env, video_folder=record_dir, name_prefix="pong_dqn")
        return env
    venv = DummyVecEnv([_make])
    try:
        print("[dtype-check/eval] obs space:", venv.observation_space)
    except Exception:
        pass
    return venv


# -------------------- Commands --------------------
def cmd_train(args):
    run_name = args.run_name or time.strftime("pong_dqn_%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / run_name
    (outdir / "best").mkdir(parents=True, exist_ok=True)
    (outdir / "eval").mkdir(parents=True, exist_ok=True)
    (outdir / "ckpts").mkdir(parents=True, exist_ok=True)
    (outdir / "videos").mkdir(parents=True, exist_ok=True)

    train_env = make_train_env("ALE/Pong-v5", n_envs=args.n_envs, seed=args.seed)
    eval_env = make_eval_env("ALE/Pong-v5", seed=args.seed + 1, sticky_actions=0.25, render_mode=None)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(outdir / "best"),
        log_path=str(outdir / "eval"),
        eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(args.ckpt_freq // max(args.n_envs, 1), 1),
        save_path=str(outdir / "ckpts"),
        name_prefix="dqn",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    policy_kwargs = dict(
        # Keep images as uint8 in replay; model will divide by 255.0 on-the-fly
        normalize_images=True
    )

    model = DQN(
        "CnnPolicy",
        train_env,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=1.0,
        gamma=0.99,
        train_freq=(4, "step"),
        gradient_steps=1,
        target_update_interval=args.target_update,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        verbose=1,
        tensorboard_log=str(outdir / "tb"),
        seed=args.seed,
        device="auto",
        # RAM savers:
        optimize_memory_usage=True,          # avoid storing next_obs redundantly
        policy_kwargs=policy_kwargs,
    )

    print(f"[train] starting… total_timesteps={args.total_timesteps:,}  run={run_name}")
    model.learn(total_timesteps=args.total_timesteps, callback=[eval_cb, ckpt_cb], progress_bar=True)

    final_path = outdir / "final.zip"
    model.save(str(final_path))
    print(f"[train] done. saved final model to: {final_path}")

    # Optional: quick video of the final policy
    if args.record_after_train:
        print("[train] recording a short rollout…")
        rec_env = make_eval_env("ALE/Pong-v5", seed=args.seed + 123, sticky_actions=0.25,
                                render_mode="rgb_array", record_dir=str(outdir / "videos"))
        enjoy(model_path=str(final_path), episodes=2, render=False, vec_env=rec_env)


def enjoy(model_path: str, episodes: int, render: bool, vec_env=None):
    """
    Run the trained policy to watch it play (or to record if vec_env is a RecordVideo env).
    """
    model = DQN.load(model_path, device="auto")

    if vec_env is None:
        vec_env = make_eval_env("ALE/Pong-v5", seed=7, sticky_actions=0.25,
                                render_mode="human" if render else None)

    env = vec_env
    for ep in range(episodes):
        obs = env.reset()
        ep_rew = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            ep_rew += float(rewards[0])
            done = bool(dones[0])
        print(f"[enjoy] episode {ep+1}: reward={ep_rew}")
    env.close()


def cmd_enjoy(args):
    enjoy(model_path=args.model, episodes=args.episodes, render=not args.headless)


def cmd_export_onnx(args):
    """
    Export the policy network to ONNX for use elsewhere (inference-only).
    """
    from stable_baselines3.common.onnx import export_model

    dummy_env = make_eval_env("ALE/Pong-v5", seed=0, sticky_actions=0.25)
    model = DQN.load(args.model, device="cpu")

    onnx_path = Path(args.onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    export_model(model, onnx_path=str(onnx_path), device="cpu", opset=12)
    print(f"[export] wrote ONNX to: {onnx_path}")


# -------------------- CLI --------------------
def main():
    p = argparse.ArgumentParser(description="DQN for Atari Pong (uint8 replay, SB3 + Gymnasium ALE)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    pt = sub.add_parser("train", help="Train a DQN on ALE/Pong-v5")
    pt.add_argument("--outdir", type=str, default="runs", help="Output directory")
    pt.add_argument("--run-name", type=str, default=None)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--n-envs", type=int, default=8)
    pt.add_argument("--total-timesteps", type=int, default=1_500_000)
    pt.add_argument("--eval-freq", type=int, default=50_000)
    pt.add_argument("--ckpt-freq", type=int, default=250_000)
    pt.add_argument("--buffer-size", type=int, default=1_000_000)
    pt.add_argument("--learning-starts", type=int, default=50_000)
    pt.add_argument("--batch-size", type=int, default=32)
    pt.add_argument("--target-update", type=int, default=10_000)
    pt.add_argument("--lr", type=float, default=1e-4)
    pt.add_argument("--record-after-train", action="store_true")
    pt.set_defaults(func=cmd_train)

    # enjoy (watch the trained agent)
    pe = sub.add_parser("enjoy", help="Watch a trained model play")
    pe.add_argument("--model", type=str, required=True)
    pe.add_argument("--episodes", type=int, default=5)
    pe.add_argument("--headless", action="store_true", help="Don’t open a window; use RGB-only")
    pe.set_defaults(func=cmd_enjoy)

    # export to ONNX
    po = sub.add_parser("export-onnx", help="Export the policy to ONNX")
    po.add_argument("--model", type=str, required=True)
    po.add_argument("--onnx-path", type=str, default="exports/pong_dqn.onnx")
    po.set_defaults(func=cmd_export_onnx)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
