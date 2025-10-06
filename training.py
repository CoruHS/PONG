# train_then_play_pong.py
import argparse
import time
import sys
import numpy as np
import pygame
import gymnasium as gym
import ale_py  # NEW: register ALE envs with Gymnasium

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper

# --- Register ALE environments (required with modern Gymnasium) ---
gym.register_envs(ale_py)

ENV_ID = "ALE/Pong-v5"
FRAME_STACK = 4

# --------- helpers ---------
def make_train_env(seed=0):
    # Vectorized env with standard Atari preprocessing
    vec = make_vec_env(
        ENV_ID,
        n_envs=8,
        seed=seed,
        wrapper_class=AtariWrapper,
        wrapper_kwargs=dict(
            clip_obs=True, clip_reward=False, terminal_on_life_loss=False
        ),
    )
    vec = VecFrameStack(vec, n_stack=FRAME_STACK)
    return vec

def _make_single_env_for_play():
    e = gym.make(ENV_ID, render_mode="rgb_array")
    e = AtariWrapper(e, clip_obs=True, clip_reward=False, terminal_on_life_loss=False)
    return e

def make_play_env():
    # Single env; we’ll render via rgb_array and show with pygame
    env = DummyVecEnv([_make_single_env_for_play])
    env = VecFrameStack(env, n_stack=FRAME_STACK)
    return env

def _unwrap_dummy_env(vec_env):
    """
    Walk through Vec wrappers to reach the base DummyVecEnv, then return its first env.
    Works for VecFrameStack(DummyVecEnv([...])) and similar stacks.
    """
    v = vec_env
    # Peel wrappers like VecFrameStack -> .venv
    while hasattr(v, "venv"):
        v = v.venv
    # v should be a DummyVecEnv now
    base_env = v.envs[0]
    # Some SB3 wrappers add .env; unwrap to the raw Gymnasium env if present
    cur = base_env
    while hasattr(cur, "env"):
        cur = cur.env
    return cur

def action_indices(raw_env):
    """
    Discover the action indices at runtime so we don't hard-code them.
    Prefers UP/DOWN, falls back to RIGHT/LEFT if needed.
    """
    meanings = raw_env.unwrapped.get_action_meanings()
    idx = {m: i for i, m in enumerate(meanings)}

    up = idx.get("UP", idx.get("RIGHT"))
    down = idx.get("DOWN", idx.get("LEFT"))
    noop = idx.get("NOOP", 0)
    fire = idx.get("FIRE", None)

    if up is None or down is None:
        # Extremely defensive: if for some reason neither mapping exists.
        # Pong always has a vertical pair, but just in case:
        raise RuntimeError(f"Could not locate UP/DOWN (or RIGHT/LEFT) in meanings: {meanings}")

    return noop, fire, up, down

def get_frame_from_vec_env(vec_env):
    """
    Try vec_env.render() first; if None (backend-dependent), grab from the base env.
    Returns an HxWxC (RGB) numpy array.
    """
    frame = None
    try:
        frame = vec_env.render()  # many vec envs return an RGB array here
    except Exception:
        frame = None
    if frame is None:
        base = _unwrap_dummy_env(vec_env)
        frame = base.render()
    return frame

# --------- training / playing ---------
def train_model(timesteps, save_path):
    env = make_train_env()
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        clip_range=0.1,
        n_epochs=4,
        tensorboard_log="./tb_pong/",
    )
    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    env.close()
    return save_path

def play_with_model(model_path):
    # Pygame setup
    pygame.init()
    pygame.display.set_caption("Pong — You + Trained Agent")
    clock = pygame.time.Clock()

    env = make_play_env()
    model = PPO.load(model_path, env=env)

    # Figure out action indices dynamically from the raw env
    raw_env = _unwrap_dummy_env(env)
    ACTION_NOOP, ACTION_FIRE, ACTION_UP, ACTION_DOWN = action_indices(raw_env)

    # Reset
    obs = env.reset()
    # Gymnasium sometimes returns (obs, info) for raw envs, but SB3 VecEnv returns obs only.
    if isinstance(obs, tuple):
        obs = obs[0]

    # Build a preview surface
    frame = get_frame_from_vec_env(env)
    h, w, _ = frame.shape
    screen = pygame.display.set_mode((w, h))

    human_mode = False   # toggle: True = human-only, False = assist (agent when idle)
    paused = False

    def get_human_action():
        keys = pygame.key.get_pressed()
        up = keys[pygame.K_UP]
        down = keys[pygame.K_DOWN]
        if up and not down:
            return ACTION_UP
        if down and not up:
            return ACTION_DOWN
        # If neither pressed, return NOOP (lets agent act in assist mode)
        return ACTION_NOOP

    print("\nControls:")
    print("  ↑ / ↓ : move paddle")
    print("  H     : toggle Human-only vs Assist mode (default Assist)")
    print("  P     : pause/unpause")
    print("  R     : reset episode")
    print("  Q / ESC: quit\n")

    while True:
        # --- handle events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    return
                if event.key == pygame.K_h:
                    human_mode = not human_mode
                    print(f"[Mode] {'Human-only' if human_mode else 'Assist (Agent when idle)'}")
                if event.key == pygame.K_p:
                    paused = not paused
                    print("[Paused]" if paused else "[Unpaused]")
                if event.key == pygame.K_r:
                    obs = env.reset()
                    if isinstance(obs, tuple):
                        obs = obs[0]
                    print("[Reset]")

        if paused:
            clock.tick(30)
            continue

        # --- choose action ---
        human_action = get_human_action()
        if human_mode:
            action = np.array([human_action], dtype=np.int64)
        else:
            if human_action == ACTION_NOOP:
                # Let agent act
                action, _ = model.predict(obs, deterministic=True)
            else:
                # Human overrides while key is held
                action = np.array([human_action], dtype=np.int64)

        # --- step env ---
        obs, rewards, dones, infos = env.step(action)

        # --- draw frame ---
        frame = get_frame_from_vec_env(env)  # robust render
        # Convert numpy array to pygame surface
        pygame.surfarray.blit_array(screen, frame.swapaxes(0, 1))
        pygame.display.flip()

        # Reset if episode done
        if dones[0]:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

        clock.tick(60)  # target ~60 FPS

def main():
    parser = argparse.ArgumentParser(description="Train Pong, then play with the trained model.")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="Training steps")
    parser.add_argument("--save_path", type=str, default="ppo_pong.zip", help="Model save path")
    args = parser.parse_args()

    print(f"Training for {args.timesteps:,} timesteps…")
    model_path = train_model(args.timesteps, args.save_path)
    print(f"Training done. Saved to {model_path}")
    print("Launching interactive play…")
    play_with_model(model_path)

if __name__ == "__main__":
    main()
