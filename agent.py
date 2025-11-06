# agent.py
# Requires: pettingzoo[atari], gymnasium, numpy
# Tip: pip install "pettingzoo[atari]" gymnasium numpy pygame
# And accept ROMs once: python -m AutoROM --accept-license

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, List

from pettingzoo.atari import pong_v3


# -------------------------- Opponent policies ---------------------------------

class OpponentPolicy:
    """Base class for a left-paddle policy driven by action meanings when available."""
    def __init__(self):
        self.noop: int = 0
        self.up: Optional[int] = None
        self.down: Optional[int] = None
        self.fire: Optional[int] = None  # used for auto-serve

    def reset(self, seed: Optional[int] = None):
        pass

    def set_action_meanings(self, meanings: Optional[List[str]], action_space: spaces.Discrete):
        self.noop, self.up, self.down, self.fire = 0, None, None, None
        if meanings:
            norm = [m.upper() for m in meanings]
            self.noop = norm.index("NOOP") if "NOOP" in norm else 0
            if "FIRE" in norm:
                self.fire = norm.index("FIRE")
            for i, m in enumerate(norm):
                if self.up is None and ("UP" == m or "UP" in m):
                    self.up = i
                if self.down is None and ("DOWN" == m or "DOWN" in m):
                    self.down = i

        # Fallbacks when meanings are missing/odd: guess common layouts
        if self.up is None or self.down is None:
            n = int(action_space.n)
            if n >= 4:
                # Common minimal Pong: [NOOP, FIRE, UP, DOWN]
                self.up = self.up if self.up is not None else 2
                self.down = self.down if self.down is not None else 3
                if self.fire is None:
                    self.fire = 1
            else:
                self.up = None
                self.down = None

    def act(self, observation: np.ndarray, action_space: spaces.Discrete) -> int:
        raise NotImplementedError


class RandomOpponent(OpponentPolicy):
    def __init__(self):
        super().__init__()
        self._rng = np.random.default_rng()

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray, action_space: spaces.Discrete) -> int:
        return int(self._rng.integers(action_space.n))


class StickyUpDownOpponent(OpponentPolicy):
    def __init__(self, flip_prob: float = 0.1):
        super().__init__()
        self.flip_prob = flip_prob
        self.prev = 0
        self._rng = np.random.default_rng()

    def reset(self, seed: Optional[int] = None):
        self.prev = self.noop
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray, action_space: spaces.Discrete) -> int:
        if self.up is None or self.down is None:
            return int(self._rng.integers(action_space.n))
        if self._rng.random() < 0.05:
            self.prev = self.noop
        elif self._rng.random() < self.flip_prob:
            self.prev = self.down if self.prev == self.up else self.up
        elif self.prev == self.noop:
            self.prev = self.up if self._rng.random() < 0.5 else self.down
        return int(self.prev)


# ---------------------- Single-agent wrapper (RIGHT acts) ---------------------

class RightAgentVsOpponent(gym.Env):
    """
    Single-agent Gymnasium env: RIGHT paddle is controlled externally (learner),
    LEFT paddle is a fixed opponent/trainer policy.

    PettingZoo agent mapping for pong_v3:
      - 'first_0'  -> LEFT paddle (opponent)
      - 'second_0' -> RIGHT paddle (learner)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        opponent: OpponentPolicy | None = None,
        render_mode: Optional[str] = None,
        full_action_space: bool = False,
        frameskip: Optional[int] = 4,
        **pong_kwargs,
    ):
        super().__init__()
        self.render_mode = render_mode

        build_kwargs = dict(
            render_mode=render_mode,
            full_action_space=full_action_space,
            **pong_kwargs,
        )
        try:
            self._pz_env = pong_v3.env(frameskip=frameskip, **build_kwargs) if frameskip is not None else pong_v3.env(**build_kwargs)
        except TypeError:
            self._pz_env = pong_v3.env(**build_kwargs)

        self.left_agent = "first_0"
        self.right_agent = "second_0"

        self._opponent = opponent if opponent is not None else RandomOpponent()

        self.observation_space = self._pz_env.observation_space(self.right_agent)
        self.action_space = self._pz_env.action_space(self.right_agent)

        meanings = self._get_action_meanings()
        self._opponent.set_action_meanings(
            meanings, self._pz_env.action_space(self.left_agent)
        )
        self._right_fire: Optional[int] = None
        if meanings:
            norm = [m.upper() for m in meanings]
            self._right_fire = norm.index("FIRE") if "FIRE" in norm else None

        self._terminated = False
        self._truncated = False

    # ------------------------ Utility helpers ---------------------------------

    def _done(self, agent: str) -> bool:
        # Keys can be absent in some wrappers; treat missing as False
        return bool(
            self._pz_env.terminations.get(agent, False)
            or self._pz_env.truncations.get(agent, False)
        )

    def _episode_done(self) -> bool:
        # Episode is over only when BOTH sides are done
        return self._done(self.left_agent) and self._done(self.right_agent)

    def _reward(self, agent: str) -> float:
        return float(self._pz_env.rewards.get(agent, 0.0))

    def _info(self, agent: str) -> Dict[str, Any]:
        return self._pz_env.infos.get(agent, {})

    # ------------------------ Gymnasium API -----------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        self._pz_env.reset(seed=seed, options=options)
        self._opponent.reset(seed=seed)

        # Auto-serve at the start so play begins immediately.
        self._auto_serve_cycle()
        # Ensure it's RIGHT's turn next if the episode hasn't ended.
        self._advance_until_right()

        obs = self._pz_env.observe(self.right_agent)
        self._terminated = self._pz_env.terminations.get(self.right_agent, False)
        self._truncated = self._pz_env.truncations.get(self.right_agent, False)
        info = self._info(self.right_agent)
        return obs, info

    def step(self, action: int):
        # Make sure control is with RIGHT, handling serve and done-agents correctly.
        self._advance_until_right()

        # If the episode ended during LEFT's internal turns, surface it now (no further steps!)
        if self._episode_done() or self._done(self.right_agent):
            obs = self._pz_env.observe(self.right_agent)
            rew = self._reward(self.right_agent)
            info = self._info(self.right_agent)
            term = bool(self._pz_env.terminations.get(self.right_agent, False))
            trunc = bool(self._pz_env.truncations.get(self.right_agent, False))
            self._terminated, self._truncated = term, trunc
            return obs, rew, term, trunc, info

        # RIGHT acts (safe-step)
        self._safe_step_current(int(action))

        # If the action ended the episode, do NOT serve/advance further.
        if self._episode_done() or self._done(self.right_agent):
            obs = self._pz_env.observe(self.right_agent)
            rew = self._reward(self.right_agent)
            term = bool(self._pz_env.terminations.get(self.right_agent, False))
            trunc = bool(self._pz_env.truncations.get(self.right_agent, False))
            info = self._info(self.right_agent)
            self._terminated, self._truncated = term, trunc
            return obs, rew, term, trunc, info

        # If a point happened on RIGHT's move, serve to restart the rally.
        if self._reward(self.right_agent) != 0.0:
            self._auto_serve_cycle()

        # Advance LEFT(s) until it's RIGHT's turn again or episode ends.
        self._advance_until_right()

        # Collect results AFTER all internal steps/serves.
        obs = self._pz_env.observe(self.right_agent)
        rew = self._reward(self.right_agent)
        term = bool(self._pz_env.terminations.get(self.right_agent, False))
        trunc = bool(self._pz_env.truncations.get(self.right_agent, False))
        info = self._info(self.right_agent)

        self._terminated, self._truncated = term, trunc
        return obs, rew, term, trunc, info


    def render(self):
        return self._pz_env.render()

    def close(self):
        self._pz_env.close()

    # ------------------------ Internals ---------------------------------------

    def _safe_step_current(self, action_or_none: Optional[int]):
        """
        Step the underlying PettingZoo env for the CURRENT selected agent.
        If all agents are done, DO NOT STEP.
        If selected agent is done, pass None as required by PettingZoo.
        """
        if self._episode_done():
            return
        sel = self._pz_env.agent_selection
        if self._done(sel):
            self._pz_env.step(None)
        else:
            self._pz_env.step(action_or_none)

    def _auto_serve_cycle(self):
        """
        Press FIRE for whichever agent is currently up, safely.
        Up to two presses (typically left then right).
        Do not step if the episode is fully done.
        """
        for _ in range(2):
            if self._episode_done():
                break

            sel = self._pz_env.agent_selection

            # If selected agent is done, advance with None (only if episode not fully done)
            if self._done(sel):
                self._pz_env.step(None)
                continue

            if sel == self.left_agent and self._opponent.fire is not None:
                self._pz_env.step(int(self._opponent.fire))
            elif sel == self.right_agent and self._right_fire is not None:
                self._pz_env.step(int(self._right_fire))
            else:
                # Can't serve (no FIRE known) -> bail out to avoid illegal action
                break

            # Stop if the step ended the episode.
            if self._episode_done():
                break

    def _advance_until_right(self):
        """
        Drive the env while it's LEFT's turn until it's RIGHT's turn or the episode ends.
        Do not step if the episode is fully done.
        """
        while (
            not self._episode_done()
            and self._pz_env.agent_selection == self.left_agent
            and not self._done(self.right_agent)
        ):
            # Handle serve states / done-agents first.
            self._auto_serve_cycle()

            # If serving handed control to RIGHT or ended the episode, stop.
            if self._episode_done() or self._pz_env.agent_selection != self.left_agent or self._done(self.right_agent):
                break

            # If LEFT is done, advance with None and continue.
            if self._done(self.left_agent):
                self._pz_env.step(None)
                continue

            # LEFT takes its turn.
            left_obs = self._pz_env.observe(self.left_agent)
            a_left = self._opponent.act(left_obs, self._pz_env.action_space(self.left_agent))
            self._pz_env.step(int(a_left))

            # If a point happened while LEFT acted and episode didn't end, serve.
            if not self._episode_done() and self._reward(self.right_agent) != 0.0:
                self._auto_serve_cycle()

    def _get_action_meanings(self) -> Optional[List[str]]:
        for obj in (self._pz_env, getattr(self._pz_env, "unwrapped", None)):
            if obj is None:
                continue
            fn = getattr(obj, "get_action_meanings", None)
            if callable(fn):
                try:
                    return list(fn())
                except Exception:
                    pass
        return None


# -------------------------- Factory ------------------------------------------

def make_env(
    opponent: str | OpponentPolicy = "random",
    render_mode: Optional[str] = None,
    **kwargs,
) -> RightAgentVsOpponent:
    if isinstance(opponent, str):
        k = opponent.lower()
        if k == "random":
            opp = RandomOpponent()
        elif k in ("sticky", "sticky_up_down"):
            opp = StickyUpDownOpponent()
        else:
            raise ValueError(f"Unknown opponent '{opponent}'")
    elif isinstance(opponent, OpponentPolicy):
        opp = opponent
    else:
        raise TypeError("opponent must be a string alias or an OpponentPolicy instance")
    return RightAgentVsOpponent(opponent=opp, render_mode=render_mode, **kwargs)


# ---------------------- Optional quick demo (toggle) --------------------------

RUN_DEMO = False

if __name__ == "__main__" and RUN_DEMO:
    OPPONENT = "sticky_up_down"
    RENDER_MODE = "human"
    env = make_env(opponent=OPPONENT, render_mode=RENDER_MODE)
    obs, info = env.reset(seed=42)
    done = trunc = False
    ret = 0.0
    rng = np.random.default_rng(0)
    env.render()
    while not (done or trunc):
        action = int(rng.integers(env.action_space.n))
        obs, r, done, trunc, info = env.step(action)
        ret += r
        env.render()
    print("Episode return (right paddle):", ret)
    env.close()
