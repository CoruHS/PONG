from dataclasses import dataclass

@dataclass
class DQNConfig:
    seed: int = 42
    device: str = "cuda"  # "cpu" if no GPU
    # observation
    frame_size: int = 84
    stack_size: int = 4
    # replay
    replay_capacity: int = 100_000
    batch_size: int = 32
    min_replay_size: int = 20_000
    # optimization
    gamma: float = 0.99
    lr: float = 1e-4
    grad_clip: float = 10.0
    # epsilon schedule
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 1_000_000
    # target network
    target_sync_interval: int = 10_000  # steps
    # training control
    total_env_steps: int = 1_000_000
    train_freq: int = 4
    eval_interval: int = 50_000
    checkpoint_interval: int = 200_000
    # logging
    log_interval: int = 10_000
    # env
    right_noise: float = 0.25
    render_eval: bool = False
