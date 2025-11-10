import torch, torch.nn as nn, torch.optim as optim


class Learner:
    def __init__(self, q_online, q_target, n_actions, lr=1e-4, gamma=0.99, grad_clip=10.0, device="cpu"):
        self.q_online = q_online.to(device)
        self.q_target = q_target.to(device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device
        self.crit = nn.SmoothL1Loss()
        self.opt = optim.Adam(self.q_online.parameters(), lr=lr)
        self.grad_clip = grad_clip

    @torch.no_grad()
    def target_update(self):
        self.q_target.load_state_dict(self.q_online.state_dict())

    def train_step(self, batch):
        # batch.{obs,next_obs}: uint8 (B,4,84,84); convert + scale
        obs = torch.from_numpy(batch.obs).float().div_(255.0).to(self.device)
        next_obs = torch.from_numpy(batch.next_obs).float().div_(255.0).to(self.device)
        actions = torch.from_numpy(batch.actions).long().to(self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.device)
        dones = torch.from_numpy(batch.dones.astype("float32")).to(self.device)

        q = self.q_online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next_max = self.q_target(next_obs).max(1).values
            target = rewards + (1.0 - dones) * self.gamma * q_next_max

        loss = self.crit(q, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_online.parameters(), self.grad_clip)
        self.opt.step()
        return float(loss.item())
    
