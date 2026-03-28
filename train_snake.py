import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import numpy as np
import gymnasium as gym
# Assuming your snake_env is in the same directory or installed
import snake_env
import os
from datetime import datetime

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RENDER = True  # set True to watch training via pygame
ENV_NAME = "Snake-v0"
BOARD_SIZE = 8
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
CRITIC_COEF = 0.5
MAX_GRAD_NORM = 0.5
ROLLOUT_STEPS = 2048
BATCH_SIZE = 64
UPDATE_EPOCHS = 10
OUTPUT_DIR = f"outputs/snake-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- THE ARCHITECTURE ---
class SnakeNet(nn.Module):
    def __init__(self, size):
        super().__init__()
        # 1. Spatial Feature Extractor (No BatchNorm!)
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        feat_dim = 64 * size * size
        self.actor = nn.Linear(feat_dim, 4)
        self.critic = nn.Linear(feat_dim, 1)

        # 2. Orthogonal Init (The 'Prodigy' secret)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)

        # Make policy flat and value small at start
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, x):
        x = x / 3.0 # Normalization
        f = self.convs(x)
        return self.actor(f), self.critic(f)

# --- THE PPO BRAIN ---
def train():
    env = gym.make(ENV_NAME, size=BOARD_SIZE, render_mode="human" if RENDER else None)
    model = SnakeNet(BOARD_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    best_reward = -float('inf')

    for iteration in range(1000):
        # 1. COLLECT ROLLOUT (Multiple episodes until 2048 steps)
        states, actions, logprobs, rewards, values, masks = [], [], [], [], [], []
        state, _ = env.reset()

        model.eval()
        first_rollout = True
        for i in range(ROLLOUT_STEPS):
            if RENDER and first_rollout:
                env.render()
            state_t = torch.tensor(state, device=DEVICE, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, val = model(state_t)
                m = dist.Categorical(logits=logits)
                action = m.sample()
                log_p = m.log_prob(action)

            next_state, reward, term, trunc, _ = env.step(action.item())
            done = term or trunc

            states.append(state)
            actions.append(action.item())
            logprobs.append(log_p.item())
            rewards.append(reward)
            values.append(val.item())
            masks.append(1.0 - done)

            state = next_state if not done else env.reset()[0]
            if done:
                first_rollout = False

        # 2. CALCULATE ADVANTAGES (GAE)
        states_t = torch.tensor(np.array(states), device=DEVICE, dtype=torch.float32)
        actions_t = torch.tensor(actions, device=DEVICE, dtype=torch.long)
        old_logprobs_t = torch.tensor(logprobs, device=DEVICE, dtype=torch.float32)

        # Final value for bootstrapping
        with torch.no_grad():
            _, next_value = model(torch.tensor(state, device=DEVICE, dtype=torch.float32).unsqueeze(0))
            next_value = next_value.item()

        returns = []
        gae = 0
        for i in reversed(range(ROLLOUT_STEPS)):
            delta = rewards[i] + GAMMA * next_value * masks[i] - values[i]
            gae = delta + GAMMA * GAE_LAMBDA * masks[i] * gae
            next_value = values[i]
            returns.insert(0, gae + values[i])

        returns_t = torch.tensor(returns, device=DEVICE, dtype=torch.float32)
        advantages_t = returns_t - torch.tensor(values, device=DEVICE, dtype=torch.float32)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # 3. PPO UPDATE
        model.train()
        for _ in range(UPDATE_EPOCHS):
            # Shuffle indices for mini-batching
            idxs = np.random.permutation(ROLLOUT_STEPS)
            for i in range(0, ROLLOUT_STEPS, BATCH_SIZE):
                batch_idx = idxs[i:i+BATCH_SIZE]

                b_states = states_t[batch_idx]
                b_actions = actions_t[batch_idx]
                b_old_logp = old_logprobs_t[batch_idx]
                b_returns = returns_t[batch_idx]
                b_advantages = advantages_t[batch_idx]

                new_logits, new_values = model(b_states)
                new_values = new_values.squeeze()
                m = dist.Categorical(logits=new_logits)
                new_logp = m.log_prob(b_actions)
                entropy = m.entropy().mean()

                ratio = (new_logp - b_old_logp).exp()
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * b_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(new_values, b_returns)

                loss = actor_loss + CRITIC_COEF * critic_loss - ENTROPY_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        # Log progress
        avg_reward = np.sum(rewards) / (ROLLOUT_STEPS / 20) # Rough estimate per 20 steps
        print(f"Iter {iteration} | Avg Step Reward: {np.mean(rewards):.4f} | Loss: {loss.item():.4f}")

        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best.pth"))

if __name__ == "__main__":
    train()
