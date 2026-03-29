import gymnasium as gym
import torch
import torch.nn as nn
import torch.distributions as dist
from tqdm import tqdm

DEVICE = "cuda"
DTYPE = torch.float32
CLIP_EPS = 0.2
LR = 1e-3
ROLLOUT_STEPS = 64

class CartPoleNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

env = gym.make("CartPole-v1", render_mode="human")
model = CartPoleNet()
model.to(DEVICE)

opt = torch.optim.Adam(model.parameters(), lr=LR)

for ep in range(500):
    print(f"Episode {ep}...")
    # PHASE 1: collect experience
    state, _ = env.reset()
    done = False
    # buffer for training later
    states, actions, logprobs, rewards = [], [], [], []
    model.eval()
    with torch.no_grad():
        for _ in range(ROLLOUT_STEPS):
            while not done:
                # get action prob and multinomial dist from the actor
                probs = model.actor(torch.tensor(state, device=DEVICE, dtype=DTYPE))
                m = dist.Categorical(probs)
                # sample one action
                action = m.sample()
                log_p = m.log_prob(action)

                # get next state and reward
                next_state, reward, done, truncated, info = env.step(action.item())
                done = done or truncated

                # store the states in the buffer
                states.append(state)
                actions.append(action.item())
                logprobs.append(log_p)
                rewards.append(reward)

                state = next_state

    # PHASE 2: calculatation
    returns = []
    G = 0
    for i in reversed(range(len(rewards))):
        G = rewards[i] + G * 0.99
        returns.append(G)
    returns.reverse()

    # convert list to tensor
    states = torch.tensor(states, device=DEVICE, dtype=DTYPE)
    actions = torch.tensor(actions, device=DEVICE, dtype=DTYPE)
    logprobs = torch.stack(logprobs).detach().to(DTYPE).to(DEVICE)
    returns = torch.tensor(returns, device=DEVICE, dtype=DTYPE)

    # PHASE 3: PPO
    print("Backward pass...")
    model.train()
    for _ in tqdm(range(5)):
        new_probs = model.actor(states)
        new_m = dist.Categorical(new_probs)
        new_logprobs = new_m.log_prob(actions)

        ratio = (new_logprobs - logprobs).exp()

        values = model.critic(states).squeeze()
        advantages = returns - values

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.MSELoss()(returns, values)

        loss = actor_loss + 0.5 * critic_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

    print("Episode finished!")
    print(f"Total reward: {sum(rewards)}")

env.close()
