import gymnasium as gym
import torch
import torch.nn as nn
import torch.distributions as dist
from tqdm import tqdm

DEVICE = "cuda"
DTYPE = torch.float32
CLAMP = 0.1

class CartPoleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

env = gym.make("CartPole-v1", render_mode="human")
model = CartPoleNet()
model.to(DEVICE)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for ep in range(500):
    print(f"Episode {ep}...")
    # PHASE 1: collect experience
    state, _ = env.reset()
    done = False
    # buffer for training later
    states, actions, logprobs, rewards = [], [], [], []
    model.eval()
    with torch.no_grad():
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
        surr2 = torch.clamp(ratio, 1.0 - CLAMP, 1.0 + CLAMP) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.MSELoss()(returns, values)

        loss = actor_loss + 0.5 * critic_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

    print("Episode finished!")
    print(f"Total reward: {sum(rewards)}")

env.close()
