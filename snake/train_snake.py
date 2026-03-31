import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
import numpy as np
import gymnasium as gym
import snake_env
import os
import shutil
from datetime import datetime

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOARD_SIZE = 8
LR = 3e-4
GAMMA, GAE_LAMBDA = 0.99, 0.95
CLIP_EPS, ENTROPY_COEF = 0.2, 0.02
ROLLOUT_STEPS = 2048
BATCH_SIZE = 128
UPDATE_EPOCHS = 4
RENDER = False
NUM_ENVS = 32

EPS = 10000
REWARD_SHAPING_MUL = 1.0

OUTPUT_DIR = f"outputs/snake"
os.makedirs(OUTPUT_DIR, exist_ok=True)
shutil.copy("train_snake.py", OUTPUT_DIR)
shutil.copy("run_snake.py", OUTPUT_DIR)
shutil.copy("snake_env.py", OUTPUT_DIR)

use_wandb = input("Use Weights & Biases? [y/n]: ").lower() == "y"
if use_wandb:
    import wandb
    wandb.init(
        project="rl",
        name=f"snake_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        config={
            "model_type": "transformer",
            "lr": LR,
            "rollout_steps": ROLLOUT_STEPS,
            "eps": EPS,
            "board_size": BOARD_SIZE,
        }
    )
    wandb.run.log_code("train_snake.py")
    wandb.run.log_code("snake_env.py")
    wandb.run.log_code("run_snake.py")

# --- MODELS ---

class SnakeTransformer(nn.Module):
    def __init__(self, size=8, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.size = size
        self.d_model = d_model

        # Each cell (pixel) on the 8x8 grid is a "token"
        # Input features: 3 (Empty, Snake, Food/Head)
        self.patch_embed = nn.Linear(3, d_model)

        # Learned Positional Embeddings - CRITICAL without CNNs
        self.pos_emb = nn.Parameter(torch.randn(1, size * size, d_model))

        # The Transformer Core
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.0, # Keep it deterministic for RL stability
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Heads
        self.actor = nn.Linear(d_model, 4)
        self.critic = nn.Linear(d_model, 1)

        self._init()

    def _init(self):
        # Orthogonal init is non-negotiable for Transformers in RL
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.orthogonal_(p, gain=1.0)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0)

    def forward(self, x):
        # x shape: (B, 3, 8, 8)
        b, c, h, w = x.shape

        # 1. Flatten the board into a sequence of tokens
        # (B, 3, 64) -> (B, 64, 3)
        tokens = x.view(b, c, h * w).permute(0, 2, 1)

        # 2. Project to d_model and add spatial information
        # Without pos_emb, the model literally wouldn't know the grid's shape
        x = self.patch_embed(tokens) + self.pos_emb

        # 3. Process with Global Self-Attention
        # Every cell looks at every other cell simultaneously
        out = self.transformer(x)

        # 4. Global Average Pooling (Latent Representation)
        global_feat = out.mean(dim=1)

        return self.actor(global_feat), self.critic(global_feat)

# --- TRAINING ENGINE ---

def make_env():
    return lambda: gym.make("Snake-v0", size=BOARD_SIZE)

def train():
    global REWARD_SHAPING_MUL

    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(NUM_ENVS)])
    eval_env = gym.make("Snake-v0", size=BOARD_SIZE, render_mode="human" if RENDER else None)
    model = SnakeTransformer(BOARD_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    best_len = 0
    current_phase = 1

    for iteration in range(1, EPS+1):
        # --- SCHEDULING ---
        frac = 1.0 - (iteration - 1.0) / EPS
        # current_ent_coef = ENTROPY_COEF * frac + 0.001 * (1.0 - frac)
        current_ent_coef = ENTROPY_COEF * (frac ** 2) + 0.0005

        # Scale LR relative to the phase
        current_lr = LR * max(frac, 0.0)

        # --- ROLLOUT ---
        states, actions, logprobs, rewards, values, masks, ep_lens = [], [], [], [], [], [], []
        state, _ = envs.reset()
        model.eval()

        # Gather standard PPO batches over parallel envs
        steps_per_env = ROLLOUT_STEPS // NUM_ENVS

        for idx in range(steps_per_env):
            st_t = torch.tensor(state, device=DEVICE, dtype=torch.float32)
            with torch.no_grad():
                logits, val = model(st_t)
                dist_m = dist.Categorical(logits=logits)
                action = dist_m.sample()

            next_state, reward, term, trunc, info = envs.step(action.cpu().numpy())
            # if iteration <= REWARD_SHAPING_EPS:
            #     reward += info["dist_reward"] * REWARD_SHAPING_MUL
            reward += info["dist_reward"] * REWARD_SHAPING_MUL
            done = term | trunc
            if done.any():
                ep_lens.extend(info["length"][done].tolist())

            states.append(state)
            actions.append(action.cpu().numpy())
            logprobs.append(dist_m.log_prob(action).cpu().numpy())
            rewards.append(reward)
            values.append(val.squeeze(-1).cpu().numpy())
            masks.append(1.0 - done.astype(np.float32))

            state = next_state

        # --- GAE & PPO UPDATE ---
        with torch.no_grad():
            _, next_v = model(torch.tensor(state, device=DEVICE, dtype=torch.float32))
            next_v = next_v.squeeze(-1).cpu().numpy()

        returns, gae = [], np.zeros(NUM_ENVS, dtype=np.float32)
        for i in reversed(range(steps_per_env)):
            delta = rewards[i] + GAMMA * next_v * masks[i] - values[i]
            gae = delta + GAMMA * GAE_LAMBDA * masks[i] * gae
            next_v = values[i]
            returns.insert(0, gae + values[i])

        states_np = np.array(states).reshape(steps_per_env * NUM_ENVS, 3, BOARD_SIZE, BOARD_SIZE)
        actions_np = np.array(actions).reshape(steps_per_env * NUM_ENVS)
        old_lp_np = np.array(logprobs).reshape(steps_per_env * NUM_ENVS)
        ret_np = np.array(returns).reshape(steps_per_env * NUM_ENVS)
        val_np = np.array(values).reshape(steps_per_env * NUM_ENVS)

        states_t = torch.tensor(states_np, device=DEVICE, dtype=torch.float32)
        actions_t = torch.tensor(actions_np, device=DEVICE, dtype=torch.long)
        old_lp_t = torch.tensor(old_lp_np, device=DEVICE, dtype=torch.float32)
        ret_t = torch.tensor(ret_np, device=DEVICE, dtype=torch.float32)
        val_t = torch.tensor(val_np, device=DEVICE, dtype=torch.float32)

        adv_t = ret_t - val_t
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        model.train()
        total_steps = steps_per_env * NUM_ENVS

        pg_losses, v_losses, ent_losses = [], [], []

        for _ in range(UPDATE_EPOCHS):
            indices = np.random.permutation(total_steps)
            for start in range(0, total_steps, BATCH_SIZE):
                idx = indices[start:start+BATCH_SIZE]
                l, v = model(states_t[idx])
                m = dist.Categorical(logits=l)

                # Policy Loss
                ratio = (m.log_prob(actions_t[idx]) - old_lp_t[idx]).exp()
                pg_loss = -torch.min(ratio * adv_t[idx], torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_t[idx]).mean()

                # Clipped Value Loss
                v_unclipped = (v.squeeze(-1) - ret_t[idx]) ** 2
                v_clipped = val_t[idx] + torch.clamp(v.squeeze(-1) - val_t[idx], -CLIP_EPS, CLIP_EPS)
                v_loss_clipped = (v_clipped - ret_t[idx]) ** 2
                v_loss_max = torch.max(v_unclipped, v_loss_clipped)
                value_loss = 0.5 * v_loss_max.mean()

                # Entropy
                entropy = m.entropy().mean()

                loss = pg_loss + value_loss - current_ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(value_loss.item())
                ent_losses.append(entropy.item())

        avg_l = np.mean(ep_lens) if ep_lens else 0
        print(f"[{current_phase}] Iter {iteration:03d} | Avg len: {avg_l:5.2f} | "
              f"Avg reward: {np.mean(rewards):7.4f} | "
              f"pL: {np.mean(pg_losses):.3f} vL: {np.mean(v_losses):.3f} ent: {np.mean(ent_losses):.3f}")
        if avg_l > 4.0 and REWARD_SHAPING_MUL > 0.0:
            REWARD_SHAPING_MUL = 0.0
            print("🐍 Snake can constantly eat the first apple!")
            print("Removed reward shaping ✨")

        if avg_l > best_len:
            best_len = avg_l
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best.pth"))

        # --- EVAL / RENDER ---
        if iteration % 10 == 0:
            model.eval()
            eval_state, _ = eval_env.reset()
            eval_total_reward = 0
            eval_steps = 0
            while True:
                if RENDER:
                    eval_env.render()
                with torch.no_grad():
                    obs = torch.tensor(eval_state, device=DEVICE, dtype=torch.float32).unsqueeze(0)
                    logits, _ = model(obs)
                    probs = torch.softmax(logits, dim=-1)
                    eval_act = torch.argmax(probs, dim=-1).item()
                eval_state, r, term, trunc, eval_info = eval_env.step(eval_act)
                eval_total_reward += r
                eval_steps += 1
                if term or trunc:
                    if RENDER:
                        eval_env.close()
                    break
            print(f"   ---> Eval Episode | Steps: {eval_steps} | Reward: {eval_total_reward:.2f} | Length: {eval_info.get('length', 'N/A')} <---")
            wandb.log({
                "metrics/eval_length": eval_info.get("length", 0.0),
                "metrics/eval_reward": eval_total_reward,
            }, step=iteration)
            model.train()
        if use_wandb:
            wandb.log({
                "metrics/avg_length": avg_l,
                "metrics/avg_reward": np.mean(rewards),
                "losses/policy_loss": np.mean(pg_losses),
                "losses/value_loss": np.mean(v_losses),
                "losses/entropy": np.mean(ent_losses),
                "charts/learning_rate": current_lr,
                "charts/entropy_coef": current_ent_coef,
            }, step=iteration)

if __name__ == "__main__":
    train()
