import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
import gymnasium as gym
import snake_env
import os
import argparse
from train_snake import SnakeHybridNet

ENV_NAME = "Snake-v0"


def play(args):
    DEVICE = torch.device(args.device)
    MODEL_PATH = args.model
    BOARD_SIZE = args.size
    NUM_EPISODES = args.num_episodes
    env = gym.make(ENV_NAME, size=BOARD_SIZE, render_mode="human")

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        print("Run train_snake.py first to train a model.")
        return

    model = SnakeHybridNet(BOARD_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")

    for ep in range(NUM_EPISODES):
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        while True:
            env.render()

            with torch.no_grad():
                obs = torch.tensor(state, device=DEVICE, dtype=torch.float32).unsqueeze(0)
                logits, _ = model(obs)
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=-1).item()

            state, reward, term, trunc, info = env.step(action)
            total_reward += reward
            steps += 1

            if term or trunc:
                break
        env.close()

        print(f"Episode {ep + 1}/{NUM_EPISODES} | Steps: {steps} | Reward: {total_reward:.2f} | Length: {info['length']}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the model file")
    parser.add_argument("--size", type=int, default=8, help="Size of the board")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to play")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()
    play(args)

