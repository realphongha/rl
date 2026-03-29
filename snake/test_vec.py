import gymnasium as gym
import snake_env

def make_env():
    return lambda: gym.make("Snake-v0", size=8)

envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(3)])

envs.reset()

for i in range(100):
    actions = envs.action_space.sample()
    obs, rew, term, trunc, info = envs.step(actions)
    done = term | trunc
    if done.any():
        print("Done:", done)
        print("Info:", info)
        break
envs.close()
