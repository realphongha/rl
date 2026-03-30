import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import deque

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    COLORS = {
        "bg": (30, 30, 30),
        "grid": (50, 50, 50),
        "head": (0, 200, 0),
        "body": (0, 140, 0),
        "tail": (0, 100, 0),
        "food": (200, 30, 30),
    }

    def __init__(self, size=10, render_mode=None, cell_size=40):
        super().__init__()
        self.size = size
        self.render_mode = render_mode
        self.cell_size = cell_size
        self._screen = None

        self.action_space = spaces.Discrete(4)
        # 3 channels: head, body (decay), food
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, size, size), dtype=np.float32
        )

        self._snake = None
        self._direction = None
        self._food = None
        self._steps = None
        self._max_steps = size * size * 2

    def _build_obs(self):
        length = len(self._snake)
        # Per your request, ensure the math below never hits a 1/0 case
        assert length >= 3, "Snake must have at least 3 segments"

        obs = np.zeros((3, self.size, self.size), dtype=np.float32)

        # ch0: head
        obs[0, self._snake[0][0], self._snake[0][1]] = 1.0

        # ch1: body decay logic (Tail is dimmest, neck is brightest)
        # Using (length - 2) is safe now because of the assertion
        for i in range(1, length):
            r, c = self._snake[i]
            obs[1, r, c] = 1.0 - ((i-1) / (length - 2)) * 0.8 # Range [0.2, 1.0]

        # ch2: food
        obs[2, self._food[0], self._food[1]] = 1.0
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mid = self.size // 2
        # Start with 3 segments: Head at (mid, mid+1), Body at (mid, mid), Tail at (mid, mid-1)
        self._snake = deque([(mid, mid + 1), (mid, mid), (mid, mid - 1)])
        self._direction = 3 # Start moving RIGHT
        self._steps = 0
        self._place_food()
        return self._build_obs(), {}

    def step(self, action):
        self._steps += 1
        self._direction = action # No more 180-degree protection! 😈

        head_r, head_c = self._snake[0]
        dr, dc = self.DIRS[action]
        new_head = (head_r + dr, head_c + dc)

        # 1. Check Collisions (Walls)
        if (new_head[0] < 0 or new_head[0] >= self.size or
            new_head[1] < 0 or new_head[1] >= self.size):
            return self._build_obs(), -1.0, True, False, {
                "length": len(self._snake), "dist_reward": 0.0
            }

        # 2. Check Collisions (Self)
        if new_head in self._snake:
            return self._build_obs(), -1.0, True, False, {
                "length": len(self._snake), "dist_reward": 0.0
            }

        # 3. Check Timeout
        if self._steps >= self._max_steps:
            return self._build_obs(), 0.0, True, False, {
                "length": len(self._snake), "dist_reward": 0.0
            }

        old_dist = manhattan(self._snake[0], self._food)

        self._snake.appendleft(new_head)

        # 4. Reward Logic
        dist_reward = 0.0
        if new_head == self._food:
            self._place_food()
            reward = 1.0
            # Reset steps on eat to reward activity
            self._steps = 0
        else:
            self._snake.pop()
            # Sparse reward: tiny living penalty to discourage looping
            reward = -0.01
            new_dist = manhattan(new_head, self._food)
            dist_reward = 0.01 if (new_dist < old_dist) else -0.01

        return self._build_obs(), reward, False, False, {
            "length": len(self._snake), "dist_reward": dist_reward
        }

    def _place_food(self):
        empty = [
            (r, c) for r in range(self.size) for c in range(self.size)
            if (r, c) not in self._snake
        ]
        if empty:
            self._food = empty[self.np_random.integers(len(empty))]

    def render(self):
        if self.render_mode != "human": return
        import pygame

        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((self.size * self.cell_size, self.size * self.cell_size))
            pygame.display.set_caption("Snake - Hard Mode")
            self._clock = pygame.time.Clock()

        self._screen.fill(self.COLORS["bg"])
        cs = self.cell_size
        for r in range(self.size):
            for c in range(self.size):
                pygame.draw.rect(self._screen, self.COLORS["grid"], (c*cs, r*cs, cs, cs), 1)

        for i, (r, c) in enumerate(self._snake):
            color = self.COLORS["head"] if i == 0 else (self.COLORS["tail"] if i == len(self._snake)-1 else self.COLORS["body"])
            pygame.draw.rect(self._screen, color, (c*cs+1, r*cs+1, cs-2, cs-2))

        fr, fc = self._food
        pygame.draw.rect(self._screen, self.COLORS["food"], (fc*cs+1, fr*cs+1, cs-2, cs-2))
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None

# Register the environment
gym.register(
    id="Snake-v0",
    entry_point=__name__ + ":SnakeEnv", # Assumes this code is in a file being run/imported
    kwargs={"size": 10},
)
