import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
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
        # 3 channels: head, body, food
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, size, size), dtype=np.float32
        )

        self._snake = None
        self._direction = None
        self._food = None
        self._steps = None
        self._max_steps = None

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _build_obs(self):
        obs = np.zeros((3, self.size, self.size), dtype=np.float32)
        # ch0: head
        obs[0, self._snake[0][0], self._snake[0][1]] = 1.0
        # ch1: body (includes tail)
        for r, c in self._snake[1:]:
            obs[1, r, c] = 1.0
        # ch2: food
        obs[2, self._food[0], self._food[1]] = 1.0
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mid = self.size // 2
        self._snake = [(mid, mid + 1), (mid, mid), (mid, mid - 1)]
        self._direction = self.LEFT
        self._steps = 0
        self._max_steps = self.size * self.size * 4
        self._place_food()
        return self._build_obs(), {}

    def step(self, action):
        self._steps += 1

        if len(self._snake) > 1 and action == self.OPPOSITE.get(self._direction):
            action = self._direction
        self._direction = action

        head_r, head_c = self._snake[0]
        dr, dc = self.DIRS[action]
        new_head = (head_r + dr, head_c + dc)

        r, c = new_head
        if r < 0 or r >= self.size or c < 0 or c >= self.size:
            return self._build_obs(), -0.5, True, False, {"length": len(self._snake)}
        if new_head in self._snake:
            return self._build_obs(), -0.5, True, False, {"length": len(self._snake)}
        if self._steps >= self._max_steps:
            return self._build_obs(), 0.0, True, False, {"length": len(self._snake)}

        old_dist = self._manhattan(self._snake[0], self._food)
        ate_food = new_head == self._food
        self._snake.insert(0, new_head)

        if ate_food:
            self._place_food()
            reward = 1.0
        else:
            self._snake.pop()
            new_dist = self._manhattan(new_head, self._food)
            reward = 0.1 if new_dist < old_dist else -0.1

        return self._build_obs(), reward, False, False, {"length": len(self._snake)}

    def _place_food(self):
        empty = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) not in self._snake
        ]
        if empty:
            self._food = empty[self.np_random.integers(len(empty))]

    def render(self):
        if self.render_mode != "human":
            return
        import pygame

        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode(
                (self.size * self.cell_size, self.size * self.cell_size)
            )
            pygame.display.set_caption("Snake")
            self._clock = pygame.time.Clock()

        self._screen.fill(self.COLORS["bg"])
        cs = self.cell_size

        for r in range(self.size):
            for c in range(self.size):
                pygame.draw.rect(
                    self._screen,
                    self.COLORS["grid"],
                    (c * cs, r * cs, cs, cs),
                    1,
                )

        for i, (r, c) in enumerate(self._snake):
            if i == 0:
                color = self.COLORS["head"]
            elif i == len(self._snake) - 1:
                color = self.COLORS["tail"]
            else:
                color = self.COLORS["body"]
            pygame.draw.rect(self._screen, color, (c * cs + 1, r * cs + 1, cs - 2, cs - 2))

        fr, fc = self._food
        pygame.draw.rect(
            self._screen, self.COLORS["food"], (fc * cs + 1, fr * cs + 1, cs - 2, cs - 2)
        )

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None


gym.register(
    id="Snake-v0",
    entry_point="snake_env:SnakeEnv",
    kwargs={"size": 10},
)
