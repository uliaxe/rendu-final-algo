# snake.py
import random
import numpy as np
from neural_network import NeuralNetwork
from config import GRID_WIDTH, GRID_HEIGHT

DIRECTIONS = [(0,-1), (1,0), (0,1), (-1,0)]

class Snake:
    def __init__(self, network=None):
        self.network = network or NeuralNetwork()
        self.reset()

    def reset(self):
        self.positions = [(GRID_WIDTH//2, GRID_HEIGHT//2)]
        self.direction = random.choice(DIRECTIONS)
        self.length = 3
        self.alive = True
        self.score = 0
        self.steps = 0
        self.steps_without_food = 0
        self.fitness = 0

    def get_head(self):
        return self.positions[0]

    def turn(self, d):
        if (-d[0], -d[1]) != self.direction:
            self.direction = d

    def move(self, food):
        if not self.alive:
            return False

        x, y = self.get_head()
        nx = (x + self.direction[0]) % GRID_WIDTH
        ny = (y + self.direction[1]) % GRID_HEIGHT
        new = (nx, ny)

        if new in self.positions:
            self.alive = False
            return False

        self.positions.insert(0, new)

        if new == food:
            self.score += 1
            self.length += 1
            self.steps_without_food = 0
            return True

        self.positions = self.positions[:self.length]
        self.steps += 1
        self.steps_without_food += 1

        if self.steps_without_food > 120:
            self.alive = False

        return False

    def get_vision(self, food):
        hx, hy = self.get_head()
        fx, fy = food

        vision = [
            (fx - hx) / GRID_WIDTH,
            (fy - hy) / GRID_HEIGHT
        ]

        for dx, dy in DIRECTIONS:
            nx, ny = (hx + dx) % GRID_WIDTH, (hy + dy) % GRID_HEIGHT
            vision.append(1 if (nx, ny) in self.positions else 0)

        for d in DIRECTIONS:
            vision.append(1 if self.direction == d else 0)

        vision.append(self.length / (GRID_WIDTH * GRID_HEIGHT))
        vision += list(self.direction)

        return vision

    def think(self, food):
        out = self.network.forward(self.get_vision(food))
        self.turn(DIRECTIONS[np.argmax(out)])

    def calculate_fitness(self):
        self.fitness = (
            self.score * 100
            + self.steps * 0.1
            - self.steps_without_food * 0.2
        )
        return max(0, self.fitness)
