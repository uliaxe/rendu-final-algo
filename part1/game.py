# game.py
import random
import pygame
from config import *

class Game:
    def __init__(self, snake, generation=1, best_fitness=0):
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 20)
        self.snake = snake
        self.food = self.spawn_food()
        self.generation = generation
        self.best_fitness = best_fitness

    def spawn_food(self):
        while True:
            p = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
            if p not in self.snake.positions:
                return p

    def update(self):
        self.snake.think(self.food)
        if self.snake.move(self.food):
            self.food = self.spawn_food()

    def draw(self, screen):
        screen.fill(BLACK)

        for x, y in self.snake.positions:
            pygame.draw.rect(
                screen, GREEN,
                (x*GRID_SIZE, y*GRID_SIZE, GRID_SIZE, GRID_SIZE)
            )

        fx, fy = self.food
        pygame.draw.rect(
            screen, RED,
            (fx*GRID_SIZE, fy*GRID_SIZE, GRID_SIZE, GRID_SIZE)
        )
        
        # Draw Score
        score_text = self.font.render(f"Score: {self.snake.score}", True, (255, 255, 255))
        screen.blit(score_text, (5, 5))
        
        # Draw Generation
        gen_text = self.font.render(f"Gen: {self.generation}", True, (255, 255, 255))
        screen.blit(gen_text, (5, 30))

        # Draw Record
        record_text = self.font.render(f"Best Fitness: {self.best_fitness:.1f}", True, (255, 255, 255))
        screen.blit(record_text, (5, 55))
