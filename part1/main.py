# main.py
import pygame
from config import *
from genetic_algorithm import GeneticAlgorithm
from game import Game
from utils import plot_fitness

def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    ga = GeneticAlgorithm()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Visualize the best snake from the previous generation (Elitism puts it at the end)
        # Note: In Gen 1, this is just a random snake.
        snake = ga.population[-1]
        
        best_of_all = max(ga.history) if ga.history else 0
        game = Game(snake, generation=ga.generation, best_fitness=best_of_all)

        while snake.alive:
            game.update()
            game.draw(screen)
            pygame.display.flip()
            clock.tick(FPS)

        ga.evaluate()
        selected = ga.select()
        ga.reproduce(selected)

    plot_fitness(ga.history)
    pygame.quit()

if __name__ == "__main__":
    main()
