import random
from snake import Snake
from neural_network import NeuralNetwork
from config import GRID_WIDTH, GRID_HEIGHT

class GeneticAlgorithm:
    def __init__(self, size=50):
        self.size = size
        self.population = [Snake() for _ in range(size)]
        self.generation = 1
        self.best_fitness = 0
        self.history = []

    def evaluate(self):
        max_fit = 0
        for snake in self.population:
            food = self._get_random_food(snake)
            snake.reset()
            
            while snake.alive:
                snake.think(food)
                if snake.move(food):
                    food = self._get_random_food(snake)
            
            fitness = snake.calculate_fitness()
            if fitness > max_fit:
                max_fit = fitness
        
        self.best_fitness = max_fit
        self.history.append(max_fit)
        print(f"Generation {self.generation} Best Fitness: {max_fit:.2f}")

    def _get_random_food(self, snake):
        while True:
            food = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
            if food not in snake.positions:
                return food

    def select(self):
        # Tournament Selection
        selected = []
        for _ in range(self.size):
            # Select k random individuals
            participants = random.sample(self.population, 3)
            # Pick the one with best fitness
            winner = max(participants, key=lambda s: s.fitness)
            selected.append(winner)
        return selected

    def reproduce(self, selected):
        new_population = []
        
        # Elitism: Keep the absolute best from previous generation logic if we wanted,
        # but pure selection passed 'selected' which are parents.
        # Let's verify instructions: "Retourner un sous-ensemble des meilleurs individus" for select.
        # "Créer une nouvelle population à partir des serpents sélectionnés" for reproduce.
        
        # Let's handle Elitism by finding the best from current population before replacing
        best_snake = max(self.population, key=lambda s: s.fitness)
        
        # We need to construct new snakes.
        # We'll create size-1 children (since we keep 1 BEST)
        
        # Note: 'selected' comes from select(). 
        # If select returns a pool of parents (size=N), we pair them up.
        
        for _ in range(self.size - 1):
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            
            child_nn = NeuralNetwork.crossover(parent1.network, parent2.network)
            child_nn.mutate(rate=0.05) # Small mutation rate
            
            new_population.append(Snake(network=child_nn))
            
        # Add the best one from previous gen (Elitism)
        # We need a fresh copy of the best snake to reset its state, but keep network
        best_clone = Snake(network=best_snake.network)
        new_population.append(best_clone)
        
        self.population = new_population
        self.generation += 1
