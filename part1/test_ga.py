import random
from genetic_algorithm import GeneticAlgorithm

def test_ga():
    print("Initializing Genetic Algorithm...")
    ga = GeneticAlgorithm(size=20)
    
    print(f"Initial population size: {len(ga.population)}")
    
    # Run for 5 generations
    for i in range(5):
        print(f"\n--- Generation {ga.generation} ---")
        
        print("Evaluating...")
        ga.evaluate()
        print(f"Best Fitness: {ga.best_fitness}")
        
        print("Selecting...")
        selected = ga.select()
        print(f"Selected {len(selected)} parents")
        
        print("Reproducing...")
        ga.reproduce(selected)
        print(f"New population size: {len(ga.population)}")
        
    print("\nTest Complete!")

if __name__ == "__main__":
    test_ga()
