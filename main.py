
from genetic import GeneticAlgorithm
import random

random.seed(0)

def generate_random_population(size: int, min: int, max: int, dimension: int) -> list[tuple]:
    population = []
    for _ in range(size):
        individual = []
        for _ in range(dimension):
            individual.append(random.randint(min, max))
        population.append(tuple(individual))
    return population
    
if __name__ == '__main__':
    minimalization = GeneticAlgorithm(
        population=generate_random_population(2000, -16, 15, 4),
        mut_prob=0.2,
        cros_prob=0.8,
        max_iter=10000,
        pen_mul=100,
        chromosome_size=5
    )
    minimalization.start()
    