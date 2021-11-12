
from genetic import GeneticAlgorithm
from statistics import stdev
import random

def generate_random_population(size: int, min: int, max: int, dimension: int) -> list[tuple]:
    population = []
    for _ in range(size):
        individual = []
        for _ in range(dimension):
            individual.append(random.randint(min, max))
        population.append(tuple(individual))
    return population

if __name__ == '__main__':
    values = []
    for i in range(25):
        minimalization = GeneticAlgorithm(
            population=generate_random_population(50, -16, 15, 4),
            mut_prob=0.2,
            cros_prob=0.9,
            max_iter=1000,
            pen_mul=1000,
            chromosome_size=5,
            print_info=False,
            assumed_min=(1, 3, 1, 1),
        )
        result = minimalization.start()
        values.append(result[1])
        print(f'{i+1}. - {result[0]} - {result[1]}')
    
    print(f'Average = {sum(values)/len(values)}')
    print(f'Standard Deviation = {stdev(values)}')


# minimum at (1, 3, 1, 1), f(x) = 0.9925037693693152
