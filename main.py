
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
    iters = []
    for i in range(25):
        minimalization = GeneticAlgorithm(
            population=generate_random_population(200, -16, 15, 4),
            mut_prob=0.2,
            cros_prob=0.9,
            max_iter=1000,
            pen_mul=1000,
            chromosome_size=5,
            print_info=False,
            assumed_min=(1, 3, 1, 1),
        )
        iter, point, value = minimalization.start()
        values.append(value)
        iters.append(iter)
        print(f'{i+1}. - {point} - {value}')
    
    print(f'Average minimum = {sum(values)/len(values)}')
    print(f'Standard Deviation = {stdev(values)}')
    print(f'Average iterations = {sum(iters)/len(iters)}')


# minimum at (1, 3, 1, 1), f(x) = 0.9925037693693152
