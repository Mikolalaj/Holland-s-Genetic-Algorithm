
from genetic import GeneticAlgorithm
from statistics import stdev
import math
import random

def generate_random_population(size: int, min: int, max: int, dimension: int) -> list[tuple]:
    population = []
    for _ in range(size):
        individual = []
        for _ in range(dimension):
            individual.append(random.randint(min, max))
        population.append(tuple(individual))
    return population

def function(x: tuple) -> float:
    return (x[0]+2*x[1]-7)**2+(2*x[0]+x[1]-5)**2 + \
            (math.sin(1.5*x[2]))**3 + \
            ((x[2]-1)**2)*(1+(math.sin(1.5*x[3]))**2) + \
            ((x[3]-1)**2)*(1+(x[3])**2)

if __name__ == '__main__':
    values = []
    iters = []
    
    pop_size = 100
    mut_prob = 0.15
    cros_prob = 0.9

    print('\nPopulation size:', pop_size)
    print('Mutation probability:', mut_prob)
    print('Crossover probability:', cros_prob, '\n')
    
    for i in range(25):
        minimalization = GeneticAlgorithm(
            population=generate_random_population(pop_size, -16, 15, 4),
            mut_prob=mut_prob,
            cros_prob=cros_prob,
            max_iter=1000,
            pen_mul=1000,
            print_info=False,
            assumed_min=(1, 3, 1, 1),
            function=function
        )
        iter, point, value = minimalization.start()
        values.append(value)
        iters.append(iter)
        if point != minimalization.assumed_min:
            print(f'{i+1:>02}. - {point} - {value:.4}')
    
    print(f'\nAverage minimum = {sum(values)/len(values):.4}')
    print(f'Standard Deviation = {stdev(values):.4}')
    print(f'Average iterations = {sum(iters)/len(iters)}\n')
