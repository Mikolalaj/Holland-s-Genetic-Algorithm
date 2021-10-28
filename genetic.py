
import random
import math
from bitstring import BitArray


def penalty_function(x: tuple) -> float:
    min = -16
    max = 15
    sum = 0
    for i in x:
        if i < min:
            sum += i-min
        elif i > 15:
            sum += i-max
    return sum*PEN_MUL


def fitness_function(x: tuple) -> float:
    function = (x[0]+2*x[1]-7)**2+(2*x[0]+x[1]-5)**2+(math.sin(1.5*x[2]))**3 + \
        ((x[2]-1)**2)*(1+(math.sin(1.5*x[3]))**2)+((x[3]-1)**2)*(1+(x[3])**2)
    penalty = penalty_function(x)
    return function + penalty


def rate_population(population: list) -> list[float]:
    return [fitness_function(individual) for individual in population]


def find_best(population: list, ratings: list) -> tuple:
    '''
    Find the best (the smallest rating) individual in
    given population based on ratings. Return tuple
    (best_individual, its_rating)
    '''
    pop_len = len(population)
    rat_len = len(ratings)
    if pop_len != rat_len:
        raise ValueError(
            f'Size of population and its ratings must be the same ({pop_len}!={rat_len})')
    if pop_len == 0:
        raise ValueError('Population must not be empty')
    return population[ratings.index(min(ratings))], min(ratings)


def reproduction(population, ratings, pop_len) -> list:
    '''
    Tournament selection
    '''
    result_population = []
    for i in range(0, pop_len):
        j = random.randint(0, pop_len-1)
        print(f'i={i}, j={j}')
        print(f'{ratings[i]}, {ratings[j]}')
        print(f'{population[i]}, {population[j]}')
        if ratings[i] < ratings[j]:
            result_population.append(population[i])
        else:
            result_population.append(population[j])
        print(result_population)
    return result_population


def crossover_mutation(population, pop_len) -> list:
    crossed_populaton = one_point_crossover(population, pop_len)
    mutated_population = mutation(crossed_populaton)
    return mutated_population


def one_point_crossover(population, pop_len) -> list:
    current_population = list(population)
    result_population = []
    while (len(current_population) > 1):
        first_parent = population.pop(random.randint(0, pop_len-1))
        second_parent = population.pop(random.randint(0, pop_len-1))
        if random.random() < CROS_PROB:
            result_population += cross_points(first_parent, second_parent)
        else:
            result_population += [first_parent, second_parent]
    # if there is odd number of parents, then add last one to result population
    if len(current_population) == 1:
        result_population += current_population
    return result_population


def cross_points(point_a: tuple, point_b: tuple) -> list[tuple, tuple]:
    new_a = []
    new_b = []
    for a, b in zip(point_a, point_b):
        crossed_numbers = cross(a, b)
        new_a.append(crossed_numbers[0])
        new_b.append(crossed_numbers[1])
    return [tuple(new_a), tuple(new_b)]


def cross(a: int, b: int) -> list:
    cut_index = random.randint(1, 4)

    a_bin = BitArray(int=a, length=5)
    b_bin = BitArray(int=b, length=5)

    c = a_bin[0:cut_index] + b_bin[cut_index:5]
    d = b_bin[0:cut_index] + a_bin[cut_index:5]

    return [c.int, d.int]


def mutation(population) -> list:
    result_population = []
    for individual in population:
        new_individual = []
        for x in individual:
            binary = BitArray(int=x, length=5)
            for i in range(0, 6):
                if random.random() < MUT_PROB:
                    binary[i] = not binary[i]
            new_individual.append(binary.int)
        result_population.append(tuple(new_individual))
    return result_population


MUT_PROB = 0  # mutation probability
CROS_PROB = 0  # crossover probability
MAX_ITER = 0  # max number of algorithm's iterations
PEN_MUL = 1  # penalty multiplier


def genetic_algorithm(population):
    pop_len = len(population)  # number of individuals in a population
    i = 0  # iterations counter
    ratings = rate_population(population)  # rating starting population
    # finding the best individual in starting population
    best_x, best_rating = find_best(population, ratings)

    while i < MAX_ITER:
        # reproduction using tournament selection
        temp1 = reproduction(population, ratings, pop_len)
        temp2 = crossover_mutation(temp1, pop_len)  # using one point crossover
        ratings = rate_population(temp2)
        cur_best_x, cur_best_rating = find_best(temp2, ratings)
        if cur_best_rating < best_rating:
            best_rating = cur_best_rating
            best_x = cur_best_x
        population = temp2
        i += 1

    return best_x
