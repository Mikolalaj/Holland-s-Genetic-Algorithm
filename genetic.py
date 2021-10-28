
import random
import math
from bitstring import BitArray

random.seed(0)

class GeneticAlgorithm:
    def __init__(self, population: list, mut_prob: float, cros_prob: float, max_iter: int, pen_mul: float, chromosome_size: int):
        self.start_population = population
        self.pop_len = len(population)  # number of individuals in a population
        self.mut_prob = mut_prob  # mutation probability
        self.cros_prob = cros_prob  # crossover probability
        self.max_iter = max_iter  # max number of algorithm's iterations
        self.pen_mul = pen_mul  # penalty multiplier
        self.chrom_size = chromosome_size # number of bits in chromosome

    def start(self):
        population = self.start_population
        i = 0  # iterations counter
        # rating starting population
        ratings = self.rate_population(population)
        # finding the best individual in starting population
        best_x, best_rating = self.find_best(population, ratings)

        while i < self.max_iter:
            print(f'Iteration {i}')
            print(f'x    = {best_x}')
            print(f'f(x) = {best_rating}')
            # reproduction using tournament selection
            temp1 = self.reproduction(population, ratings)
            temp2 = self.crossover_mutation(temp1)  # using one point crossover
            ratings = self.rate_population(temp2)
            cur_best_x, cur_best_rating = self.find_best(temp2, ratings)
            if cur_best_rating < best_rating:
                best_rating = cur_best_rating
                best_x = cur_best_x
                '''print('Found new best x!')
                print(f'Iteration {i}')
                print(f'x    = {best_x}')
                print(f'f(x) = {best_rating}')'''
            population = temp2
            i += 1

        return best_x

    def penalty_function(self, x: tuple) -> float:
        min = -16
        max = 15
        sum = 0
        for i in x:
            if i < min:
                sum += i-min
            elif i > 15:
                sum += i-max
        return sum*self.pen_mul

    def fitness_function(self, x: tuple) -> float:
        function = (x[0]+2*x[1]-7)**2+(2*x[0]+x[1]-5)**2 + \
            (math.sin(1.5*x[2]))**3 + \
            ((x[2]-1)**2)*(1+(math.sin(1.5*x[3]))**2) + \
            ((x[3]-1)**2)*(1+(x[3])**2)
        penalty = self.penalty_function(x)
        return function + penalty

    def rate_population(self, population: list) -> list[float]:
        return [self.fitness_function(individual) for individual in population]

    def find_best(self, population: list, ratings: list) -> tuple:
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

    def reproduction(self, population, ratings) -> list:
        '''
        Tournament selection
        '''
        result_population = []
        for i in range(0, self.pop_len):
            j = random.randint(0, self.pop_len-1)
            if ratings[i] < ratings[j]:
                result_population.append(population[i])
            else:
                result_population.append(population[j])
        return result_population

    def crossover_mutation(self, population) -> list:
        crossed_populaton = self.one_point_crossover(population)
        mutated_population = self.mutation(crossed_populaton)
        return mutated_population

    def one_point_crossover(self, population) -> list:
        current_population = list(population)
        result_population = []
        while (len(current_population) > 1):
            first_parent = current_population.pop(random.randint(0, len(current_population)-1))
            second_parent = current_population.pop(random.randint(0, len(current_population)-1))
            if random.random() < self.cros_prob:
                result_population += self.cross_points(
                    first_parent, second_parent)
            else:
                result_population += [first_parent, second_parent]
        # if there is odd number of parents, then add last one to result population
        if len(current_population) == 1:
            result_population += current_population
        return result_population

    def cross_points(self, point_a: tuple, point_b: tuple) -> list[tuple, tuple]:
        new_a = []
        new_b = []
        for a, b in zip(point_a, point_b):
            crossed_numbers = self.cross(a, b)
            new_a.append(crossed_numbers[0])
            new_b.append(crossed_numbers[1])
        return [tuple(new_a), tuple(new_b)]

    def cross(self, a: int, b: int) -> list:
        cut_index = random.randint(1, self.chrom_size-1)

        a_bin = BitArray(int=a, length=self.chrom_size)
        b_bin = BitArray(int=b, length=self.chrom_size)

        c = a_bin[0:cut_index] + b_bin[cut_index:self.chrom_size]
        d = b_bin[0:cut_index] + a_bin[cut_index:self.chrom_size]

        return [c.int, d.int]

    def mutation(self, population) -> list:
        result_population = []
        for individual in population:
            new_individual = []
            for x in individual:
                binary = BitArray(int=x, length=self.chrom_size)
                for i in range(0, self.chrom_size):
                    if random.random() < self.mut_prob:
                        binary[i] = not binary[i]
                new_individual.append(binary.int)
            result_population.append(tuple(new_individual))
        return result_population
