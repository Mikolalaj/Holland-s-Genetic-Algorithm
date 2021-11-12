
import random
import math
import numpy as np

class GeneticAlgorithm:
    def __init__(self, population: list, mut_prob: float, cros_prob: float, max_iter: int, pen_mul: float, chromosome_size: int, print_info: bool, assumed_min: tuple):
        self.start_population = population
        self.pop_len = len(population)  # number of individuals in a population
        self.mut_prob = mut_prob  # mutation probability
        self.cros_prob = cros_prob  # crossover probability
        self.max_iter = max_iter  # max number of algorithm's iterations
        self.pen_mul = pen_mul  # penalty multiplier
        self.chrom_size = chromosome_size # number of bits in chromosome
        self.print_info = print_info
        self.assumed_min = assumed_min

    def start(self) -> tuple:
        population = self.start_population
        i = 0  # iterations counter
        # rating starting population
        ratings = self.rate_population(population)
        # finding the best individual in starting population
        best_x, best_rating = self.find_best(population, ratings)

        if self.print_info:
            print('Starting Algorithm')
        
        while i < self.max_iter and best_x != self.assumed_min:
            if self.print_info:
                progress = i*100/self.max_iter
                if progress%1 == 0:
                    print(f'Progress - {progress}%')
            # roulette selection
            temp1 = self.reproduction(population, ratings)
            # one point crossover and mutation
            temp2 = self.crossover_mutation(temp1)
            ratings = self.rate_population(temp2)
            cur_best_x, cur_best_rating = self.find_best(temp2, ratings)
            if cur_best_rating < best_rating:
                best_rating = cur_best_rating
                best_x = cur_best_x
                if self.print_info:
                    print('Found new best x!')
                    print(f'Iteration {i}')
                    print(f'x    = {best_x}')
                    print(f'f(x) = {best_rating}')
            population = temp2
            i += 1
        return i, best_x, best_rating

    def penalty_function(self, x: tuple) -> float:
        min = -16
        max = 15
        sum = 0
        for i in x:
            if i < min:
                sum += i-min
            elif i > 15:
                sum += i-max
        return abs(sum*self.pen_mul)

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
        Roulette selection
        '''
        mn = min(ratings)
        mx = max(ratings)
        
        # normalize ratings to range [0.1 - 1]
        # 0.1 for the biggest rating, 1 for the smallest
        probabilities = [(x-mx)/(mn-mx)+(0.1*(1-(x-mx)/(mn-mx))) for x in ratings]
        
        return random.choices(population, probabilities, k=self.pop_len)

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
                result_population += self.cross_points(first_parent, second_parent)
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

    def cross(self, a, b) -> list:
        a_bin, a_sign = self.int_to_bin(a)
        b_bin, b_sign = self.int_to_bin(b)
        
        a_len = len(a_bin)
        b_len = len(b_bin)
        
        if max(a_len, b_len) == 1:
            return [a, b]
        
        if a_len > b_len:
            diff = a_len - b_len
            b_bin.insert(0, 0*diff)
        elif b_len > a_len:
            diff = b_len - a_len
            a_bin.insert(0, 0*diff)
        
        cut_index = random.randint(1, len(a_bin)-1)

        c_bin = a_bin[:cut_index] + b_bin[cut_index:]
        d_bin = b_bin[:cut_index] + a_bin[cut_index:]

        return [self.bin_to_int(c_bin, a_sign), self.bin_to_int(d_bin, b_sign)]

    def mutation(self, population) -> list:
        result_population = []
        for individual in population:
            new_individual = []
            for x in individual:
                binary, sign = self.int_to_bin(x)
                for i, bit in enumerate(binary):
                    if random.random() < self.mut_prob:
                        binary[i] = abs(bit - 1)
                new_individual.append(self.bin_to_int(binary, sign))
            result_population.append(tuple(new_individual))
        return result_population

    def int_to_bin(self, x):
        bnx = np.binary_repr(x)
        neg = 0
        if bnx[0] == "-":
            bnx = bnx[1:]
            neg = 1
        binary_x = [int(l) for l in bnx]
        return binary_x, neg

    def bin_to_int(self, binary_x, neg):
        bin_x = np.array(binary_x)
        x = bin_x.dot(2**np.arange(bin_x.size)[::-1])
        x = x.item()
        if neg == 1:
            x = x - 2*x
        return x
