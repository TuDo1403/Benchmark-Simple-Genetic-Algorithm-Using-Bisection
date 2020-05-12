import numpy as np
from enum import Enum
import fitness_function as ff

def initialize_population(pop_size, problem_size):
    population = np.random.randint(low=0, high=2, size=(pop_size, problem_size))
    return population


def evaluate(pop, f_func):
    return np.array(list(map(f_func, pop)))

from copy import deepcopy
def variate(pop, crossover_mode):
    (num_inds, num_params) = np.shape(pop)
    indices = np.array(range(num_inds))

    offsprings = []
    np.random.shuffle(indices)

    for i in range(0, num_inds, 2):
        index1 = indices[i]
        index2 = indices[i+1]
        offspring1 = pop[index1].tolist()
        offspring2 = pop[index2].tolist()

        for j in range(num_params):
            if np.random.randint(low=0, high=2) == 1:
                if crossover_mode == Crossover.ONEPOINT:
                    offspring1[:j], offspring2[:j] = deepcopy(offspring2[:j]), offspring1[:j]
                    break
                else:
                    offspring1[j], offspring2[j] = offspring2[j], offspring1[j]

        offsprings.append(np.array(offspring1))
        offsprings.append(np.array(offspring2))

    return np.reshape(offsprings, (num_inds, num_params))

def tournament_selection(pool_fitness, tournament_size, selection_size):
    num_individuals = len(pool_fitness)
    indices = np.array(range(num_individuals))
    selected_indices = []

    while len(selected_indices) < selection_size:
        np.random.shuffle(indices)

        for i in range(0, num_individuals, tournament_size):
            idx_tournament = indices[i:i+tournament_size]
            winner = list(filter(lambda x : pool_fitness[x] == max(pool_fitness[idx_tournament]), idx_tournament))
            selected_indices.append(np.random.choice(winner))

    return selected_indices

def POPOP(user_options, func_inf, seed_num=1):
    np.random.seed(seed_num)

    population = initialize_population(user_options.POP_SIZE, user_options.PROBLEM_SIZE)
    pop_fitness = evaluate(population, func_inf.F_FUNC)
    num_eval_func_calls = len(pop_fitness)

    selection_size = len(population)
    generation = 0

    while len(np.unique(pop_fitness)) != 1:
        offsprings = variate(population, user_options.CROSSOVER_MODE)
        off_fitness = evaluate(offsprings, func_inf.F_FUNC)
        num_eval_func_calls += len(off_fitness)

        pool = np.vstack((population, offsprings))
        pool_fitness = np.hstack((pop_fitness, off_fitness))

        pool_indices = tournament_selection(pool_fitness, user_options.TOURNAMENT_SIZE, selection_size)
        population = pool[pool_indices]
        pop_fitness = pool_fitness[pool_indices]

        generation += 1

    optimized_solution_found = user_options.PROBLEM_SIZE == max(pop_fitness)
    print(optimized_solution_found, num_eval_func_calls)
    return (optimized_solution_found, num_eval_func_calls)

class Crossover(Enum):
    ONEPOINT = 1
    UX = 2

class POPOPConfig:
    PROBLEM_SIZE = 4
    POP_SIZE = 4
    TOURNAMENT_SIZE = 4
    CROSSOVER_MODE = Crossover.UX

    def __init__(self, pop_size, problem_size=4, tournament_size=4, crossover_mode=Crossover.UX):
        self.PROBLEM_SIZE = problem_size
        self.POP_SIZE = pop_size
        self.TOURNAMENT_SIZE = tournament_size
        self.CROSSOVER_MODE = crossover_mode

        





