#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import random
import time
import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from environment.env_nn import Environment
numpy.set_printoptions(linewidth=10000, precision=5)


k=312
d=15
env = Environment(k=k, d=d)
env.set_fixed_xo(env.x_o)
assert env.if_set_fixed_xo()
x_o = env.x_o
call_counts = 0
seed = 2008
wall_time_limit = 1
version = 'long'
pop_size = 10
hof_size = 10
CXPB = 0.2     # the probability with which two individuals are crossed
MUTPB = 0.2    # the probability for mutating an individual


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", numpy.ndarray, typecode='b', fitness=creator.FitnessMax)
toolbox = base.Toolbox()
# Structure initializers
def my_individual_creator():
    random_xp = numpy.zeros(k, dtype=numpy.int8)  # state + step
    one_idx = numpy.random.choice(k, d, replace=False)
    random_xp[one_idx] = 1
    return random_xp
toolbox.register("my_individual_creator", my_individual_creator)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.my_individual_creator)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)



def evalOneMax(individual):
    global call_counts
    # barrier method to invalidate constrained violation
    if numpy.sum(individual) != d:
        out = - numpy.abs(numpy.sum(individual) - d)
    else:
        # in GA, we do not use distill output
        out = env.still(env.output(numpy.hstack((x_o, individual, 0))))  # the last one is step placeholder
        call_counts += 1
    return float(out),

toolbox.register("evaluate", evalOneMax)


def cxTwoPoint(ind1, ind2):
    """
    Exchange non-zero indices
    """
    ind1_idx = numpy.nonzero(ind1)[0]
    ind2_idx = numpy.nonzero(ind2)[0]
    ind1_idx_not_in_ind2 = numpy.setdiff1d(ind1_idx, ind2_idx)
    ind2_idx_not_in_ind1 = numpy.setdiff1d(ind2_idx, ind1_idx)

    size = len(ind1_idx_not_in_ind2)
    if size  == 0:
        return ind1, ind2

    size_of_sample = numpy.random.randint(1, size + 1)
    ind1_idx_sample = numpy.random.choice(ind1_idx_not_in_ind2,
                                          size=size_of_sample, replace=False)
    ind2_idx_sample = numpy.random.choice(ind2_idx_not_in_ind1,
                                          size=size_of_sample, replace=False)

    new_ind1_idx = numpy.union1d(
        numpy.setdiff1d(ind1_idx, ind1_idx_sample),
        ind2_idx_sample)
    new_ind2_idx = numpy.union1d(
        numpy.setdiff1d(ind2_idx, ind2_idx_sample),
        ind1_idx_sample)
    new_ind1 = numpy.zeros(len(ind1), dtype=ind1.dtype)
    new_ind2 = numpy.zeros(len(ind2), dtype=ind2.dtype)
    new_ind1[new_ind1_idx] = 1
    new_ind2[new_ind2_idx] = 1
    return new_ind1, new_ind2


def mutShuffleIndexes(individual):
    """
    Exchange zero index with non-zero index
    """
    zero_idx = numpy.where(individual == 0)[0]
    one_idx = numpy.where(individual == 1)[0]
    individual[numpy.random.choice(zero_idx)] = 1
    individual[numpy.random.choice(one_idx)] = 0
    return individual,


toolbox.register("mate", cxTwoPoint)
toolbox.register("mutate", mutShuffleIndexes)
toolbox.register("select", tools.selTournament, tournsize=3)


def select_best_from_hof(hof):
    res = []
    for ind_x in hof:
        noiseless_val = env.still(env.output_noiseless(numpy.hstack((x_o, ind_x, 0))))
        res.append((noiseless_val, ind_x))
    best_noiseless_val, best_ind_x = max(res, key=lambda x: x[0])
    return best_noiseless_val, best_ind_x


def main_short():
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(hof_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50,
                                   stats=stats, halloffame=hof, verbose=True)

    return select_best_from_hof(hof)


def main_long():
    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=pop_size)

    print("Start of evolution")
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    print("  Evaluated %i individuals \n" % len(pop))

    # Variable keeping track of the number of generations
    g = 0
    start_time = time.time()

    while True:
        # A new generation
        g = g + 1
        print("-- Generation {}, Time {} --".format(g, time.time()-start_time))

        # for p in pop:
        #     print(p)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        print(" Call counts %s\n" % call_counts)

        # select best ind in all generations
        # this may end up with a solution with a low mean but high variance
        # best_ind_cand = tools.selBest(pop, 1)[0]
        # best_ind_cand_val = best_ind_cand.fitness.values[0]
        # if best_ind_cand_val > best_ind_val:
        #     best_ind_val = best_ind_cand_val
        #     best_ind = best_ind_cand

        if time.time() - start_time > wall_time_limit:
            break

    print("-- End of (successful) evolution --")

    best_noiseless_val, best_ind_x = select_best_from_hof(tools.selBest(pop, hof_size))
    # print("Best individual is %s\n %s" % (best_noiseless_val, best_ind_x))

    return best_noiseless_val, best_ind_x


if __name__ == "__main__":
    if version == 'long':
        best_ga_noiseless_val, ga_x = main_long()
    else:
        best_ga_noiseless_val, ga_x = main_short()

        # monte carlo (noiseless)
    mc_val, mc_x, _, _, _, _ = env.monte_carlo(MONTE_CARLO_ITERATIONS=call_counts)

    print('ga stilled noiseless val:', best_ga_noiseless_val)
    print('ga x*:', ga_x)
    print('ga x* deck size:', numpy.sum(ga_x))
    print('ga call counts:', call_counts)
    print('monte carlo optimized val:', mc_val)
    print('monte carlo x*:', mc_x[k:-1])

