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

import array
import random
import time
import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from environment.env_nn import Environment


k=200
d=30
env = Environment(k=k, d=d)
env.set_fixed_xo(env.x_o)
assert env.if_set_fixed_xo()
x_o = env.x_o
call_counts = 0
seed = 2008
wall_time_limit = 60
version = 'long'

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)
toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 1)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, k)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# def evalOneMax(individual):
#     return float(sum(individual)),


def evalOneMax(individual):
    global call_counts
    # barrier method to invalidate constrained violation
    if numpy.sum(individual) != d:
        out = - numpy.abs(numpy.sum(individual) - d)
    else:
        out = env.output(numpy.hstack((x_o, individual, 0)))  # the last one is step placeholder
    call_counts += 1
    # print("{} call, x: {}, out: {}".format(call_counts, individual, out))
    return float(out),

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def main_short():
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=500,
                                   stats=stats, halloffame=hof, verbose=True)

    return pop, log, hof


def main_long():
    random.seed(64)

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=300)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Variable keeping track of the number of generations
    g = 0
    start_time = time.time()

    # Begin the evolution
    while True:
        # A new generation
        g = g + 1
        print("-- Generation {}, Time {} --".format(g, time.time()-start_time))

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

        if time.time() - start_time > wall_time_limit:
            break

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    # a fake hall of fame object
    hof = (best_ind,)
    return None, None, hof


if __name__ == "__main__":
    if version == 'long':
        _, _, hof = main_long()
    else:
        _, _, hof = main_short()

    ga_val, ga_x = env.still(evalOneMax(hof[0])[0]), hof[0]
    # monte carlo
    mc_val, mc_x, _, _, _ = env.monte_carlo(MONTE_CARLO_ITERATIONS=call_counts)

    print('ga stilled val:', ga_val)
    print('ga x*:', ga_x)
    print('ga x* deck size:', numpy.sum(ga_x))
    print('ga call counts:', call_counts)
    print('monte carlo optimized val:', mc_val)
    print('monte carlo x*:', mc_x[k:-1])

