import random
import time
import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from environment.env_nn_noisy import Environment
from genetic_algorithm import cxTwoDeck, mutSwapCard, my_deck_creator_func, select_best_from_hof
numpy.set_printoptions(linewidth=10000, precision=5)


k=312
d=15
env = Environment(k=k, d=d)
env.set_fixed_xo(env.x_o)
assert env.if_set_fixed_xo()
call_counts = 0
seed = 2008
wall_time_limit = 3
version = 'long'
pop_size = 10
hof_size = 10
CXPB = 0.2     # the probability with which two individuals are crossed
MUTPB = 0.2    # the probability for mutating an individual


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", numpy.ndarray, typecode='b', fitness=creator.FitnessMax)
toolbox = base.Toolbox()
my_deck_creator = my_deck_creator_func(k, d)
toolbox.register("my_individual_creator", my_deck_creator)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.my_individual_creator)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalOneMax(individual):
    global call_counts
    # barrier method to invalidate constrained violation
    if numpy.sum(individual) != d:
        out = - numpy.abs(numpy.sum(individual) - d)
    else:
        # in GA, we do not use distill output
        out = env.still(env.output(numpy.hstack((env.x_o, individual, 0))))  # the last one is step placeholder
        call_counts += 1
    return float(out),

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", cxTwoDeck)
toolbox.register("mutate", mutSwapCard)
toolbox.register("select", tools.selTournament, tournsize=3)


def main_short():
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(hof_size, similar=numpy.array_equal)
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

    best_noiseless_val, best_ind_x = select_best_from_hof(tools.selBest(pop, hof_size), env)
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

