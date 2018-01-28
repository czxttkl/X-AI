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

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)
toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 1)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, k)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    return float(sum(individual)),


def evalOneMax(individual):
    global call_counts
    # barrier method to invalidate constrained violaion
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


def main():
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=1000,
                                   stats=stats, halloffame=hof, verbose=True)

    return pop, log, hof


if __name__ == "__main__":
    _, log, hof = main()
    ga_val, ga_x = env.still(evalOneMax(hof[0])[0]), hof[0]
    # monte carlo
    mc_val, mc_x, _, _, _ = env.monte_carlo(MONTE_CARLO_ITERATIONS=call_counts)

    print('ga stilled val:', ga_val)
    print('ga x*:', ga_x)
    print('ga call counts:', call_counts)
    print('monte carlo optimized val:', mc_val)
    print('monte carlo x*:', mc_x[k:-1])

