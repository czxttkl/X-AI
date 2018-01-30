"""
To be able to use bonmin and ipopt solver, you need to put the binaries into a folder belonging to
system PATH variable
"""
import rbfopt
import numpy as np
from environment.env_nn import Environment

k=200
d=30
env = Environment(k=k, d=d)
env.set_fixed_xo(env.x_o)
assert env.if_set_fixed_xo()
x_o = env.x_o
call_counts = 0
seed = 2008

np.random.seed(seed)


def output(x):
    global call_counts
    # barrier method to invalidate constrained violation
    if np.sum(x) != d:
        out = 0. + np.abs(np.sum(x) - d)
    else:
        # since rbf-opt is a minimizer, we need to make the output negative
        out = - env.output(np.hstack((x_o, x, 0)))  # the last one is step placeholder
    call_counts += 1
    print("{} call, x: {}, out: {}".format(call_counts, x, out))
    return out


bb = rbfopt.RbfoptUserBlackBox(k, np.array([0] * k), np.array([1] * k),
                               np.array(['I'] * k), output)

# since evaluating f(x) would require parallelism for the deck recommendation problem,
# we don't increase num_cpus here
settings = rbfopt.RbfoptSettings(max_evaluations=1e30, max_iterations=1e30,
                                 max_clock_time=7200, num_cpus=1)
alg = rbfopt.RbfoptAlgorithm(settings, bb)
# minimize
val, x, itercount, evalcount, faslt_evalcount = alg.optimize()
# monte carlo
mc_val, mc_x, _, _, _ = env.monte_carlo()

print('rbf optimized val:', val)
print('rbf distilled val:', env.still(-val))
print('rbf x*:', x)
print('call counts:', call_counts)
print('eval counts:', evalcount)
print('monte carlo optimized val:', mc_val)
print('monte carlo x*:', mc_x[k:-1])