import numpy


def cxTwoDeck(ind1, ind2):
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
    ind1[:] = 0
    ind2[:] = 0
    ind1[new_ind1_idx] = 1
    ind2[new_ind2_idx] = 1
    return ind1, ind2


def mutSwapCard(individual):
    """
    Exchange zero index with non-zero index
    """
    zero_idx = numpy.where(individual == 0)[0]
    one_idx = numpy.where(individual == 1)[0]
    individual[numpy.random.choice(zero_idx)] = 1
    individual[numpy.random.choice(one_idx)] = 0
    return individual,


def my_deck_creator_func(k, d):
    def my_deck_creator():
        random_xp = numpy.zeros(k, dtype=numpy.int8)  # state + step
        one_idx = numpy.random.choice(k, d, replace=False)
        random_xp[one_idx] = 1
        return random_xp
    return my_deck_creator


def select_best_from_hof(hof, env):
    assert env.if_set_fixed_xo()
    res = []
    for ind_x in hof:
        noiseless_val = env.still(env.output_noiseless(numpy.hstack((env.x_o, ind_x, 0))))
        res.append((noiseless_val, ind_x))
    best_noiseless_val, best_ind_x = max(res, key=lambda x: x[0])
    return best_noiseless_val, best_ind_x
