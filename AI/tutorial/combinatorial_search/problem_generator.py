import optparse
import os
import numpy


def gen_deck_env_greedymove(k, d, pv):
    assert k == 312 and d == 15 and 0 <= pv <= 9
    # legend pirate
    if pv == 0:
        one_idx = numpy.array([15, 25, 32, 44, 59, 64, 122, 127, 150, 198, 224, 249, 256, 288, 297])
    elif pv == 1:
        one_idx = numpy.array([15, 37, 59, 64, 84, 122, 127, 150, 165, 198, 220, 224, 249, 256, 297])
    elif pv == 2:
        one_idx = numpy.array([2, 9, 32, 55, 122, 129, 150, 153, 184, 207, 267, 268, 282, 293, 302])
    elif pv == 3:
        one_idx = numpy.array([2, 9, 32, 37, 49, 96, 100, 122, 129, 150, 153, 184, 267, 282, 293])
    elif pv == 4:
        one_idx = numpy.array([1, 2, 9, 19, 32, 37, 100, 122, 129, 153, 170, 247, 267, 282, 293])
    elif pv == 5:
        one_idx = numpy.array([15, 39, 43, 51, 59, 64, 122, 127, 156, 198, 220, 249, 262, 288, 297])
    elif pv == 6:
        one_idx = numpy.array([2, 9, 26, 32, 55, 58, 122, 129, 150, 153, 184, 267, 282, 293, 302])
    elif pv == 7:
        one_idx = numpy.array([1, 2, 37, 100, 122, 129, 149, 150, 153, 184, 264, 267, 282, 293, 310])
    elif pv == 8:
        one_idx = numpy.array([15, 45, 59, 64, 122, 127, 150, 198, 220, 224, 229, 236, 249, 256, 297])
    elif pv == 9:
        one_idx = numpy.array([2, 9, 32, 55, 86, 100, 122, 129, 152, 153, 184, 267, 282, 293, 302])

    deck = numpy.zeros(k)
    deck[one_idx] = 1
    assert numpy.sum(deck) == d
    return deck


def gen_deck_env_gamestate(k, d, pv):
    assert k == 312 and d == 15 and 0 <= pv <= 9
    # from gen deck env_greedymove pv 9
    if pv == 0:
        one_idx = numpy.array([2, 9, 32, 55, 86, 100, 122, 129, 152, 153, 184, 267, 282, 293, 302])
    if pv == 1:
        one_idx = numpy.array([8, 10, 35, 39, 42, 48, 98, 125, 150, 199, 224, 252, 260, 302, 309])

    deck = numpy.zeros(k)
    deck[one_idx] = 1
    assert numpy.sum(deck) == d
    return deck


if __name__ == '__main__':
    numpy.set_printoptions(linewidth=10000)

    parser = optparse.OptionParser(usage="usage: %prog [options]")
    parser.add_option("--k", dest="k",
                      help="size of total cards",
                      type="int", default=200)
    parser.add_option("--d", dest="d",
                      help="size of deck",
                      type="int", default=30)
    parser.add_option("--env", dest="env",
                      help="environment",
                      type="string", default="env_kk")
    parser.add_option("--pv", dest="pv",
                      help="problem version",
                      type="int", default=0)
    parser.add_option("--env_seed", dest="env_seed",
                      help="environment seed (used to generate environment coefficient, etc.)",
                      type="int", default=0)

    (kwargs, args) = parser.parse_args()

    prob_dir = "test_probs/prob_{}_pv{}_envseed{}".format(kwargs.env, kwargs.pv, kwargs.env_seed)

    os.makedirs(prob_dir, exist_ok=False)

    if kwargs.env == 'env_nn':
        from environment.env_nn import Environment
        numpy.random.seed(kwargs.pv)  # use problem version to seed xo generation
        # xo = numpy.zeros(kwargs.k)
        # one_idx = numpy.random.choice(kwargs.k, kwargs.d, replace=False)
        # xo[one_idx] = 1
        env = Environment(k=kwargs.k, d=kwargs.d, COEF_SEED=kwargs.env_seed)
        # env.reset()
        env.save(prob_dir)
        print('env_nn generated.')
        print('env_nn.w1:', env.w1)
        print('env_nn.b1:', env.b1)
        print('env_nn.w2:', env.w2)
        print('env_nn.b2:', env.b2)
        print('env_nn.xo', env.x_o)
        print('env_nn.xp', env.x_p)
        print('env_nn.if_set_fixed_xo', env.if_set_fixed_xo())
        assert not env.if_set_fixed_xo()
    elif kwargs.env == 'env_nn_noisy':
        from environment.env_nn_noisy import Environment
        numpy.random.seed(kwargs.pv)  # use problem version to seed xo generation
        # xo = numpy.zeros(kwargs.k)
        # one_idx = numpy.random.choice(kwargs.k, kwargs.d, replace=False)
        # xo[one_idx] = 1
        env = Environment(k=kwargs.k, d=kwargs.d, COEF_SEED=kwargs.env_seed)
        # env.reset()
        env.save(prob_dir)
        print('env_nn generated.')
        print('env_nn.w1:', env.w1)
        print('env_nn.b1:', env.b1)
        print('env_nn.w2:', env.w2)
        print('env_nn.b2:', env.b2)
        print('env_nn.xo', env.x_o)
        print('env_nn.xp', env.x_p)
        print('env_nn.if_set_fixed_xo', env.if_set_fixed_xo())
        assert not env.if_set_fixed_xo()
    elif kwargs.env == 'env_greedymove':
        from environment.env_greedymove import Environment
        numpy.random.seed(kwargs.pv)  # use problem version to seed xo generation
        x_o = gen_deck_env_greedymove(kwargs.k, kwargs.d, kwargs.pv)
        env = Environment(k=kwargs.k, d=kwargs.d, fixed_xo=x_o)
        # we want the problem to have x_o as the generated deck,
        # however, we don't want it to be fixed_xo because some
        # algorithms requires fixed_xo = False
        env.unset_fixed_xo()
        env.save(prob_dir)
        print('env_greedymove generated.')
        print('env_greedymove.xo', env.x_o)
        print('env_greedymove.xp', env.x_p)
        print('env_greedymove.if_set_fixed_xo', env.if_set_fixed_xo())
        print('')
        assert not env.if_set_fixed_xo()
    elif kwargs.env == 'env_gamestate':
        from environment.env_gamestate import Environment
        numpy.random.seed(kwargs.pv)  # use problem version to seed xo generation
        x_o = gen_deck_env_gamestate(kwargs.k, kwargs.d, kwargs.pv)
        env = Environment(k=kwargs.k, d=kwargs.d, fixed_xo=x_o)
        # we want the problem to have x_o as the generated deck,
        # however, we don't want it to be fixed_xo because some
        # algorithms requires fixed_xo = False
        env.unset_fixed_xo()
        env.save(prob_dir)
        print('env_gamestate generated.')
        print('env_gamestate.xo', env.x_o)
        print('env_gamestate.xp', env.x_p)
        print('env_gamestate.if_set_fixed_xo', env.if_set_fixed_xo())
        print('')
        assert not env.if_set_fixed_xo()




