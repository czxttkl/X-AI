import optparse
import os
import numpy


def gen_deck(k, d, pv):
    assert k == 312 and d == 15 and 0 <= pv <= 3
    # legend pirate
    if pv == 0:
        one_idx = numpy.array([14,21,40,12,43,7,23,15,307,109,266,269,265,49,253])
    # trinity pirate
    elif pv == 1:
        one_idx = numpy.array([21,12,43,7,23,15,19,307,109,269,265,253,303,256,141])
    # taunt warrior
    elif pv == 2:
        one_idx = numpy.array([48,36,32,43,24,45,288,11,5,46,225,129,156,228,277])
    # c'Thun control
    elif pv == 3:
        one_idx = numpy.array([48,5,36,33,43,24,45,46,25,41,225,306,237,58,198])
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
        x_o = gen_deck(kwargs.k, kwargs.d, kwargs.pv)
        env = Environment(k=kwargs.k, d=kwargs.d, fixed_xo=x_o)
        # we want the problem to have x_o as the generated deck,
        # however, we don't want it to be fixed_xo because some
        # algorithms requires fixed_xo = False
        env.unset_fixed_xo()
        env.save(prob_dir)
        print('env_nn generated.')
        print('env_nn.xo', env.x_o)
        print('env_nn.xp', env.x_p)
        print('env_nn.if_set_fixed_xo', env.if_set_fixed_xo())
        print('')
        assert not env.if_set_fixed_xo()





