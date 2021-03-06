import optparse
import os
import numpy


def gen_deck_env_greedymove(k, d, pv):
    assert k == 312 and d == 15 and 0 <= pv <= 19
    # legend pirate
    if pv == 0:
        one_idx = numpy.array([15, 32, 33, 47, 64, 71, 122, 150, 170, 249, 256, 288, 294, 297, 298])
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
    elif pv == 10:
        one_idx = numpy.array([41, 90, 99, 129, 140, 151, 163, 175, 188, 217, 249, 253, 275, 279, 291])
    elif pv == 11:
        one_idx = numpy.array([2, 6, 8, 35, 41, 43, 45, 53, 90, 101, 134, 154, 252, 274, 295])
    elif pv == 12:
        one_idx = numpy.array([30, 41, 90, 95, 129, 151, 163, 175, 188, 217, 249, 268, 275, 279, 290])
    elif pv == 13:
        one_idx = numpy.array([32, 41, 87, 90, 99, 129, 140, 151, 163, 188, 194, 209, 249, 275, 279])
    elif pv == 14:
        one_idx = numpy.array([2, 6, 8, 33, 41, 46, 80, 90, 134, 154, 182, 215, 226, 253, 290])
    elif pv == 15:
        one_idx = numpy.array([8, 35, 38, 39, 41, 42, 92, 98, 148, 195, 199, 229, 260, 267, 301])
    elif pv == 16:
        one_idx = numpy.array([6, 15, 26, 32, 51, 53, 64, 92, 122, 127, 150, 224, 288, 297, 301])
    elif pv == 17:
        one_idx = numpy.array([41, 90, 99, 129, 140, 151, 157, 163, 175, 188, 249, 253, 275, 279, 291])
    elif pv == 18:
        one_idx = numpy.array([15, 20, 32, 39, 40, 64, 73, 119, 150, 198, 220, 224, 227, 256, 297])
    elif pv == 19:
        one_idx = numpy.array([6, 15, 32, 44, 64, 122, 127, 134, 146, 196, 198, 224, 227, 281, 308])

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
    # random naive
    if pv == 2:
        one_idx = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    if pv == 3:
        one_idx = numpy.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
    # best result from pv1
    if pv == 4:
        one_idx = numpy.array([28, 56, 65, 77, 85, 115, 120, 150, 153, 198, 204, 212, 249, 285, 302])
    # rlprtr result from pv2
    if pv == 5:
        one_idx = numpy.array([2, 14, 53, 63, 66, 160, 164, 188, 204, 224, 241, 258, 266, 300, 303])


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
        xo = numpy.zeros(kwargs.k)
        one_idx = numpy.random.choice(kwargs.k, kwargs.d, replace=False)
        xo[one_idx] = 1
        env = Environment(k=kwargs.k, d=kwargs.d, COEF_SEED=kwargs.env_seed, fixed_xo=xo)
        env.save(prob_dir)
        print('env_nn generated.')
        print('env_nn.w1:', env.w1)
        print('env_nn.b1:', env.b1)
        print('env_nn.w2:', env.w2)
        print('env_nn.b2:', env.b2)
        print('env_nn.xo', env.x_o)
        print('env_nn.xp', env.x_p)
        print('env_nn.if_set_fixed_xo', env.if_set_fixed_xo())
        assert env.if_set_fixed_xo()
    elif kwargs.env == 'env_nn_noisy':
        from environment.env_nn_noisy import Environment
        numpy.random.seed(kwargs.pv)  # use problem version to seed xo generation
        env = Environment(k=kwargs.k, d=kwargs.d, COEF_SEED=kwargs.env_seed)
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




