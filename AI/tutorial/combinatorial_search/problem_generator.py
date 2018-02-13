import optparse
import os
import numpy


def gen_deck(k, d, pv):
    assert k == 312 and d == 15 and 0 <= pv <= 11
    # the first 4 decks are expert curated from icy-veins website
    # legend pirate
    if pv == 0:
        one_idx = numpy.array([7,12,14,15,21,23,40,43,49,109,253,265,266,269,307])
    # trinity pirate
    elif pv == 1:
        one_idx = numpy.array([7,12,15,19,21,23,43,109,141,253,256,265,269,303,307])
    # taunt warrior
    elif pv == 2:
        one_idx = numpy.array([5,11,24,32,36,43,45,46,48,129,156,225,228,277,288])
    # c'Thun control
    elif pv == 3:
        one_idx = numpy.array([5,24,25,33,36,41,43,45,46,48,58,198,225,237,306])

    # GA generated from legend pirate
    elif pv == 4:
        one_idx = numpy.array([6,15,32,44,55,64,122,127,150,165,224,249,256,288,297])
    # GA generated from trinity pirate
    elif pv == 5:
        one_idx = numpy.array([15,59,64,122,127,150,165,198,220,224,229,249,256,288,297])
    # GA generated from taunt warrior
    elif pv == 6:
        one_idx = numpy.array([8,35,39,41,42,58,98,125,165,199,229,260,267,279,301])
    # GA generated c'Thun control
    elif pv == 7:
        one_idx = numpy.array([15,32,40,59,64,122,127,150,165,198,220,224,249,288,297])

    # GA generated against pv 4
    elif pv == 8:
        one_idx = numpy.array([32,45,55,64,122,127,150,198,220,224,249,256,270,288,297])
    # GA generated against pv 5
    elif pv == 9:
        one_idx = numpy.array([15,59,64,71,77,122,127,150,151,198,224,249,256,288,297])
    # GA generated against pv 6
    elif pv == 10:
        one_idx = numpy.array([71,77,79,106,108,130,140,151,199,261,268,271,277,294,304])
    # GA generated against pv 7
    elif pv == 11:
        one_idx = numpy.array([15,25,32,44,59,64,122,127,150,198,224,249,256,288,297])

    elif pv == 12:
        15, 32, 40, 55, 64, 122, 146, 150, 198, 220, 224, 249, 256, 288, 297
    elif pv == 13:
        15, 32, 43, 59, 122, 150, 161, 165, 198, 224, 249, 256, 288, 294, 297
    elif pv == 14:
        15, 32, 51, 59, 64, 122, 127, 150, 165, 198, 220, 224, 256, 288, 297
    elif pv == 15:
        15, 37, 59, 64, 84, 122, 127, 150, 165, 198, 220, 224, 249, 256, 297

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





