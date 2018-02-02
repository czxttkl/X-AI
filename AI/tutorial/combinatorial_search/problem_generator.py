import optparse
import os
import numpy


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


