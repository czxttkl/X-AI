"""
Very similar to env_nn.Environment.
However, it is added with output delay and noise
such that the output behavior is closer to deck evaluation
"""
import environment.env_nn as env_nn
import numpy
import time


class Environment(env_nn.Environment):

    def monte_carlo(self,
                    MONTE_CARLO_ITERATIONS=20000,
                    WALL_TIME_LIMIT=9e30,
                    noise_var=0):
        """
        Use monte carlo to find the max value

        MONTE_CARLO_ITERATIONS: maximum number of trials
        WALL_TIME_LIMIT: maximum wall time
        noise_var: noise variance to add on the output (before distill).
        """
        min_val = 9e16
        max_val = -9e16

        start_time = time.time()
        last_print_time = 0

        for i in range(MONTE_CARLO_ITERATIONS):
            x_o = self.cur_state[:self.k]

            random_xp = numpy.zeros(self.k + 1)  # state + step
            # the last component (step) will always be zero as a placeholder
            one_idx = numpy.random.choice(self.k, self.d, replace=False)
            random_xp[one_idx] = 1

            random_state = numpy.hstack((x_o, random_xp))
            random_state_output = self.output(random_state, delay=0, noise_var=noise_var)

            if random_state_output < min_val:
                min_val = random_state_output
                min_state = random_state
            if random_state_output > max_val:
                max_val = random_state_output
                max_state = random_state

            duration = time.time() - start_time
            if duration > WALL_TIME_LIMIT:
                break
            if duration - last_print_time > 15:
                print("monte carlo duration: {}, max: {}".format(duration, self.still(max_val)))
                last_print_time = duration

        return self.still(max_val), max_state, self.still(min_val), min_state, duration

    def output(self, state, delay=0.0, noise_var=0.07):
        """ output with delay and noise"""
        # WHY delay=0 BY DEFAULT?
        # if we allow RL to pretrain for one day,
        # and allow it to train 10K rounds,
        # then each function evaluation should finish within:
        # 3600*24/10000/30 = 0.288
        # however, to quickly perform experiments, we set delay to zero
        # WHY noise_var=0.07 BY DEFAULT?
        # based on preliminary tests, random plays have ~7% variance
        assert len(state.shape) == 1 and state.shape[0] == 2 * self.k + 1
        time.sleep(delay)
        out = numpy.dot(
            self.relu(
                numpy.dot(state[:-1], self.w1)
                + self.b1
            ),
            self.w2) + self.b2
        noise = numpy.random.normal(0, noise_var)
        out += noise
        out = self.sigmoid(out[0])
        out = self.distill(out)
        return out

    def outputs(self, states):
        raise NotImplementedError

