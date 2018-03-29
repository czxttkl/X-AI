import time
import numpy


class RandomSearch:

    def __init__(self, env):
        self.env = env

    def random_search(self, iteration_limit, wall_time_limit):
        min_val = 9e16
        max_val = -9e16

        start_time = time.time()
        last_print_time = 0

        for i in range(iteration_limit):
            x_o = self.env.cur_state[:self.env.k]

            random_xp = numpy.zeros(self.env.k + 1)  # state + step
            # the last component (step) will always be zero as a placeholder
            one_idx = numpy.random.choice(self.env.k, self.env.d, replace=False)
            random_xp[one_idx] = 1

            random_state = numpy.hstack((x_o, random_xp))
            random_state_output = self.env.output(random_state)

            if random_state_output < min_val:
                min_val = random_state_output
                min_state = random_state
            if random_state_output > max_val:
                max_val = random_state_output
                max_state = random_state

            duration = time.time() - start_time
            if duration > wall_time_limit:
                break
            if duration - last_print_time > 15:
                print("random search duration: {}, max: {}".format(duration, self.env.still(max_val)))
                last_print_time = duration

        return self.env.still(max_val), max_state, self.env.still(min_val), min_state, duration, i+1

