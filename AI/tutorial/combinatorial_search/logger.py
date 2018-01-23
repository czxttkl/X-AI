""" A self-defined logger to log data """

class Logger:
    def __init__(self, log_path):
        self.log_path = log_path

    def log_monte_carlo(self, max_val, max_state, min_val, min_state, duration):
        output_str = "monte carlo max: {0} at {1}\nmin: {2} at {3}\ntake time: {4} seconds\n".\
            format(max_val, max_state, min_val, min_state, duration)
        print(output_str)
        with open(self.log_path, 'a') as f:
            f.write(output_str)

    def log_wall_time(self, wall_time):
        output_str = 'program wall time:{}'.format(wall_time)
        print(output_str)
        with open(self.log_path, 'a') as f:
            f.write(output_str)

    def log_test(self, output_mc, state_mc, duration_mc,
                 output_rl, state_rl, duration_rl,
                 learn_step_counter, wall_time):
        output_str = 'learn step:' + str(learn_step_counter) + ', wall time:' + str(wall_time) + '\n'
        output_str += 'qlearning: {}, duration: {:.2f}, state: {}\n'.format(output_rl, duration_rl, state_rl)
        output_str += 'monte carlo: {}, duration: {:.2f}, state: {}\n\n'.format(output_mc, duration_mc, state_mc)
        with open(self.log_path, 'a') as f:
            f.write(output_str)
