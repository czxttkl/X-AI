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
