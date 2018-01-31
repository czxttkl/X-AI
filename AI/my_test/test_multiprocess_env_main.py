"""
This script tests whether environment.output can be calculated with multiprocessing
"""
from my_test.test_multiprocess_env import Environment
import numpy


if __name__ == "__main__":
    # by the way, test variance of random play
    win_rates = []
    for i in range(30):
        env = Environment()
        player1_win_rate = env.output(match_num=8000)
        win_rates.append(player1_win_rate)

    print("win rate trials:", len(win_rates))
    print("win rate mean:", numpy.mean(win_rates))
    print("win rate std:", numpy.std(win_rates))