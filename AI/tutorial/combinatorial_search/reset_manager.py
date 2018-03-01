import heapq
from collections import deque
import numpy


class ResetManager:

    def __init__(self):
        self.xo_heap = []
        self.xp_queue = deque(maxlen=1000)
        self.mode = ''

    def reset(self, env_name, env, epsilon):
        # only apply to env_gamestate or env_greedymove
        # only when exploration has been for a while
        # only work for non fixed xo environment
        if env_name not in ['env_gamestate', 'env_greedymove'] or epsilon < 0.1 or env.if_set_fixed_xo():
            reset_state = env.reset()
            return reset_state

        # rules for env_gamestate:
        # suppose x is the lowest win rate against the most powerful x_o
        # 1. x/2% randomly reset
        # 2. x/2% return powerful xp
        # 3. (1-x)% return powerful xo
        r = numpy.random.rand()
        if self.xo_heap:
            x, y, reset_xo = self.xo_heap[0]
            if r >= x:
                print("sample reset pick xo_heap. queue size: {}, heap size: {}. least win rate: ({:.3f}, {}), r: {:.3f}"
                      .format(len(self.xp_queue), len(self.xo_heap), x, y, r))
                reset_state = env.reset(xo=reset_xo)
                self.mode = 'pick_xo'
                return reset_state

        r = numpy.random.rand()
        if 0. <= r <= 0.5 and self.xp_queue:
            x, reset_xo = self.xp_queue[-1]
            print("sample reset pick xp_queue. queue size: {}, heap size: {}. xp win rate: {:.3f}, r: {:.3f}"
                  .format(len(self.xp_queue), len(self.xo_heap), x, r))
            reset_state = env.reset(xo=reset_xo)
            self.mode = 'pick_xp'
        else:
            print("sample reset pick random. queue size: {}, heap size: {}. r: {}"
                  .format(len(self.xp_queue), len(self.xo_heap), r))
            reset_state = env.reset()
            self.mode = 'pick_random'
        return reset_state

    def update(self, env_name, env, win_rate, epsilon, trial_size):
        # only apply to env_gamestate or env_greedymove
        # only when exploration has been for a while
        # only work for non fixed xo environment
        if env_name not in ['env_gamestate', 'env_greedymove'] or epsilon < 0.1 or env.if_set_fixed_xo():
            return

        # only update reset pool at the end of one episode
        assert env.cur_state[-1] == trial_size

        if self.mode == 'pick_xp':
            # pop last xp
            self.xp_queue.pop()

        # powerful x_p
        if win_rate > 0.7:
            self.xp_queue.append((win_rate, env.x_p))
            print("update queue. queue size: {}, heap size: {}".format(len(self.xp_queue), len(self.xo_heap)))

        # powerful x_o
        if self.mode == 'pick_xo':
            x, y, xo = heapq.heappop(self.xo_heap)
            print("update pick xo", x, y)
            x = (x * y + win_rate) / (y + 1)  # new average win rate
            y += 1
        else:
            x, y = win_rate, 1
        heapq.heappush(self.xo_heap, (x, y, env.x_o))
        print("update heap. current most powerful xo win rate: ({:.3f}, {}), queue size: {}, heap size: {}"
              .format(self.xo_heap[0][0], self.xo_heap[0][1], len(self.xp_queue), len(self.xo_heap)))

        if len(self.xo_heap) > 1000:
            self.xo_heap = heapq.nsmallest(500, self.xo_heap)
