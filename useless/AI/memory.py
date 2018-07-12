from collections import deque
import random
import numpy
import pickle
import constant


class MonteCarloMemory:
    """
    Memory used for experience replay
    """
    def __init__(self, qvalues_impl):
        self.neg_memory = deque(maxlen=constant.ql_dqn_mem_neg_size)
        self.pos_memory = deque(maxlen=constant.ql_dqn_mem_pos_size)
        self.qvalues_impl = qvalues_impl
        self.buffer = []

    def start_train(self):
        """ check if the memory is full """
        if len(self.neg_memory) >= constant.ql_dqn_memory_start_train_size and \
                len(self.pos_memory) >= constant.ql_dqn_memory_start_train_size:
            return True
        return False

    def __repr__(self):
        return "memory pos: {0}, neg: {1}".format(
            len(self.pos_memory), len(self.neg_memory))

    def sample_minibatch(self, mem, size):
        mini_batch = random.sample(mem, size)
        features = numpy.zeros((size, self.qvalues_impl.k))
        target = numpy.zeros(size)

        for i in range(size):
            features[i] = mini_batch[i][0]
            r = mini_batch[i][1]
            next_features_over_acts = mini_batch[i][2]
            match_end = mini_batch[i][3]
            if match_end:
                target[i] = r
            else:
                # Double DQN
                # action selection is from model, target q(s', a') is from lag_model
                max_a_idx = numpy.argmax(self.qvalues_impl.model.predict(next_features_over_acts))
                max_q_s_a = self.qvalues_impl.lag_model.predict(next_features_over_acts)[max_a_idx]
                # normal DQN
                # max_a_idx = numpy.argmax(self.qvalues_impl.lag_model.predict(next_features_over_acts))
                # max_q_s_a = self.qvalues_impl.lag_model.predict(next_features_over_acts)[max_a_idx]
                target[i] = r + self.qvalues_impl.gamma * max_q_s_a

        return features, target

    def sample(self):
        features_pos, target_pos = self.sample_minibatch(self.pos_memory, constant.ql_dqn_pos_batch_size)
        features_neg, target_neg = self.sample_minibatch(self.neg_memory, constant.ql_dqn_neg_batch_size)
        features = numpy.vstack((features_pos, features_neg))
        target = numpy.concatenate((target_pos, target_neg))
        idx = list(range(len(target)))
        random.shuffle(idx)
        return features[idx], target[idx]

    def save(self):
        with open(self.qvalues_impl.file_name_memory(), 'wb') as f:
            pickle.dump((self.pos_memory, self.neg_memory), f, protocol=4)

    def load(self):
        with open(self.qvalues_impl.file_name_memory(), 'rb') as f:
            self.pos_memory, self.neg_memory = pickle.load(f)

    def append(self, features, reward, next_features_over_acts, match_end):
        self.buffer.append([features, reward, next_features_over_acts, match_end])
        if match_end:
            len_buffer = len(self.buffer)
            for idx, b in enumerate(self.buffer):
                discount = self.qvalues_impl.gamma ** (len_buffer - (idx + 1))
                b[1] = discount * reward
                b[3] = True   # make match_end=True so that only reward is used
                              # to approximate Q(s,a) (No max a' Q(s',a'))
                if reward < 0:
                    self.neg_memory.append(b)
                else:
                    self.pos_memory.append(b)
            self.buffer = []

