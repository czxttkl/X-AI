from collections import deque
import random
import numpy
import pickle


class Memory:
    """
    Memory used for experience replay
    """
    def __init__(self, neg_reward_size, pos_reward_size, neg_batch_size, pos_batch_size, qvalues_impl):
        self.neg_reward_size = neg_reward_size
        self.pos_reward_size = pos_reward_size
        self.neg_batch_size = neg_batch_size
        self.pos_batch_size = pos_batch_size
        self.neg_memory = deque(maxlen=self.neg_reward_size)
        self.pos_memory = deque(maxlen=self.pos_reward_size)
        self.qvalues_impl = qvalues_impl
        self.buffer = []

    def full(self):
        """ check if the memory is full """
        if len(self.neg_memory) == self.neg_reward_size and len(self.pos_memory) == self.pos_reward_size:
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
            r = mini_batch[i][2]
            next_features_over_acts = mini_batch[i][3]
            match_end = mini_batch[i][4]
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
        features_pos, target_pos = self.sample_minibatch(self.pos_memory, self.pos_batch_size)
        features_neg, target_neg = self.sample_minibatch(self.neg_memory, self.neg_batch_size)
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

    def append(self, features, last_act, reward, next_features_over_acts, match_end):
        self.buffer.append([features, last_act, reward, next_features_over_acts, match_end])
        if match_end:
            for tuple in self.buffer:
                tuple[2] = reward
                tuple[4] = True   # make match_end=True so that only reward is used
                                  # to approximate Q(s,a) (No max a' Q(s',a'))
                if reward < 0:
                    self.neg_memory.append(tuple)
                else:
                    self.pos_memory.append(tuple)
            self.buffer = []

        # if reward < 0:
        #     self.neg_memory.append((features, last_act, reward, next_features_over_acts, match_end))
        # elif reward > 0:
        #     self.pos_memory.append((features, last_act, reward, next_features_over_acts, match_end))
        # else:
        #     self.neu_memory.append((features, last_act, reward, next_features_over_acts, match_end))

        # if reward != 0:
        #     self.memory.append((features, last_act, reward, next_features_over_acts, match_end))
        # else:
        #     # gradually accept zero reward intermediate state
        #     thres = 60. / numpy.log(self.num_match + 2)
        #     seed = numpy.random.random()
        #     if seed > thres:
        #         self.memory.append((features, last_act, reward, next_features_over_acts, match_end))
