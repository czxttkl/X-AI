from collections import deque
import random
import numpy


class Memory:
    """
    Memory used for experience replay
    """
    def __init__(self, neg_reward_size, pos_reward_size, neu_reward_size,
                 neg_batch_size, pos_batch_size, neu_batch_size,
                 k, gamma, model, lag_model=None):
        self.neg_reward_size = neg_reward_size
        self.pos_reward_size = pos_reward_size
        self.neu_reward_size = neu_reward_size

        self.neg_batch_size = neg_batch_size
        self.pos_batch_size = pos_batch_size
        self.neu_batch_size = neu_batch_size

        self.neg_memory = deque(maxlen=self.neg_reward_size)
        self.pos_memory = deque(maxlen=self.pos_reward_size)
        self.neu_memory = deque(maxlen=self.neu_reward_size)

        self.k = k
        self.gamma = gamma
        self.model = model
        self.lag_model = lag_model

    def full(self):
        """ check if the memory is full """
        if len(self.neg_memory) == self.neg_reward_size and len(self.pos_memory) == self.pos_reward_size:
            return True
        return False

    def __repr__(self):
        return "memory pos: {0}, neg: {1}, neural: {2}".format(
            len(self.pos_memory), len(self.neg_memory), len(self.neu_memory))

    def sample_minibatch(self, mem, size):
        mini_batch = random.sample(mem, size)
        features = numpy.zeros((size, self.k))
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
                # max_a_idx = numpy.argmax(self.model.predict(next_features_over_acts))
                # max_q_s_a = self.lag_model.predict(next_features_over_acts)[max_a_idx]
                # normal DQN
                max_a_idx = numpy.argmax(self.lag_model.predict(next_features_over_acts))
                max_q_s_a = self.lag_model.predict(next_features_over_acts)[max_a_idx]
                target[i] = r + self.gamma * max_q_s_a

        return features, target

    def sample(self):
        features_pos, target_pos = self.sample_minibatch(self.pos_memory, self.pos_batch_size)
        features_neg, target_neg = self.sample_minibatch(self.neg_memory, self.neg_batch_size)
        features_neu, target_neu = self.sample_minibatch(self.neu_memory, self.neu_batch_size)
        features = numpy.vstack((features_pos, features_neg, features_neu))
        target = numpy.concatenate((target_pos, target_neg, target_neu))
        idx = list(range(len(target)))
        random.shuffle(idx)
        return features[idx], target[idx]

    def append(self, features, last_act, reward, next_features_over_acts, match_end):
        if reward < 0:
            self.neg_memory.append((features, last_act, reward, next_features_over_acts, match_end))
        elif reward > 0:
            self.pos_memory.append((features, last_act, reward, next_features_over_acts, match_end))
        else:
            self.neu_memory.append((features, last_act, reward, next_features_over_acts, match_end))

        # if reward != 0:
        #     self.memory.append((features, last_act, reward, next_features_over_acts, match_end))
        # else:
        #     # gradually accept zero reward intermediate state
        #     thres = 60. / numpy.log(self.num_match + 2)
        #     seed = numpy.random.random()
        #     if seed > thres:
        #         self.memory.append((features, last_act, reward, next_features_over_acts, match_end))
