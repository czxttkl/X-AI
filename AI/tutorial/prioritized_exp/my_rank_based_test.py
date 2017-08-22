import rank_based
import numpy
from collections import Counter


def t():
    conf = {'size': 50,
            'learn_start': 10,
            'partition_num': 5,
            'total_step': 100,
            'batch_size': 4}
    experience = rank_based.Experience(conf)

    # insert to experience
    print('test insert experience')
    for i in range(1, 1151):
        # tuple, like(state_t, a, r, state_t_1, t)
        to_insert = (i, 1, 1, i, 1)
        experience.store(to_insert)
    print(experience.priority_queue)
    print(experience._experience[1])
    print(experience._experience[2])
    print('test replace')
    to_insert = (51, 1, 1, 51, 1)
    experience.store(to_insert)
    print(experience.priority_queue)
    print(experience._experience[1])
    print(experience._experience[2])

    # sample
    print('test sample')
    sample, w, e_id = experience.sample(51)
    print(sample)
    print(w)
    print(e_id)

    e_ids = Counter()
    for i in range(10000):
        sample, w, e_id = experience.sample(51)
        e_ids += Counter(e_id)
    print('before update delta', sorted(e_ids.items()))

    # update delta to priority
    print('test update delta')
    delta = [500]
    e_id = [34]
    experience.update_priority(e_id, delta)

    e_ids = Counter()
    for i in range(10000):
        sample, w, e_id = experience.sample(51)
        e_ids += Counter(e_id)
    print('after update delta', sorted(e_ids.items()))

    print(experience.priority_queue)
    sample, w, e_id = experience.sample(51)
    print(sample)
    print(w)
    print(e_id)

    # rebalance
    print('test rebalance')
    experience.rebalance()
    print(experience.priority_queue)


def main():
    t()


if __name__ == '__main__':
    main()