from RL_brain import Memory
import numpy
numpy.random.seed()

memory = Memory(capacity=30)
for i in range(20):
    memory.store((i, i))

# just check the sampling when all data are new. this should act like a uniform sampling
print('check sampling when all data are new.')
for i in range(3):
    tree_idx, batch_memory, ISWeights = memory.sample(5)
    print('tree_idx', tree_idx)
    print('data', batch_memory)
    print('is weights', ISWeights)
print()
print()

print('check update sample priority')
print('total priority before update', memory.tree.total_p)
tree_idx, batch_memory, ISWeights = memory.sample(5)
# abs_errors = numpy.array([float(bm[0]) for bm in batch_memory])
abs_errors = numpy.array([100. for bm in batch_memory])
print('tree_idx', tree_idx)
print('data', batch_memory)
print('is weights', ISWeights)
print('abs error', abs_errors)
memory.batch_update(tree_idx, abs_errors)
print('total priority after update', memory.tree.total_p)
print()
print()

# what you will see is that:
# (1) for data points not with high abs_errors, their importance weights are high. But they are sampled less often.
# (2) for data points with high abs_errors, their importance weights are low. But they are sampled more often.
print('check sampling after update')
for i in range(3):
    tree_idx, batch_memory, ISWeights = memory.sample(5)
    print('tree_idx', tree_idx)
    print('data', batch_memory)
    print('is weights', ISWeights)

print('total priority', memory.tree.total_p)
print()