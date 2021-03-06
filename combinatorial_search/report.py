"""
Report test results in test_probs
"""
import glob
from collections import defaultdict
import numpy
import optparse
import pprint
from scipy import stats
numpy.set_printoptions(linewidth=10000, precision=3)


opt_val_dict = defaultdict(list)
duration_dict = defaultdict(list)
function_call_dict = defaultdict(list)
generation_dict = defaultdict(list)
file_count = 0

env = "env_greedymove"
# env = "env_nn_noisy"

parser = optparse.OptionParser(usage="usage: %prog [options]")
parser.add_option("--env", dest="env",
                  type="string", default=env)
# you can order by:
# 1. name: method name
# 2. opt_val: opt value
# 3. fc: function calls
parser.add_option("--order", dest="order",
                  type="string", default="name")

(kwargs, args) = parser.parse_args()

for filename in glob.iglob('test_probs/prob_{}_pv*/test_result.csv'.format(kwargs.env), recursive=True):
    file_count += 1
    with open(filename, 'r') as f:
        lines = f.readlines()[1:]
    fields = map(lambda line: line.split(','), lines)

    opt_val_pv_dict = defaultdict(list)
    duration_pv_dict = defaultdict(list)
    function_call_pv_dict = defaultdict(list)
    generation_pv_dict = defaultdict(list)

    for field in fields:
        method = field[0]
        wall_time_limit = field[1]
        generation = field[3]
        opt_val = field[5]
        duration = field[2]
        function_calls = field[4]

        if method == 'rl_prtr':
            # method + wall_time_limit + generation
            method_key = '{:<20s}'.format(method + '{:>6s}'.format(wall_time_limit) + '{:>5s}'.format(generation))
        elif method == 'sl':
            # method + generation (# of training data) + function calls (# of samples)
            method_key = '{:<20s}'.format(method + '{:>10s}'.format(generation) + '{:>10s}'.format(function_calls))
        else:
            # method + wall_time_limit
            method_key = '{:<10s}'.format(method) + '{:<10s}'.format(wall_time_limit)

        opt_val_pv_dict[method_key].append(float(opt_val))
        duration_pv_dict[method_key].append(float(duration))
        function_call_pv_dict[method_key].append(int(function_calls))
        generation_pv_dict[method_key].append(int(generation))

    for method_key in opt_val_pv_dict.keys():
        opt_val_dict[method_key].append(numpy.median(opt_val_pv_dict[method_key]))
        duration_dict[method_key].append(numpy.median(duration_pv_dict[method_key]))
        function_call_dict[method_key].append(numpy.median(function_call_pv_dict[method_key]))
        generation_dict[method_key].append(numpy.median(generation_pv_dict[method_key]))

print('opt_val_dict:')
pprint.pprint(opt_val_dict)
print('duration dict:')
pprint.pprint(duration_dict)
print("function call dict:")
pprint.pprint(function_call_dict)
print("")

print("Average over {} files".format(file_count))

# Paired T-Test
# reference:
# 1. https://stackoverflow.com/questions/14176280/test-for-statistically-significant-difference-between-two-arrays
# 2. https://stats.stackexchange.com/questions/320469/how-to-test-if-xi-is-significantly-greater-than-yi
print("two-sided paired t-tests")
for key1 in opt_val_dict.keys():
    for key2 in opt_val_dict.keys():
        if key1 == key2:
            continue
        diffs = numpy.array(opt_val_dict[key1]) - numpy.array(opt_val_dict[key2])
        zeros = numpy.zeros(len(diffs))
        tval, pval = stats.ttest_rel(diffs, zeros)
        print('{} vs. {}, p-value: {:.5f}, t-vale: {:3f}, diffs: {:.3f}, {}'
              .format(key1, key2, pval, tval, numpy.mean(diffs), diffs))
        rank1 = numpy.greater(numpy.array(opt_val_dict[key1]), numpy.array(opt_val_dict[key2])).astype(int)
        rank2 = numpy.greater(numpy.array(opt_val_dict[key2]), numpy.array(opt_val_dict[key1])).astype(int)
        # rank_diffs = rank1 - rank2
        # tval, pval = stats.ttest_rel(rank1, rank2)
        # # one side t test
        # pval = pval / 2
        # print('{} vs. {}, p-value: {:.5f}, t: {:.3f}, diffs: {:.3f}, {}'.
        #       format(key1, key2, pval, tval, numpy.mean(rank_diffs), rank_diffs))

print()

if kwargs.order == 'name':
    method_key_sorted = sorted(opt_val_dict.keys())
elif kwargs.order == 'opt_val':
    method_key_sorted = sorted(opt_val_dict.keys(), key=lambda x: opt_val_dict[x])
elif kwargs.order == 'fc':
    method_key_sorted = sorted(function_call_dict.keys(), key=lambda x: function_call_dict[x])

print('{:20s}'.format('method'),
      '    {:7s}'.format('opt_value'),
      ' {:7s}'.format('func calls'),
      '   {:7s}'.format('duration'),
      '   {:7s}'.format('generation'))
for key in method_key_sorted:
    print('{:>20s}:'.format(key),
          '   {:.6f}:'.format(numpy.mean(opt_val_dict[key])),
          '   {:>7.0f}:'.format(numpy.mean(function_call_dict[key])),
          '   {:.6f}'.format(numpy.mean(duration_dict[key])),
          '   {:.1f}'.format(numpy.mean(generation_dict[key])))




