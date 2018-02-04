"""
Report test results in test_probs
"""
import glob
from collections import defaultdict
import numpy

duration_dict = defaultdict(list)
opt_val_dict = defaultdict(list)
file_count = 0

prob_prefix = "prob_env_nn_noisy_pv"
for filename in glob.iglob('test_probs/{}*/test_result.csv'.format(prob_prefix), recursive=True):
    file_count += 1
    with open(filename, 'r') as f:
        lines = f.readlines()[1:]
    fields = map(lambda line: line.split(','), lines)
    for field in fields:
        # method + wall_time_limit
        method_key = field[0] + field[1]
        opt_val = field[4]
        duration = field[2]
        opt_val_dict[method_key].append(float(opt_val))
        duration_dict[method_key].append(float(duration))

print(opt_val_dict)
print(duration_dict)
print("")

print("Average over {} files".format(file_count))
method_key_sorted = sorted(list(key for key in opt_val_dict))
for key in method_key_sorted:
    print('{:>15s}:'.format(key), '{:.6f}'.format(numpy.mean(opt_val_dict[key])), ":", numpy.mean(duration_dict[key]))



