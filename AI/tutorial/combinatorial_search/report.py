"""
Report test results in test_probs
"""
import glob
from collections import defaultdict
import numpy
import optparse

duration_dict = defaultdict(list)
opt_val_dict = defaultdict(list)
function_call_dict = defaultdict(list)
file_count = 0

env = "prob_env_nn_noisy_pv"

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
    for field in fields:
        # method + wall_time_limit
        method_key = '{:<7s}'.format(field[0]) + '{:>7s}'.format(field[1])
        opt_val = field[5]
        duration = field[2]
        function_calls = field[4]
        opt_val_dict[method_key].append(float(opt_val))
        duration_dict[method_key].append(float(duration))
        function_call_dict[method_key].append(int(function_calls))

print(opt_val_dict)
print(duration_dict)
print(function_call_dict)
print("")

print("Average over {} files".format(file_count))
print('{:15s}'.format('method'),
      '    {:7s}'.format('opt_value'),
      ' {:7s}'.format('func calls'),
      '   {:7s}'.format('duration'))

if kwargs.order == 'name':
    method_key_sorted = sorted(opt_val_dict.keys())
elif kwargs.order == 'opt_val':
    method_key_sorted = sorted(opt_val_dict.keys(), key=lambda x: opt_val_dict[x])
elif kwargs.order == 'fc':
    method_key_sorted = sorted(function_call_dict.keys(), key=lambda x: function_call_dict[x])

for key in method_key_sorted:
    print('{:>15s}:'.format(key),
          '   {:.6f}:'.format(numpy.mean(opt_val_dict[key])),
          '   {:>7.0f}:'.format(numpy.mean(function_call_dict[key])),
          '   {:.6f}'.format(numpy.mean(duration_dict[key])))



