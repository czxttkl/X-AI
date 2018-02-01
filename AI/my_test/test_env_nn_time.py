from environment.env_nn import Environment
import  time


k=200
d=30
env = Environment(k=k, d=d)
state = env.cur_state
test_num = 100000

start_time = time.time()
for i in range(test_num):
    env.output(state)
duration = time.time() - start_time

print('duration:', duration)
print('function calls:', test_num)
print('average function call time:', duration / test_num)

# result 1
# duration: 8.00745177268982
# function calls: 100000
# average function call time: 8.00745177268982e-05

# result 2
# duration: 7.5600855350494385
# function calls: 100000
# average function call time: 7.560085535049438e-05