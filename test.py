from itertools import product
import numpy as np

weight = [0, 20, 200, 500, 1000]
action_total = list(product(weight, repeat=4))

for i in range(len(action_total)):
    if action_total[i].count(weight[0])>=3 or action_total[i].count(weight[1])>=3 or action_total[i].count(weight[2])>=3 or action_total[i].count(weight[3])>=3 or action_total[i].count(weight[4])>=3 :
        action_total[i] = []

action_total_tmp = [x for x in action_total if x]
action_total = action_total_tmp

for i in range(len(action_total)):
    action_total[i] = np.array(action_total[i])
    if i <= 256:
        print(action_total[i])
action_total = np.array(action_total)
print(action_total.shape[0])