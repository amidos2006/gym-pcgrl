import time
import numpy as np
import gym_pcgrl
from gym_pcgrl.envs.helper import gen_random_map, gen_random_map_slice

random = np.random
width = 14
height = 14
prob = {'0':0.3, '2':0.7}
n_maps = 10000
s = time.time()
for i in range(n_maps):
    gen_random_map_legacy(random, width, height, prob)
e = time.time()
print('gen_random_map avg: {}'.format((e - s)/n_maps))

s = time.time()
for i in range(n_maps):
    gen_random_map(random, width, height, prob)
e = time.time()
print('gen_random_map_slice avg: {}'.format((e - s)/n_maps))
