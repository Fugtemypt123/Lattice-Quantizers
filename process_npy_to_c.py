import numpy as np


with open('best.npy.txt', 'w') as f:
    x = np.load('best.npy')
    for i in range(32):
        f.write(' '.join([f"{y:12.8f}" for y in x[i]]) + '\n')
