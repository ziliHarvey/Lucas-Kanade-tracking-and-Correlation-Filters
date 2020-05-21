import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from SubtractDominantMotion import *
from LucasKanadeAffine import *
import time

frames = np.load('../data/aerialseq.npy')
time_total = 0
seq_len = frames.shape[2]

for i in range(seq_len):
    if i == 0:
        continue
    print("Processing frame %d" % i)

    start = time.time()
    image1 = frames[:, :, i-1]
    image2 = frames[:, :, i]
    mask = SubtractDominantMotion(image1, image2)
    end = time.time()
    time_total += end - start

    if i == 31 or i == 61 or i == 91 or i == 121:    
        plt.figure()
        plt.imshow(image1, cmap='gray')
        for r in range(mask.shape[0]-1):
            for c in range(mask.shape[1]-1):
                if mask[r,c]:
                    plt.scatter(c, r, s = 1, c = 'r', alpha=0.5)
        plt.show()

print('Finished, the tracking frequency is %.4f' % (seq_len / time_total))
