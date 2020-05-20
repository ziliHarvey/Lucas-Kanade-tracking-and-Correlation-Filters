import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import *
import time

# write your script here, we recommend the above libraries for making your animation
frames = np.load('../data/carseq.npy')
rect = [59, 116, 145, 151]
width = rect[3] - rect[1]
length = rect[2] - rect[0]
rectList = []
time_total = 0
seq_len = frames.shape[2]

for i in range(seq_len):
    if (i == 0):
        continue
    print("Processing frame %d" % i)
    a = rect.copy()
    rectList.append(a)

    start = time.time()
    It = frames[:,:,i-1]
    It1 = frames[:,:,i]
    p = LucasKanade(It, It1, rect)
    rect[0] += p[0]
    rect[1] += p[1]
    rect[2] += p[0]
    rect[3] += p[1]
    end = time.time()
    time_total += end - start

    if i % 100 == 0 or i == 1:
        plt.figure()
        plt.imshow(frames[:,:,i],cmap='gray')
        bbox = patches.Rectangle((int(rect[0]), int(rect[1])), length, width,
                                 fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(bbox)
        plt.title('frame %d'%i)
        plt.show()
np.save('carseqrects.npy',rectList)
print('Finished, the tracking frequency is %.4f' % (seq_len / time_total))
