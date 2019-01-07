import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import *

# write your script here, we recommend the above libraries for making your animation
frames = np.load('../data/carseq.npy')
rect = [59, 116, 145, 151]
width = rect[3] - rect[1]
length = rect[2] - rect[0]
rectList = []
for i in range(frames.shape[2]-1):
#    plt.imshow(frames[:,:,i],cmap='gray')
#    plt.pause(0.001)
    a = rect.copy()
    rectList.append(a)
    It = frames[:,:,i]
    It1 = frames[:,:,i+1]
    p = LucasKanade(It, It1, rect)
    rect[0] += p[0]
    rect[1] += p[1]
    rect[2] += p[0]
    rect[3] += p[1]
    num = i + 1
    if num % 100 == 0 or num == 1:
        plt.figure()
        plt.imshow(frames[:,:,i],cmap='gray')
        bbox = patches.Rectangle((int(rect[0]), int(rect[1])), length, width,
                                 fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(bbox)
        plt.title('frame %d'%num)
        plt.show()
np.save('carseqrects.npy',rectList)