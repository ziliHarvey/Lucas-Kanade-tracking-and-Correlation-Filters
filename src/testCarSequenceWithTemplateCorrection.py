import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from scipy.interpolate import RectBivariateSpline
from LucasKanade import *
from TemplateCorrection import *
# write your script here, we recommend the above libraries for making your animation

frames = np.load('../data/carseq.npy')
frames0 = frames[:,:,0]
rectList0 = np.load('./carseqrects.npy')
rect = [59, 116, 145, 151]
rect0 = [59, 116, 145, 151]
width = rect[3] - rect[1]
length = rect[2] - rect[0]
rectList = []
rectList_new = []
  
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
    #drift correction
    p_star = TemplateCorrection(frames0, It1, rect0, rect)
    rect[0] += p_star[0]
    rect[1] += p_star[1]
    rect[2] += p_star[0]
    rect[3] += p_star[1]
    b = rect.copy()
    rectList_new.append(b)
    num = i + 1
    if num % 100 == 0 or num == 1:
        plt.figure()
        plt.imshow(frames[:,:,i],cmap='gray')
        bbox0 = patches.Rectangle((int(rectList0[i,0]), int(rectList0[i,1])), length, width,
                                 fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(bbox0)
        plt.show()
        bbox1 = patches.Rectangle((int(rect[0]), int(rect[1])), length, width,
                                 fill=False, edgecolor='blue', linewidth=2)
        plt.gca().add_patch(bbox1)
        plt.title('frame %d'%num)

np.save('carseqrects-wcrt.npy',rectList_new)




























