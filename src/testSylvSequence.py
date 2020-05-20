import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanadeBasis import *
from LucasKanade import *
from TemplateCorrection import *

# write your script here, we recommend the above libraries for making your animation
bases = np.load('../data/sylvbases.npy')
frames = np.load('../data/sylvseq.npy')
rect = [101, 61, 155, 107]
rect_baseline = [101, 61, 155, 107]
width = rect[3] - rect[1]
length = rect[2] - rect[0]
rectList = []
rectList_baseline = []

#Apply LucasKanadeWithTemplateCorrection Algorithm
for i in range(frames.shape[2]-1):
    a_baseline = rect_baseline.copy()
    rectList_baseline.append(a_baseline)
    It = frames[:,:,i]
    It1 = frames[:,:,i+1]
    p_baseline = LucasKanade(It, It1, rect_baseline)
    rect_baseline[0] += p_baseline[0]
    rect_baseline[1] += p_baseline[1]
    rect_baseline[2] += p_baseline[0]
    rect_baseline[3] += p_baseline[1]
    p_star = TemplateCorrection(frames[:,:,0], It1, rect, rect_baseline)
    rect_baseline[0] += p_star[0]
    rect_baseline[1] += p_star[1]
    rect_baseline[2] += p_star[0]
    rect_baseline[3] += p_star[1]

#Apply LucasKanadeBasis Algorithm
for i in range(frames.shape[2]-1):
#    plt.imshow(frames[:,:,i],cmap='gray')
#    plt.pause(0.001)
    a = rect.copy()
    rectList.append(a)
    It = frames[:,:,i]
    It1 = frames[:,:,i+1]
    p = LucasKanadeBasis(It, It1, rect, bases)
    rect[0] += p[0]
    rect[1] += p[1]
    rect[2] += p[0]
    rect[3] += p[1]
    num = i + 1
    if num == 1 or num == 100 or num == 200 or num == 300 or num == 350 or num == 400:
        plt.figure()
        plt.imshow(frames[:,:,i],cmap='gray')
        bbox1 = patches.Rectangle((int(rect[0]), int(rect[1])), length, width,
                                 fill=False, edgecolor='blue', linewidth=2)
        plt.gca().add_patch(bbox1)
        bbox0 = patches.Rectangle((int(rectList_baseline[i][0]), int(rectList_baseline[i][1])), length, width,
                                 fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(bbox0)
        plt.title('frame %d'%num)
        plt.show()
np.save('Sylvseqrects.npy',rectList)