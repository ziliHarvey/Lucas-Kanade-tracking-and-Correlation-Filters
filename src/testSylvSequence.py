import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanadeBasis import *
from LucasKanade import *
from TemplateCorrection import *
import time

def copyRect(rect):
    rect_new = []
    for ele in rect:
        rect_new += [ele]
    return rect_new

# write your script here, we recommend the above libraries for making your animation
bases = np.load('../data/sylvbases.npy')
frames = np.load('../data/sylvseq.npy')
seq_len = frames.shape[2]
frame0 = frames[:,:,0]
rect = [101, 61, 155, 107]
rect_baseline = [101, 61, 155, 107]
width = rect[3] - rect[1]
length = rect[2] - rect[0]
rectList = [copyRect(frame0)]
rectList_baseline = [copyRect(frame0)]
time_total = 0

# since template driftingb uses only the first ever frame
# lots of things can be pre-computed here
rows_img, cols_img = frame0.shape
x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
rows_rect, cols_rect = x2 - x1, y2 - y1
y = np.arange(0, rows_img, 1)
x = np.arange(0, cols_img, 1)
c = np.linspace(x1, x2, cols_rect)
r = np.linspace(y1, y2, rows_rect)
cc, rr = np.meshgrid(c, r)
spline = RectBivariateSpline(y, x, frame0)
T = spline.ev(rr, cc)

#Apply LucasKanadeWithTemplateCorrection Algorithm
for i in range(seq_len):
    if i == 0:
        continue
    It = frames[:,:,i-1]
    It1 = frames[:,:,i]
    p_baseline = LucasKanade(It, It1, rect_baseline)
    rect_baseline[0] += p_baseline[0]
    rect_baseline[1] += p_baseline[1]
    rect_baseline[2] += p_baseline[0]
    rect_baseline[3] += p_baseline[1]
    TemplateCorrection(T, It1, rect_baseline)
    rectList_baseline.append(copyRect(rect_baseline))

#Apply LucasKanadeBasis Algorithm
for i in range(seq_len):
    if i == 0:
        continue
    print("Processing frame %d" % i)

    start = time.time()
    It = frames[:,:,i-1]
    It1 = frames[:,:,i]
    p = LucasKanadeBasis(It, It1, rect, bases)
    rect[0] += p[0]
    rect[1] += p[1]
    rect[2] += p[0]
    rect[3] += p[1]
    end = time.time()
    time_total += end - start

    rectList.append(copyRect(rect))

    if i == 1 or i == 100 or i == 200 or i == 300 or i == 350 or i == 400:
        plt.figure()
        plt.imshow(frames[:,:,i],cmap='gray')
        bbox1 = patches.Rectangle((int(rectList[i][0]), int(rectList[i][1])), length, width,
                                 fill=False, edgecolor='blue', linewidth=2)
        plt.gca().add_patch(bbox1)
        bbox0 = patches.Rectangle((int(rectList_baseline[i][0]), int(rectList_baseline[i][1])), length, width,
                                 fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(bbox0)
        plt.title('frame %d' % i)
        plt.show()

np.save('Sylvseqrects.npy',rectList)
print('Finished, the tracking frequency is %.4f' % (seq_len / time_total))
