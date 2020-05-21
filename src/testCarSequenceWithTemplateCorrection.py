import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from scipy.interpolate import RectBivariateSpline
from LucasKanade import *
from TemplateCorrection import *
import time

frames = np.load('../data/carseq.npy')
frames0 = frames[:,:,0]
rectList = np.load('./carseqrects.npy')


def copyRect(rect):
    rect_new = []
    for ele in rect:
        rect_new += [ele]
    return rect_new
  
rect = [59, 116, 145, 151]
width = rect[3] - rect[1]
length = rect[2] - rect[0]
rect0 = [59, 116, 145, 151] 
rectList_new = [copyRect(rect0)] # objects coordinates after correction
seq_len = frames.shape[2]
time_total = 0

# since template driftingb uses only the first ever frame
# lots of things can be pre-computed here
start0 = time.time()
rows_img, cols_img = frames0.shape
x1, y1, x2, y2 = rect0[0], rect0[1], rect0[2], rect0[3]
rows_rect, cols_rect = x2 - x1, y2 - y1
y = np.arange(0, rows_img, 1)
x = np.arange(0, cols_img, 1)
c = np.linspace(x1, x2, cols_rect)
r = np.linspace(y1, y2, rows_rect)
cc, rr = np.meshgrid(c, r)
spline = RectBivariateSpline(y, x, frames0)
T = spline.ev(rr, cc)
end0 = time.time()
time_total += end0 - start0

for i in range(seq_len):
    if i == 0:
        continue
    print("Processing frame %d" % i)

    start1 = time.time()
    It = frames[:,:,i-1]
    It1 = frames[:,:,i]
    p = LucasKanade(It, It1, rect)
    rect[0] += p[0]
    rect[1] += p[1]
    rect[2] += p[0]
    rect[3] += p[1]
    end1 = time.time()
    time_total += end1 - start1
    
    # drift correction
    start2 = time.time()
    TemplateCorrection(T, It1, rect)
    end2 = time.time()
    time_total += end2 - start2

    rectList_new.append(copyRect(rect))
    if i % 100 == 0 or i == 1:
        plt.figure()
        plt.imshow(frames[:,:,i],cmap='gray')
        bbox0 = patches.Rectangle((int(rectList[i-1][0]), int(rectList[i-1][1])), length, width,
                                 fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(bbox0)
        bbox1 = patches.Rectangle((int(rectList_new[i-1][0]), int(rectList_new[i-1][1])), length, width,
                                 fill=False, edgecolor='blue', linewidth=2)
        plt.gca().add_patch(bbox1)
        plt.title('frame %d' % i)
        plt.show()

np.save('carseqrects-wcrt.npy',rectList_new)
print('Finished, the tracking frequency is %.4f' % (seq_len / time_total))
