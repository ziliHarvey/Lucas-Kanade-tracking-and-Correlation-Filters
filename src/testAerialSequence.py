import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from SubtractDominantMotion import *
from LucasKanadeAffine import *

# write your script here, we recommend the above libraries for making your animation
frames = np.load('../data/aerialseq.npy')
for i in range(frames.shape[2]-1):
    image1 = frames[:, :, i]
    image2 = frames[:, :, i+1]
    mask = SubtractDominantMotion(image1, image2)
    if i == 31 or i == 61 or i == 91 or i == 121:
        
        plt.figure()
        for r in range(mask.shape[0]):
            for c in range(mask.shape[1]):
                if mask[r,c] == True:
                    plt.scatter(c, r, s = 10, c = 'b')
                    plt.show()
        plt.imshow(image1, cmap='gray')
       
        
 



