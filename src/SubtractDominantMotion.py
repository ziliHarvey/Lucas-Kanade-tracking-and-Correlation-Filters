import numpy as np
from LucasKanadeAffine import *
from scipy.ndimage import affine_transform
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import cv2
import matplotlib.pyplot as plt
from InverseCompositionAffine import *

def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
    # put your implementation here
    
    mask = np.ones(image1.shape, dtype=bool)
    
    M = LucasKanadeAffine(image1, image2)
    image2_w = cv2.warpAffine(image2, M, image1.T.shape)
    
#    UNCOMMENT THE FOLLOWING IF YOU WANT TO USE INVERSE COMPOSITION
#    M = InverseCompositionAffine(image1, image2)
#    image2_w = cv2.warpAffine(image2, M[:2, :], image1.T.shape)
    image2_w = binary_erosion(image2_w)
    image2_w = binary_dilation(image2_w)
    diff = np.abs(image1 - image2_w)
    threshold = 0.75
    mask = (diff > threshold)
    
    
    
    
    return mask
