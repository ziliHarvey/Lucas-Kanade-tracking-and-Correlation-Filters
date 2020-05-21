import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

def LucasKanadeAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
    
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    threshold = 5
    dp = [[1], [0], [0], [0], [1], [0]]
    p = np.zeros(6)
    x1, y1, x2, y2 = 0, 0, It.shape[1], It.shape[0]
    rows, cols = It.shape

    # pre-computed
    Iy, Ix = np.gradient(It1)
    y = np.arange(0, rows, 1)
    x = np.arange(0, cols, 1)     
    c = np.linspace(x1, x2, cols)
    r = np.linspace(y1, y2, rows)
    cc, rr = np.meshgrid(c, r)
    spline = RectBivariateSpline(y, x, It)
    T = spline.ev(rr, cc)
    spline_gx = RectBivariateSpline(y, x, Ix)
    spline_gy = RectBivariateSpline(y, x, Iy)
    spline1 = RectBivariateSpline(y, x, It1)

    while np.square(dp).sum() > threshold:

        W = np.array([[1.0 + p[0], p[1], p[2]],
                       [p[3], 1.0 + p[4], p[5]]])
    
        x1_w = W[0,0] * x1 + W[0,1] * y1 + W[0,2]
        y1_w = W[1,0] * x1 + W[1,1] * y1 + W[1,2]
        x2_w = W[0,0] * x2 + W[0,1] * y2 + W[0,2]
        y2_w = W[1,0] * x2 + W[1,1] * y2 + W[1,2]
    
        cw = np.linspace(x1_w, x2_w, It.shape[1])
        rw = np.linspace(y1_w, y2_w, It.shape[0])
        ccw, rrw = np.meshgrid(cw, rw)
        
        warpImg = spline1.ev(rrw, ccw)

        #compute error image
        #errImg is (n,1)
        err = T - warpImg
        errImg = err.reshape(-1,1)
        
        #compute gradient
        Ix_w = spline_gx.ev(rrw, ccw)
        Iy_w = spline_gy.ev(rrw, ccw)
        #I is (n,2)
        I = np.vstack((Ix_w.ravel(),Iy_w.ravel())).T
        
        #evaluate delta = I @ jac is (n, 6)
        delta = np.zeros((It.shape[0]*It.shape[1], 6))
   
        for i in range(It.shape[0]):
            for j in range(It.shape[1]):
                #I is (1,2) for each pixel
                #Jacobiani is (2,6)for each pixel
                I_indiv = np.array([I[i*It.shape[1]+j]]).reshape(1,2)
                
                jac_indiv = np.array([[j, 0, i, 0, 1, 0],
                                      [0, j, 0, i, 0, 1]]) 
                delta[i*It.shape[1]+j] = I_indiv @ jac_indiv
        
        #compute Hessian Matrix
        #H is (6,6)
        H = delta.T @ delta
        
        #compute dp
        #dp is (6,6)@(6,n)@(n,1) = (6,1)
        dp = np.linalg.inv(H) @ (delta.T) @ errImg
        
        #update parameters
        p[0] += dp[0,0]
        p[1] += dp[1,0]
        p[2] += dp[2,0]
        p[3] += dp[3,0]
        p[4] += dp[4,0]
        p[5] += dp[5,0]

    M =  np.array([[1.0 + p[0], p[1], p[2]], [p[3], 1.0 + p[4], p[5]]])   
    return M




