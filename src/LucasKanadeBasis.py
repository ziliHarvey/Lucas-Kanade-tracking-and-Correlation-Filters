import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeBasis(It, It1, rect, bases, p0 = np.zeros(2)):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	bases: [n, m, k] where nxm is the size of the template.
	# Output:
	#	p: movement vector [dp_x, dp_y]

    # Put your implementation here
    threshold = 0.05
    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
    Iy, Ix = np.gradient(It1)
    dp = 1
    #calculate appearance bases
    num_bases = bases.shape[2]
    #calculate BB.T
    orthobases = bases.reshape(-1,num_bases)
    bases_sum = 0
    for i in range(num_bases):
        bases_sum += orthobases[:,i] @ orthobases[:,i].T
  
    
    
    # pre-computed
    rows_img, cols_img = It.shape
    rows_rect, cols_rect = x2 - x1, y2 - y1
    dp = [[cols_img], [rows_img]]

    # template-related can be precomputed
    Iy, Ix = np.gradient(It1)
    y = np.arange(0, rows_img, 1)
    x = np.arange(0, cols_img, 1)     
    c = np.linspace(x1, x2, cols_rect)
    r = np.linspace(y1, y2, rows_rect)
    cc, rr = np.meshgrid(c, r)
    spline = RectBivariateSpline(y, x, It)
    T = spline.ev(rr, cc)
    spline_gx = RectBivariateSpline(y, x, Ix)
    spline_gy = RectBivariateSpline(y, x, Iy)

    #evaluate jacobian (2,2)
    jac = np.array([[1,0],[0,1]])

    while np.square(dp).sum() > threshold:
        
        #warp image
        x1_w, y1_w = x1 + p0[0], y1 + p0[1]
        x2_w, y2_w = x2 + p0[0], y2 + p0[1]
    
        cw = np.linspace(x1_w, x2_w, cols_rect)
        rw = np.linspace(y1_w, y2_w, rows_rect)
        ccw, rrw = np.meshgrid(cw, rw)
        spline1 = RectBivariateSpline(y, x, It1)
        warpImg = spline1.ev(rrw, ccw)
        
        #compute error image
        err = T - warpImg
        errImg = err.reshape(-1,1) 
        errImg = (1 - bases_sum) * errImg
        
        #compute gradient
        Ix_w = spline_gx.ev(rrw, ccw)
        Iy_w = spline_gy.ev(rrw, ccw)
        #I is (n,2)
        I = np.vstack((Ix_w.ravel(),Iy_w.ravel())).T
        
        #computer Hessian
        delta = I @ jac 
        delta = (1 - bases_sum) * delta
        #H is (2,2)
        H = delta.T @ delta
           
        #compute dp
        #dp is (2,2)@(2,n)@(n,1) = (2,1)
        dp = np.linalg.inv(H) @ (delta.T) @ errImg
        
        #update parameters
        p0[0] += dp[0,0]
        p0[1] += dp[1,0]
    return p0  
