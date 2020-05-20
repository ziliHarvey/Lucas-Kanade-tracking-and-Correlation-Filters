import numpy as np
from scipy.interpolate import RectBivariateSpline


def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]
    #   o____x
    #   |
    #   |
    #   y image(y, x) opencv convention
	
    
    threshold = 0.1
    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
    Iy, Ix = np.gradient(It1)

    rows_img, cols_img = It.shape
    rows_rect, cols_rect = x2 - x1, y2 - y1
    dp = [[cols_img], [rows_img]] #just an intial value to enforce the loop

    # template-related can be precomputed
    y = np.arange(0, rows_img, 1)
    x = np.arange(0, cols_img, 1)     
    c = np.linspace(x1, x2, cols_rect)
    r = np.linspace(y1, y2, rows_rect)
    cc, rr = np.meshgrid(c, r)
    spline = RectBivariateSpline(y, x, It)
    T = spline.ev(rr, cc)

    # in translation model jacobian is not related to coordinates
    jac = np.array([[1,0],[0,1]])

    while np.square(dp).sum() > threshold:
            
        # warp image using translation motion model
        x1_w, y1_w, x2_w, y2_w = x1+p0[0], y1+p0[1], x2+p0[0], y2+p0[1]
        
        cw = np.linspace(x1_w, x2_w, cols_rect)
        rw = np.linspace(y1_w, y2_w, rows_rect)
        ccw, rrw = np.meshgrid(cw, rw)
        spline1 = RectBivariateSpline(y, x, It1)
        warpImg = spline1.ev(rrw, ccw)
        
        #compute error image
        err = T - warpImg
        errImg = err.reshape(-1,1) 
        
        #compute gradient
        spline_gx = RectBivariateSpline(y, x, Ix)
        Ix_w = spline_gx.ev(rrw, ccw)

        spline_gy = RectBivariateSpline(y, x, Iy)
        Iy_w = spline_gy.ev(rrw, ccw)
        #I is (n,2)
        I = np.vstack((Ix_w.ravel(),Iy_w.ravel())).T
        
        
        #computer Hessian
        delta = I @ jac 
        #H is (2,2)
        H = delta.T @ delta
        
        #compute dp
        #dp is (2,2)@(2,n)@(n,1) = (2,1)
        dp = np.linalg.inv(H) @ (delta.T) @ errImg
        
        #update parameters
        p0[0] += dp[0,0]
        p0[1] += dp[1,0]
        
    p = p0
    return p
