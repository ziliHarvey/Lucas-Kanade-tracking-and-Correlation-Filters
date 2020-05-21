import numpy as np
from scipy.interpolate import RectBivariateSpline

def TemplateCorrection(T, It1, rect, p0 = np.zeros(2)):
    threshold = 0.1
    x1_t, y1_t, x2_t, y2_t = rect[0], rect[1], rect[2], rect[3]
    Iy, Ix = np.gradient(It1)
    
    rows_img, cols_img = It1.shape
    rows_rect, cols_rect = T.shape
    dp = [[cols_img], [rows_img]]

    # what can be precomputed
    y = np.arange(0, rows_img, 1)
    x = np.arange(0, cols_img, 1)
    spline1 = RectBivariateSpline(y, x, It1)
    spline_gx = RectBivariateSpline(y, x, Ix)
    spline_gy = RectBivariateSpline(y, x, Iy)
    jac = np.array([[1,0],[0,1]])
    
    while np.square(dp).sum() > threshold:
        x1_w, y1_w = x1_t + p0[0], y1_t + p0[1] 
        x2_w, y2_w = x2_t + p0[0], y2_t + p0[1]
    
        cw = np.linspace(x1_w, x2_w, cols_rect)
        rw = np.linspace(y1_w, y2_w, rows_rect)
        ccw, rrw = np.meshgrid(cw, rw)
        warpImg = spline1.ev(rrw, ccw)
        
        #compute error image
        err = T - warpImg
        errImg = err.reshape(-1,1) 
        
        #compute gradient
        
        Ix_w = spline_gx.ev(rrw, ccw) 
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
    
    rect[0] += p0[0]
    rect[1] += p0[1]
    rect[2] += p0[0]
    rect[3] += p0[1]
