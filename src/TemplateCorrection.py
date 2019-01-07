# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 15:42:31 2018

@author: de'l'l
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline
def TemplateCorrection(I0, It1, rect0, rect, p0 = np.zeros(2)):
    threshold = 0.1
    x1, y1, x2, y2 = rect0[0], rect0[1], rect0[2], rect0[3]
    x1_t, y1_t, x2_t, y2_t = rect[0], rect[1], rect[2], rect[3]
    Iy, Ix = np.gradient(It1)
    dp = 10
    while np.square(dp).sum() > threshold:
        px, py = p0[0], p0[1]
        x1_w, y1_w, x2_w, y2_w = x1_t+px, y1_t+py, x2_t+px, y2_t+py
        
        x = np.arange(0, I0.shape[0], 1)
        y = np.arange(0, I0.shape[1], 1)
        
        c = np.linspace(x1, x2, 87)
        r = np.linspace(y1, y2, 36)
        cc, rr = np.meshgrid(c, r)
    
        cw = np.linspace(x1_w, x2_w, 87)
        rw = np.linspace(y1_w, y2_w, 36)
        ccw, rrw = np.meshgrid(cw, rw)
        
        spline = RectBivariateSpline(x, y, I0)
        T = spline.ev(rr, cc)
        
        spline1 = RectBivariateSpline(x, y, It1)
        warpImg = spline1.ev(rrw, ccw)
        
        #compute error image
        err = T - warpImg
        errImg = err.reshape(-1,1) 
        
        #compute gradient
        spline_gx = RectBivariateSpline(x, y, Ix)
        Ix_w = spline_gx.ev(rrw, ccw)

        spline_gy = RectBivariateSpline(x, y, Iy)
        Iy_w = spline_gy.ev(rrw, ccw)
        #I is (n,2)
        I = np.vstack((Ix_w.ravel(),Iy_w.ravel())).T
        
        #evaluate jacobian (2,2)
        jac = np.array([[1,0],[0,1]])
        
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
        
    p_star = p0
    return p_star