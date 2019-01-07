import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image

	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    threshold = 1
    dp = 10
    Iy, Ix = np.gradient(It1)
    x1, y1, x2, y2 = 0, 0, It.shape[1], It.shape[0]
    
    #gradient, jac and hessian can be pre-computed
    
    #I here is the gradient of the template 
    Iy, Ix = np.gradient(It)
    I = np.vstack((Ix.ravel(),Iy.ravel())).T
    
    delta = np.zeros((It.shape[0]*It.shape[1], 6))
    for i in range(It.shape[0]):
        for j in range(It.shape[1]):
            #I is (1,2) for each pixel
            #Jacobian is (2,6) for each pixel
            I_indiv = np.array([I[i*It.shape[1]+j]]).reshape(1,2)
            
            jac_indiv = np.array([[j, 0, i, 0, 1, 0],
                                  [0, j, 0, i, 0, 1]]) 
            delta[i*It.shape[1]+j] = I_indiv @ jac_indiv
    
    H = delta.T @ delta
    
    M = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])
    
    while np.square(dp).sum() > threshold:
        
        x1_w = M[0,0] * x1 + M[0,1] * y1 + M[0,2]
        y1_w = M[1,0] * x1 + M[1,1] * y1 + M[1,2]
        x2_w = M[0,0] * x2 + M[0,1] * y2 + M[0,2]
        y2_w = M[0,0] * x2 + M[0,1] * y2 + M[0,2]
        
        x = np.arange(0, It.shape[0], 1)
        y = np.arange(0, It.shape[1], 1)
        
        c = np.linspace(x1, x2, It.shape[1])
        r = np.linspace(y1, y2, It.shape[0])
        cc, rr = np.meshgrid(c, r)
    
        cw = np.linspace(x1_w, x2_w, It.shape[1])
        rw = np.linspace(y1_w, y2_w, It.shape[0])
        ccw, rrw = np.meshgrid(cw, rw)
        
        spline = RectBivariateSpline(x, y, It)
        T = spline.ev(rr, cc)
        
        spline1 = RectBivariateSpline(x, y, It1)
        warpImg = spline1.ev(rrw, ccw)
    
        #compute error image
        #errImg is (n,1)
        err = T - warpImg
        errImg = err.reshape(-1,1)
        
        #update M
        dp = np.linalg.inv(H) @ (delta.T) @ errImg
        dM = np.vstack((dp.reshape(2,3), [0, 0, 1]))
        M = M @ np.linalg.inv(dM)
    
    return M
