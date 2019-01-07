import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import animation
import matplotlib.patches as patches
from scipy.ndimage import correlate, convolve
from mpl_toolkits.mplot3d import Axes3D

img = np.load('lena.npy')

# template cornes in image space [[x1, x2, x3, x4], [y1, y2, y3, y4]]
pts = np.array([[248, 292, 248, 292],
                [252, 252, 280, 280]])

# size of the template (h, w)
dsize = np.array([pts[1, 3] - pts[1, 0] + 1,
                  pts[0, 1] - pts[0, 0] + 1])

# set template corners
tmplt_pts = np.array([[0, dsize[1]-1, 0, dsize[1], -1],
                      [0, 0, dsize[0] - 1, dsize[0] - 1]])


# apply warp p to template region of img
def imwarp(p):
    global img, dsize
    return img[p[1]:(p[1]+dsize[0]), p[0]:(p[0]+dsize[1])]


# get positive example
gnd_p = np.array([252, 248])  # ground truth warp
x = imwarp(gnd_p)  # the template

# stet up figure
fig, axarr = plt.subplots(1, 3)
axarr[0].imshow(img, cmap=plt.get_cmap('gray'))
patch = patches.Rectangle((gnd_p[0], gnd_p[1]), dsize[1], dsize[0],
                          linewidth=1, edgecolor='r', facecolor='none')
axarr[0].add_patch(patch)
axarr[0].set_title('Image')

cropax = axarr[1].imshow(x, cmap=plt.get_cmap('gray'))
axarr[1].set_title('Cropped Image')

dx = np.arange(-np.floor(dsize[1]/2), np.floor(dsize[1]/2)+1, dtype=int)
dy = np.arange(-np.floor(dsize[0]/2), np.floor(dsize[0]/2)+1, dtype=int)
[dpx, dpy] = np.meshgrid(dx, dy)
dpx = dpx.reshape(-1, 1)
dpy = dpy.reshape(-1, 1)
dp = np.hstack((dpx, dpy))
N = dpx.size

all_patches = np.ones((N*dsize[0], dsize[1]))
all_patchax = axarr[2].imshow(all_patches, cmap=plt.get_cmap('gray'),
                              aspect='auto', norm=colors.NoNorm())
axarr[2].set_title('Concatenation of Sub-Images (X)')

X = np.zeros((N, N))
Y = np.zeros((N, 1))

sigma = 5


def init():
    return [cropax, patch, all_patchax]


def animate(i):
    global X, Y, dp, gnd_p, sigma, all_patches, patch, cropax, all_patchax, N

    if i < N:  # If the animation is still running
        xn = imwarp(dp[i, :] + gnd_p)
        X[:, i] = xn.reshape(-1)
        Y[i] = np.exp(-np.dot(dp[i, :], dp[i, :])/sigma)
        all_patches[(i*dsize[0]):((i+1)*dsize[0]), :] = xn
        cropax.set_data(xn)
        all_patchax.set_data(all_patches.copy())
        all_patchax.autoscale()
        patch.set_xy(dp[i, :] + gnd_p)
        return [cropax, patch, all_patchax]
    else:  # Stuff to do after the animation ends
        fig3d = plt.figure()  
        ax3d = fig3d.add_subplot(111, projection='3d')        
        ax3d.plot_surface(dpx.reshape(dsize), dpy.reshape(dsize),
                          Y.reshape(dsize), cmap=plt.get_cmap('coolwarm'))

        # Place your solution code for question 4.3 here
        
        #when penalty is 0
        penalty = 0
        # visualize g
        g = np.linalg.inv(X @ X.T + penalty * np.eye(X.shape[0])) @ X @ Y  
        plt.figure()
        plt.imshow(g.reshape(dsize), cmap = 'gray')
        plt.title('Visualization of weight vector g when penalty is 0')
        
        #correlate
        plt.figure()
        result = correlate(img, g.reshape(dsize))
        plt.imshow(result, cmap='gray')
        plt.title('Visualization of correlation responses when penalty is 0')

#        
#        #convolve
        plt.figure()
        result = convolve(img, g.reshape(dsize))
        plt.imshow(result, cmap='gray')
        plt.title('Before Rotation: Visualization of convolution responses when penalty is 0')
#        #here we rotate filter twice
        g = np.rot90(g, 2)
        result = convolve(img, g.reshape(dsize))
        plt.figure()
        plt.imshow(result, cmap='gray')
        plt.title('After Rotation: Visualization of convolution responses when penalty is 0')
#        
#        #when penalty is 1
        penalty = 1   
        # g is (N, 1)
        g = np.linalg.inv(X @ X.T + penalty * np.eye(X.shape[0])) @ X @ Y  
        plt.figure()
        plt.imshow(g.reshape(dsize), cmap='gray')
        plt.title('Visualization of weight vector g when penalty is 1')
 
#        #correlate
        plt.figure()
        result = correlate(img, g.reshape(dsize))
        plt.imshow(result, cmap='gray')
        plt.title('Visualization of correlation responses when penalty is 1')
       
#        #convolve
        plt.figure()
        result = convolve(img, g.reshape(dsize))
        plt.imshow(result, cmap='gray')
        plt.title('Before Rotation: Visualization of convolution responses when penalty is 1')
        #here we rotate filter twice
        g = np.rot90(g, 2)
        plt.figure()
        result = convolve(img, g.reshape(dsize))
        plt.imshow(result, cmap='gray')
        plt.title('After Rotation: Visualization of convolution responses when penalty is 1')

        return []


# Start the animation
ani = animation.FuncAnimation(fig, animate, frames=N+1,
                              init_func=init, blit=True,
                              repeat=False, interval=10)
plt.show()


