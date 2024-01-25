import numpy as np

class WeightFunctions:
    def Gaussian(x_dist, eps=1.):
        return np.exp(-(eps*x_dist)**2)

    def BumpFunction(x_dist, eps=1.):
        # filter x_dist to get rid of x_dist >= 1/eps, this prevents overflow warnings
        x_dist_filt = np.where(x_dist < 1/eps, x_dist, 0.)
        f_mat = np.where(x_dist < 1/eps, np.exp(-1./(1.-(eps*x_dist_filt)**2)), 0.)
        return f_mat/np.exp(-1)  # normalize output so x_dist = 0 corresponds to f = 1
    
    def ThinPlateSpline(x_dist):
        return np.multiply(np.power(x_dist, 2), np.log(x_dist))