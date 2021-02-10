import numpy as np
import scipy as sp
from skimage import morphology 

class MahalanobisDetector ():
    def __init__(self):
        self.mean = None
        self.cov = None
        self.cov_i = None

    def calculate_statistics (self, data_train):
        self.mean = np.mean(data_train, 0)
        self.cov = np.cov(data_train.T)
        self.cov_i = np.linalg.inv(self.cov)

    def calculate_distance (self, f_valids):
        y_valid = []
        for f_valid in f_valids:
            ret = sp.spatial.distance.mahalanobis(f_valid, self.mean, self.cov_i)
            y_valid.append(ret) 
        return np.array(y_valid)