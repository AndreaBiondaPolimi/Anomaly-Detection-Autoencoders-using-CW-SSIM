import numpy as np
from skimage.metrics import structural_similarity as ssim
from Steerables.metrics_TF import Metric_win


def cw_ssim_metric (x_valid, y_valid, pad_size_x, pad_size_y):
    #Residual map ssim
    ssim_configs = [17, 15, 13, 11, 9, 7, 5, 3]
    residual_ssim = np.zeros_like(x_valid)
    for win_size in ssim_configs:
        residual_ssim += (1 - ssim(x_valid, y_valid, win_size=win_size, full=True, data_range=1.)[1])
    residual_ssim = residual_ssim / len(ssim_configs)
    residual_ssim = residual_ssim[pad_size_x: residual_ssim.shape[0]-pad_size_x, pad_size_y:residual_ssim.shape[1]-pad_size_y]
    #visualize_results(residual_ssim, y_valid, "aa")  

    #Residual map cwssim
    cwssim_configs = [9, 8, 7]
    residual_cwssim = np.expand_dims(np.expand_dims(np.zeros_like(x_valid), 0), 3)
    metric_tf_7 = Metric_win (window_size=7, patch_size=1024)
    for height in cwssim_configs:
        residual_cwssim += (1 - metric_tf_7.CWSSIM(np.expand_dims(np.expand_dims(x_valid, 0), 3), np.expand_dims(np.expand_dims(y_valid, 0), 3), 
                        height=height, orientations=6, full=True).numpy()[0])
    residual_cwssim = residual_cwssim/len(cwssim_configs)
    residual_cwssim = np.squeeze(residual_cwssim)
    residual_cwssim = residual_cwssim[pad_size_x: residual_cwssim.shape[0]-pad_size_x, pad_size_y:residual_cwssim.shape[1]-pad_size_y]
    #visualize_results(residual_cwssim, y_valid, "aa") 

    #residual = (residual_cwssim + residual_ssim) / 2
    residual = residual_cwssim 

    return residual


def ssim_metric (x_valid, y_valid, pad_size_x, pad_size_y):
    residual = (1 - ssim(x_valid, y_valid, win_size=11, full=True)[1])
    residual = residual[pad_size_x: residual.shape[0]-pad_size_x, pad_size_y:residual.shape[1]-pad_size_y]
    return residual


def l2_metric (x_valid, y_valid, pad_size_x, pad_size_y):
    #residual = np.square(x_valid - y_valid)
    residual = np.abs(x_valid - y_valid)
    residual = residual[pad_size_x: residual.shape[0]-pad_size_x, pad_size_y:residual.shape[1]-pad_size_y]
    return residual