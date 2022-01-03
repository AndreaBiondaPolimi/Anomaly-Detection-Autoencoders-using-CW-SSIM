import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import albumentations as A

from AutoencoderModels import Model_ssim_skip, Model_noise_skip, Model_noise_skip_01
from DataLoader import load_patches, load_patches_from_file_fixed, load_patches_from_file, load_gt_from_file, load_images, load_patches_from_image
from multiprocessing import Pool

from Utils import get_ovr, visualize_results, preprocess_data, bg_mask, batch_evaluation, get_performance, get_roc, image_evaluation, get_iou


import metrics_color_TF as mct

tf.keras.backend.set_floatx('float64')

if __name__ == "__main__":
    window_size = 7
    scales = 5
    orients = 5
    patch_size = 256
    

    train_patches = load_patches('Dataset\\MVTec_Data\\' + "carpet" + '\\Normal', patch_size=patch_size, n_patches=1, random=True, preprocess_limit=0, resize=None, grayscale=False)
    im1 = preprocess_data(train_patches, False)

    train_patches = load_patches('Dataset\\MVTec_Data\\' + "carpet" + '\\Normal', patch_size=patch_size, n_patches=1, random=True, preprocess_limit=0, resize=None, grayscale=False)
    im2 = preprocess_data(train_patches, False)
   
    #mct.color_quantization(im1)
    
    
    mct.sklearn_color_quantization(im1)