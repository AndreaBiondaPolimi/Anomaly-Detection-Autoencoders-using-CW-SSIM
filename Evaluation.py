import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import albumentations as A
import cv2

from collections import OrderedDict
from AutoencoderModels import Model_ssim_skip, Model_noise_skip, Model_noise_mod
from DataLoader import load_patches, load_patches_from_file_fixed, load_patches_from_file, load_gt_from_file
from Steerables.metrics_TF import Metric_win
from skimage.metrics import structural_similarity as ssim
from skimage import morphology 
from scipy import integrate 
from multiprocessing import Pool
from Utils import visualize_results, preprocess_data, bg_mask, post_reconstruction, batch_evaluation, get_performance

tf.keras.backend.set_floatx('float64')

ae_patch_size = 256
ae_stride = 16
ae_batch_splits = 12

cut_size = (0, 688, 0, 1024)
border_size = 5


### Utils ###
def image_reconstruction (y_valid, loss_type):
    reconstrunction = np.zeros((cut_size[1]-cut_size[0], cut_size[3]-cut_size[2]))
    normalizator = np.zeros((cut_size[1]-cut_size[0], cut_size[3]-cut_size[2]))
    
    i=0; j=0
    for idx in range (len(y_valid)):
        #reconstrunction [j:j+ae_patch_size, i:i+ae_patch_size] += post_reconstruction(y_valid[idx], loss_type)
        reconstrunction [j:j+ae_patch_size, i:i+ae_patch_size] += y_valid[idx]
        normalizator [j:j+ae_patch_size, i:i+ae_patch_size] += np.ones((ae_patch_size,ae_patch_size))
        if (i+ae_patch_size < cut_size[3]-cut_size[2]):
            i=i+ae_stride
        else:
            i=0; j=j+ae_stride
    reconstrunction =  reconstrunction/normalizator

    return reconstrunction

def compute_performance(args):
    tresh = args['tresh']
    x_valid = args['x_valid']
    residual = args['residual']
    valid_gt = args['valid_gt']

    scoremap = get_scoremap(x_valid, residual, tresh)
    iou, tpr, fpr, ovr = get_performance(valid_gt, scoremap)

    return {'tresh': tresh, 'iou':iou, 'tpr': tpr, 'fpr': fpr, 'ovr': ovr}

### Validation ###
def validation_complete():
    tprs = []; fprs = []; ious = []; ovrs = []
    for i in range (1,41):
        iou, tpr, fpr, ovr = validation(str(i).zfill(2))
        tprs.append(tpr); fprs.append(fpr); ious.append(iou); ovrs.append(ovr)
        print ()
    
    flat_ovr = [item for sublist in ovrs for item in sublist]
    flat_ovr = np.sort(flat_ovr)
    print ("Mean IoU:", np.mean(ious))
    print ("Min OVR:", np.min(flat_ovr[int(-len(flat_ovr)/2):]))    
    tprs = np.mean(tprs, axis=0); fprs = np.mean(fprs, axis=0)
    print ("AUC: " + str(-1 * integrate.trapz(tprs, fprs)))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot (fprs, tprs)
    plt.show()

def validation (n_img):
    print(n_img)
    valid_patches, valid_img = load_patches_from_file('Dataset\\SEM_Data\\Anomalous\\images\\ITIA11' + n_img + '.tif', patch_size=ae_patch_size, 
        random=False, stride=ae_stride, cut_size=cut_size) 
    valid_gt = load_gt_from_file ('Dataset\\SEM_Data\\Anomalous\\gt\\ITIA11' + n_img + '_gt.png', cut_size=cut_size)
    valid_gt[valid_gt > 0] = 1

    autoencoder = Model_noise_skip(input_shape=(ae_patch_size,ae_patch_size,1), latent_dim=500)
    loss_type = 'cwssim_loss'
    #loss_type = 'ms_ssim_loss'
    #loss_type = 'ssim_loss'
    autoencoder.load_weights('Weights\\' + loss_type + '\\check_epoch120.h5')

    _, y_valid = batch_evaluation(valid_patches, autoencoder, ae_batch_splits)
    reconstruction = image_reconstruction(y_valid, loss_type)

    iou, tprs, fprs, ovr = model_evaluation (valid_img, reconstruction, valid_gt)

    return iou, tprs, fprs, ovr

def model_evaluation (x_valid, y_valid, valid_gt, step=0.01):
    #Compute residual map
    residual = get_residual(x_valid.copy(), y_valid.copy())

    #Compute scores async
    tprs = []; fprs = []; ious = []; ovrs = []
    args = [{'tresh': tresh, 'x_valid': x_valid.copy(), 'residual': residual.copy(), 'valid_gt': valid_gt.copy()} for tresh in np.arange (0., 0.4, step)] 
    with Pool(processes=6) as pool:  # multiprocessing.cpu_count()
        results = pool.map(compute_performance, args, chunksize=1)
    for result in results:
        tresh = result['tresh']; iou = result['iou']; tpr = result['tpr']; fpr = result['fpr']; ovr = result['ovr']
        tprs.append(tpr); fprs.append(fpr); ious.append(iou); ovrs.append(ovr)

    #Calculate evaluations
    tprs = np.array(tprs); fprs = np.array(fprs); ovrs = np.array(ovrs)
    iou = np.max(ious); iouidx = np.argmax(ious)*step; ovr = ovrs[fprs<=0.05][0]

    #Print evaluations
    print ("IoU:", iou , "(tresh:", iouidx , ")")
    print ("AUC: " + str(-1 * integrate.trapz(tprs, fprs)))

    #Print heatmap & scoremap
    #scoremap = get_scoremap(x_valid.copy(), residual.copy(), np.argmax(ious)*step)
    #visualize_results(x_valid/255, y_valid, "Residual map")
    #visualize_results(valid_gt, scoremap, "Score map")

    return iou, tprs, fprs, ovr


def get_residual (x_valid, y_valid):
    depr_mask = np.ones_like(x_valid) * 0.5
    depr_mask[border_size:x_valid.shape[0]-border_size, border_size:x_valid.shape[1]-border_size] = 1
    pad_size = int((1024 - cut_size[1])/2)
    #data_range = np.max(y_valid)
    padding = A.PadIfNeeded(1024, 1024, p=1.0)
    
    x_valid = padding(image=x_valid/255)['image']
    y_valid = padding(image=y_valid)['image']

    #Residual map ssim
    ssim_configs = [17, 15, 13, 11, 9, 7, 5, 3]
    residual_ssim = np.zeros_like(x_valid)
    for win_size in ssim_configs:
        residual_ssim += (1 - ssim(x_valid, y_valid, win_size=win_size, full=True, data_range=1.)[1])
    residual_ssim = residual_ssim / len(ssim_configs)
    residual_ssim = residual_ssim[pad_size: residual_ssim.shape[0]-pad_size, 0:residual_ssim.shape[1]]
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
    residual_cwssim = residual_cwssim[pad_size: residual_cwssim.shape[0]-pad_size, 0:residual_cwssim.shape[1]]
    #visualize_results(residual_cwssim, y_valid, "aa") 

    residual = (residual_cwssim + residual_ssim) / 2
    residual = residual * depr_mask
    #visualize_results(residual, y_valid, "aa")  

    return residual      

def get_scoremap(x_valid, residual_ssim, ssim_treshold=0.15):
    bg_m = bg_mask(x_valid, 30, cv2.THRESH_BINARY_INV)
    scoremap = np.zeros_like(x_valid)
    kernel = morphology.disk(10)
    
    #Apply treshold
    scoremap[residual_ssim >= ssim_treshold] = 1

    #Postprocessing
    scoremap = morphology.opening(scoremap, kernel)
    if (ssim_treshold > 0):
        scoremap = scoremap * (1 - bg_m)

    return scoremap 


if __name__ == "__main__":
    #validation('06')
    validation_complete()











def batch_reconstruction_augmented (valid_patches, autoencoder):
    x_valid = preprocess_data(valid_patches)
    x_valid = np.squeeze(x_valid)
    y_valid = []

    for x in x_valid:
        batch = []
        batch.append(x)

        for angle in range(30, 331, 30):
            r_rotation = A.Rotate((-angle,-angle), border_mode=cv2.BORDER_REFLECT101, p=1.)
            augmented = r_rotation(image=x.copy())['image']
            batch.append(augmented)

        batch = np.array(batch)[..., tf.newaxis]
        batch_res = autoencoder(batch).numpy()     
        batch_res = np.squeeze(batch_res)

        res = batch_res[0]
        for angle in range(30, 331, 30):
            l_rotation = A.Rotate((angle,angle), border_mode=cv2.BORDER_REFLECT101, p=1.)
            l_norm_rotation = A.Rotate((angle,angle), border_mode=cv2.BORDER_CONSTANT, p=1.)

            augmented = l_rotation(image=batch_res[int(angle/30)])['image']
            norm_aug = l_norm_rotation(image=batch_res[int(angle/30)])['image']

            normalizator = np.ones_like(augmented)
            temp = norm_aug.copy()
            temp[temp > 0] = 1
            normalizator += temp

            res = res + (augmented * temp)
            res = res / normalizator

        y_valid.append(res)

    y_valid = np.array(y_valid)
    return x_valid, y_valid

def fast_model_evaluation(x_valid, y_valid, valid_gt, tresh=0.16):
    scoremap = get_scoremap(x_valid.copy(), y_valid.copy(), tresh)
    iou, _, _ = get_performance(valid_gt, scoremap)
    
    print ("IoU:", iou)
    visualize_results(x_valid/255, y_valid, "Residual map")
    visualize_results(valid_gt, scoremap, "Score map")

def get_residual_patchwise (x_valid, y_valid):
    residual = []
    metric_tf_7 = Metric_win (window_size=7, patch_size=ae_patch_size)

    for idx in range (len(y_valid)): 
        residual_ssim = (1 - ssim(x_valid[idx], y_valid[idx], win_size=17, full=True, data_range=1.)[1]) 
        residual_ssim += (1 - ssim(x_valid[idx], y_valid[idx], win_size=15, full=True, data_range=1.)[1]) 
        residual_ssim += (1 - ssim(x_valid[idx], y_valid[idx], win_size=13, full=True, data_range=1.)[1]) 
        residual_ssim += (1 - ssim(x_valid[idx], y_valid[idx], win_size=11, full=True, data_range=1.)[1]) 
        residual_ssim += (1 - ssim(x_valid[idx], y_valid[idx], win_size=9, full=True, data_range=1.)[1]) 
        residual_ssim += (1 - ssim(x_valid[idx], y_valid[idx], win_size=7, full=True, data_range=1.)[1]) 
        residual_ssim += (1 - ssim(x_valid[idx], y_valid[idx], win_size=5, full=True, data_range=1.)[1]) 
        residual_ssim += (1 - ssim(x_valid[idx], y_valid[idx], win_size=3, full=True, data_range=1.)[1]) 
        residual_ssim = residual_ssim / 8
        
        residual_cwssim = metric_tf_7.CWSSIM(np.expand_dims(np.expand_dims(x_valid[idx], 0), 3), np.expand_dims(np.expand_dims(y_valid[idx], 0), 3), height=9, orientations=5, full=True).numpy()[0]
        residual_cwssim += metric_tf_7.CWSSIM(np.expand_dims(np.expand_dims(x_valid[idx], 0), 3), np.expand_dims(np.expand_dims(y_valid[idx], 0), 3), height=7, orientations=5, full=True).numpy()[0]
        residual_cwssim = np.squeeze(residual_cwssim)
        residual_cwssim = residual_cwssim/2
        residual_cwssim = 1 - residual_cwssim

        residual.append ((residual_cwssim + residual_ssim) / 2)

    return np.array(residual)