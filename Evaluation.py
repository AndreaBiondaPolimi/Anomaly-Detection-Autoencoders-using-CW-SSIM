import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import albumentations as A
import cv2

from collections import OrderedDict
from AutoencoderModels import Model_ssim_skip, Model_noise_skip, Model_noise_skip_01
from DataLoader import load_patches, load_patches_from_file_fixed, load_patches_from_file, load_gt_from_file
from Steerables.metrics_TF import Metric_win
from skimage.metrics import structural_similarity as ssim
from skimage import morphology 
from scipy import integrate 
from multiprocessing import Pool
from Utils import visualize_results, preprocess_data, bg_mask, batch_evaluation, get_performance, get_roc, get_ovr_iou, image_evaluation
from Steerables.AnomalyMetrics import cw_ssim_metric, ssim_metric, l2_metric

tf.keras.backend.set_floatx('float64')

weights_file = 'Weights\\cwssim_loss\\check_epoch_256_6_5_7.h5'
anomaly_metrics = 'cwssim_loss'
ae_patch_size = 256
ae_stride = 16
ae_batch_splits = 12
invert_reconstruction = False
fpr_value = 0.05

cut_size = (0, 688, 0, 1024)
border_size = 5
step = 0.01

### Utils ###
def image_reconstruction (y_valid):
    reconstrunction = np.zeros((cut_size[1]-cut_size[0], cut_size[3]-cut_size[2]))
    normalizator = np.zeros((cut_size[1]-cut_size[0], cut_size[3]-cut_size[2]))
    
    i=0; j=0
    for idx in range (len(y_valid)):
        if (invert_reconstruction):
            reconstrunction [j:j+ae_patch_size, i:i+ae_patch_size] += (y_valid[idx] * -1)
        else:
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
    tpr, fpr = get_roc(valid_gt, scoremap)

    return {'tpr': tpr, 'fpr': fpr}





### Validation ###
def validation ():
    print ("START VALIDATION PHASE")
    
    valid_fprs = []
    for i in [8,15,27,31,35]:
        iou, tpr, fpr, ovr = evaluation(str(i).zfill(2), None, False)
        fpr[fpr > fpr_value] = 0
        valid_fprs.append (np.argmax(fpr))

    ovr_threshold = np.mean(valid_fprs) * step
    print ("OVR Threshold:", ovr_threshold)

    return ovr_threshold



def evaluation_complete(ovr_threshold):
    print ("START TEST PHASE")    
    tprs = []; fprs = []; ious = []; ovrs = []
    for i in range (1,41):
        if (i not in [8,15,27,31,35]):
            iou, tpr, fpr, ovr = evaluation(str(i).zfill(2), ovr_threshold, False)
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



def evaluation (n_img, ovr_threshold, to_show):
    print("TEST IMAGE ", n_img)
    valid_patches, valid_img = load_patches_from_file('Dataset\\SEM_Data\\Anomalous\\images\\ITIA11' + n_img + '.tif', patch_size=ae_patch_size, 
        random=False, stride=ae_stride, cut_size=cut_size) 
    valid_gt = load_gt_from_file ('Dataset\\SEM_Data\\Anomalous\\gt\\ITIA11' + n_img + '_gt.png', cut_size=cut_size)
    valid_gt[valid_gt > 0] = 1

    autoencoder = Model_noise_skip(input_shape=(ae_patch_size,ae_patch_size,1))
    autoencoder.load_weights(weights_file)

    #Patch-Wise reconstruction
    _, y_valid = batch_evaluation(valid_patches, autoencoder, ae_batch_splits)
    reconstruction = image_reconstruction(y_valid)

    #Full reconstruction
    #reconstruction = image_evaluation(valid_img, autoencoder)

    iou, tprs, fprs, ovr = model_evaluation (valid_img, reconstruction, valid_gt, ovr_threshold, to_show)
    
    return iou, tprs, fprs, ovr



def model_evaluation (x_valid, y_valid, valid_gt, ovr_threshold, to_show):
    #Compute residual map
    residual = get_residual(x_valid.copy(), y_valid.copy())


    #Compute roc scores async
    tprs = []; fprs = [] #ious = []; ovrs = []
    args = [{'tresh': tresh, 'x_valid': x_valid.copy(), 'residual': residual.copy(), 'valid_gt': valid_gt.copy()} for tresh in np.arange (0., 0.4, step)] 
    with Pool(processes=6) as pool:  # multiprocessing.cpu_count()
        results = pool.map(compute_performance, args, chunksize=1)
    
    for result in results:
        tpr = result['tpr']; fpr = result['fpr']
        tprs.append(tpr); fprs.append(fpr)
    tprs = np.array(tprs); fprs = np.array(fprs)
    

    #Compute iou and ovr scores
    ovr=None; iou=None
    if (ovr_threshold is not None):
        #Check threshold
        aa = np.copy(fprs)
        aa[aa > fpr_value] = 0
        idx = np.argmax(aa)
        if (ovr_threshold < idx * step):
            ovr_threshold = idx * step


        scoremap = get_scoremap(x_valid.copy(), residual.copy(), ovr_threshold)
        ovr, iou = get_ovr_iou(valid_gt, scoremap)
        tpr, fpr = get_roc(valid_gt, scoremap)
        
        #Print evaluations
        print ("FPR:", fpr)
        print ("OVR:", ovr)
        print ("IoU:", iou)
        print ("AUC: " + str(-1 * integrate.trapz(tprs, fprs)))

        #Print reconstruction, residual & score map
        if (to_show):
            visualize_results(x_valid/255, y_valid, "Reconstruction map")
            visualize_results(x_valid/255, residual, "Residual map")
            visualize_results(valid_gt, scoremap, "Score map")

    return iou, tprs, fprs, ovr


def get_residual (x_valid, y_valid):
    depr_mask = np.ones_like(x_valid) * 0.5
    depr_mask[border_size:x_valid.shape[0]-border_size, border_size:x_valid.shape[1]-border_size] = 1
    pad_size = int((1024 - cut_size[1])/2)
    #data_range = np.max(y_valid)
    padding = A.PadIfNeeded(1024, 1024, p=1.0)
    
    x_valid = padding(image=x_valid/255)['image']
    y_valid = padding(image=y_valid)['image']

    if (anomaly_metrics == 'cwssim_loss'):
        residual = cw_ssim_metric (x_valid, y_valid, pad_size)    
    elif (anomaly_metrics == 'ssim_loss' or anomaly_metrics == 'ms_ssim_loss'):
        residual = ssim_metric (x_valid, y_valid, pad_size)    
    elif (anomaly_metrics == 'l2_loss'):
        residual = l2_metric (x_valid, y_valid, pad_size)    
    else:
        residual = cw_ssim_metric (x_valid, y_valid, pad_size)    

    residual = residual * depr_mask
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
    tresh = validation ()
    evaluation_complete(tresh)
    
    #evaluation("25", tresh, True)



