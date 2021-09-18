import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import albumentations as A
import cv2

from collections import OrderedDict
from AutoencoderModels import Model_ssim_skip, Model_noise_skip, Model_noise_skip_01
from DataLoader import load_patches, load_patches_from_file_fixed, load_patches_from_file, load_gt_from_file, load_images, load_patches_from_image
from Steerables.metrics_TF import Metric_win
from skimage.metrics import structural_similarity as ssim
from skimage import morphology 
from scipy import integrate 
from multiprocessing import Pool
from Utils import get_ovr, visualize_results, preprocess_data, bg_mask, batch_evaluation, get_performance, get_roc, image_evaluation, get_iou
from Steerables.AnomalyMetrics import cw_ssim_metric, ssim_metric, l2_metric

tf.keras.backend.set_floatx('float64')

weights_file = 'Weights\\new_weights\\check_epoch120.h5'
anomaly_metrics = 'cwssim_loss'
ae_patch_size = 256
ae_stride = 16
ae_batch_splits = 64
invert_reconstruction = False
fpr_value = 0.01

cut_size = (0, 1024, 0, 1024)
border_size = 5
step = 0.0005

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
    residual = args['residual']
    valid_gt = args['valid_gt']

    scoremap = get_scoremap(residual, tresh)
    tpr, fpr = get_roc(valid_gt, scoremap)
    iou = get_iou(valid_gt, scoremap)
    ovr = get_ovr(valid_gt, scoremap)

    tresh = None; residual = None; valid_gt = None

    return {'tpr': tpr, 'fpr': fpr, 'iou': iou, 'ovr': ovr}





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
    imgs = load_images("Dataset\\MVTec_Data\\Anomalous\\IMG", 512, preprocess_limit=0)
    gts = load_images("Dataset\\MVTec_Data\\Anomalous\\GT", 512, preprocess_limit=0)
    
    avg_au_roc = 0; avg_au_iou = 0; avg_au_pro = 0
    for i in range (len(imgs)):
        au_roc, au_iou, au_pro = evaluation(str(i).zfill(2), imgs[i], gts[i], ovr_threshold, False)
        
        avg_au_roc += au_roc
        avg_au_iou += au_iou
        avg_au_pro += au_pro
        print ()
    
    print ("MEAN Area under ROC:", avg_au_roc/len(imgs))
    print ("MEAN Area under IOU:", avg_au_iou/len(imgs))
    print ("MEAN Area under PRO:", avg_au_pro/len(imgs))



def evaluation (n_img, valid_img, valid_gt, ovr_threshold, to_show):
    print("TEST IMAGE ", n_img)
    valid_patches, valid_img = load_patches_from_image(valid_img, patch_size=ae_patch_size, random=False, stride=ae_stride) 
    valid_gt[valid_gt > 0] = 1

    autoencoder = Model_noise_skip(input_shape=(ae_patch_size,ae_patch_size,1))
    autoencoder.load_weights(weights_file)
    #autoencoder.summary()

    #Patch-Wise reconstruction
    _, y_valid = batch_evaluation(valid_patches, autoencoder, ae_batch_splits)
    reconstruction = image_reconstruction(y_valid)

    #Full reconstruction
    #reconstruction = image_evaluation(valid_img, autoencoder)
    
    #visualize_results(valid_img/255, reconstruction, "original vs reco")

    au_roc, au_iou, au_pro = model_evaluation (valid_img, reconstruction, valid_gt, ovr_threshold, to_show)
    
    return au_roc, au_iou, au_pro



def model_evaluation (x_valid, y_valid, valid_gt, ovr_threshold, to_show):
    #Compute residual map
    residual = get_residual(x_valid.copy(), y_valid.copy())
    #visualize_results(y_valid, residual, "reco vs residual")


    #for tresh in np.arange (0.1, 0.6, step):
        #scoremap = get_scoremap(residual, tresh)
        #ovr = get_ovr(valid_gt, scoremap)
        #iou = get_iou(valid_gt, scoremap)
        #print (iou)
        #print (ovr)


    #Compute roc scores async
    tprs = []; fprs = []; ious = [] ;ovrs = []
    args = [{'tresh': tresh.copy(), 'residual': residual.copy(), 'valid_gt': valid_gt.copy()} for tresh in np.arange (0., 0.3, step)] 
    with Pool(processes=4) as pool:  # multiprocessing.cpu_count()
        results = pool.map(compute_performance, args, chunksize=1)
    
    for result in results:
        tpr = result['tpr']; fpr = result['fpr']; iou = result['iou']; ovr = result['ovr']
        #print(fpr)
        if (fpr <= 0.3):
            #print (tpr, fpr, iou)
            tprs.append(tpr); fprs.append(fpr); ious.append(iou); ovrs.append(ovr)
    tprs = np.array(tprs); fprs = np.array(fprs); ious = np.array(ious); ovrs = np.array(ovrs)
    
    au_roc = (-1 * integrate.trapz(tprs, fprs))/(np.max(fprs)*np.max(tprs))
    au_iou = (-1 * integrate.trapz(ious, fprs))/(np.max(fprs)*np.max(ious))
    au_pro = (-1 * integrate.trapz(ovrs, fprs))/(np.max(fprs)*np.max(ovrs))

    print ("COUNT FPR: ", len(fprs))
    print ("MAX FPR: ", np.max(fprs))
    print ("Area under ROC:", au_roc)
    print ("Area under IOU:", au_iou)  
    print ("Area under PRO:", au_pro)  

    #plt.xlabel('FPR')
    #plt.ylabel('TPR')
    #plt.plot (fprs, ious)
    #plt.show()

    #plt.xlabel('FPR')
    #plt.ylabel('TPR')
    #plt.plot (fprs, tprs)
    #plt.show()


    return au_roc, au_iou, au_pro




def get_residual (x_valid, y_valid):
    #depr_mask = np.ones_like(x_valid) * 0.5
    #depr_mask[border_size:x_valid.shape[0]-border_size, border_size:x_valid.shape[1]-border_size] = 1
    #pad_size = int((1024 - cut_size[1])/2)
    #data_range = np.max(y_valid)
    #padding = A.PadIfNeeded(1024, 1024, p=1.0)
    
    #x_valid = padding(image=x_valid/255)['image']
    #y_valid = padding(image=y_valid)['image']

    x_valid = x_valid/255

    if (anomaly_metrics == 'cwssim_loss'):
        residual = cw_ssim_metric (x_valid, y_valid, 0)    
    elif (anomaly_metrics == 'ssim_loss' or anomaly_metrics == 'ms_ssim_loss'):
        residual = ssim_metric (x_valid, y_valid, 0)    
    elif (anomaly_metrics == 'l2_loss'):
        residual = l2_metric (x_valid, y_valid, 0)    
    else:
        residual = cw_ssim_metric (x_valid, y_valid, 0)    

    #residual = residual * depr_mask
    return residual      

def get_scoremap(residual_ssim, ssim_treshold=0.15):
    #bg_m = bg_mask(x_valid, 30, cv2.THRESH_BINARY_INV)
    scoremap = np.zeros_like(residual_ssim)
    kernel = morphology.disk(10)
    
    #Apply treshold
    scoremap[residual_ssim >= ssim_treshold] = 1

    #Postprocessing
    scoremap = morphology.opening(scoremap, kernel)
    #if (ssim_treshold > 0):
        #scoremap = scoremap * (1 - bg_m)

    #plt.imshow(scoremap)
    #plt.show()

    return scoremap 


if __name__ == "__main__":
    #tresh = validation ()
    tresh = 0.1
    evaluation_complete(tresh)
    
    #evaluation("25", tresh, True)



