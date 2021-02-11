import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from AutoencoderModels import Model_ssim_skip, Model_noise_skip, Model_noise_skip_small, Model_noise_skip_wide
from DataLoader import load_patches, load_patches_from_file_fixed, load_patches_from_file
from Steerables.metrics_TF import Metric_win
from skimage.metrics import structural_similarity as ssim
from Utils import batch_evaluation
from Steerables.AnomalyMetrics import ssim_metric, cw_ssim_metric, l2_metric
import albumentations as A

ae_patch_size = 128
cut_size = (0, 688, 0, 1024)
ae_stride = 8

def visualize_results (img, res, txt):
    f = plt.figure(figsize=(12, 4))

    f.add_subplot(1,2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title("Image")
    plt.imshow(img)

    f.add_subplot(1,2, 2)
    plt.xticks([])
    plt.yticks([])
    plt.title("Reconstructed")
    plt.imshow(res)

    f.text(.5, .05, txt, ha='center')
    plt.show()

def preprocess_data (patches):
    x = np.array(patches)
    x = x.astype('float64') / 255.
    x = x[..., tf.newaxis]
    return x

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



def validation ():
    valid_patches = load_patches_from_file_fixed('Dataset\\SEM_Data\\Anomalous\\images\\ITIA1103.tif', patch_size=ae_patch_size, 
        positions = ((194,315),(139,257),(119,484)))
    x_valid = preprocess_data(valid_patches)

    tf.keras.backend.set_floatx('float64')
    autoencoder = Model_noise_skip_small(input_shape=(ae_patch_size,ae_patch_size,1), latent_dim=500)
    autoencoder.load_weights('Weights\\cwssim_reg_loss\\check_epoch195.h5')

    decoded_imgs = autoencoder(x_valid).numpy() 

    print(tf.image.ssim(tf.constant(x_valid), tf.constant(decoded_imgs), 1.0))
    
    for i in range(len(decoded_imgs)):
        visualize_results (np.reshape(x_valid[i],(ae_patch_size,ae_patch_size)), np.reshape(decoded_imgs[i],(ae_patch_size,ae_patch_size)), "aa")

        #diff = ssim(np.reshape(x_valid[i],(patch_size,patch_size)), np.reshape(decoded_imgs[i],(patch_size,patch_size)), win_size=11, full=True)[1]
        ssim_configs = [17, 15, 13, 11, 9, 7, 5, 3]
        residual_ssim = np.zeros_like(np.squeeze(decoded_imgs[i]))
        for win_size in ssim_configs:
            residual_ssim += (1 - ssim(np.squeeze(x_valid[i]), np.squeeze(decoded_imgs[i]), win_size=win_size, full=True, data_range=1.)[1])
        residual_ssim = residual_ssim / len(ssim_configs)


        cwssim_configs = [9, 8, 7]
        residual_cwssim = np.expand_dims(np.zeros_like(decoded_imgs[i]), 0)
        metric_tf_7 = Metric_win (window_size=7, patch_size=ae_patch_size)
        for height in cwssim_configs:
            residual_cwssim += (1 - metric_tf_7.CWSSIM(np.expand_dims(x_valid[i], 0), np.expand_dims(decoded_imgs[i], 0), 
                        height=height, orientations=6, full=True).numpy()[0])
        residual_cwssim = residual_cwssim/len(cwssim_configs)
        residual_cwssim = np.squeeze(residual_cwssim)

        residual = (residual_cwssim + residual_ssim) / 2
        #diff[diff < 0.9] = 0
        
        visualize_results (np.reshape(x_valid[i],(ae_patch_size,ae_patch_size)), residual, "aa")
        plt.show()

def  validation_reco (n_img):
    print(n_img)
    valid_patches, valid_img = load_patches_from_file('Dataset\\SEM_Data\\Anomalous\\images\\ITIA11' + n_img + '.tif', patch_size=ae_patch_size, 
        random=False, stride=ae_stride, cut_size=cut_size) 
    valid_img = valid_img / 255

    autoencoder = Model_noise_skip(input_shape=(ae_patch_size,ae_patch_size,1), latent_dim=500)
    autoencoder.load_weights('Weights\\cwssim_loss\\check_epoch120.h5')
    #autoencoder.load_weights('Weights\\l2_loss\\check_epoch150.h5')

    
    #Patch-Wise reconstruction
    print (len(valid_patches))
    _, y_valid = batch_evaluation(valid_patches, autoencoder, 12)
    reconstruction = image_reconstruction(y_valid, "loss_type")
    visualize_results(valid_img, reconstruction, "Reco")

    #Patch cropping visualization
    start_index = (490, 417)
    valid_patch = valid_img [start_index[0]: start_index[0] + 128 , start_index[1]: start_index[1] + 128]
    reco_patch = reconstruction [start_index[0]: start_index[0] + 128 , start_index[1]: start_index[1] + 128]
    visualize_results(valid_patch, reco_patch, "Reco")


    

    #Get residual
    border_size = 5
    depr_mask = np.ones_like(valid_img) * 0.5
    depr_mask[border_size:valid_img.shape[0]-border_size, border_size:valid_img.shape[1]-border_size] = 1
    pad_size = int((1024 - cut_size[1])/2)
    padding = A.PadIfNeeded(1024, 1024, p=1.0)
    x_valid = padding(image=valid_img)['image']
    y_valid = padding(image=reconstruction)['image']
    
    residual = cw_ssim_metric (x_valid, y_valid, pad_size)
    #residual = ssim_metric (x_valid, y_valid, pad_size)
    #residual = l2_metric (x_valid, y_valid, pad_size)
    
    residual = residual * depr_mask
    visualize_results (valid_img, residual, "Residual")

    #Patch cropping visualization
    start_index = (490, 417)
    valid_patch = valid_img [start_index[0]: start_index[0] + 128 , start_index[1]: start_index[1] + 128]
    resi_patch = residual [start_index[0]: start_index[0] + 128 , start_index[1]: start_index[1] + 128]
    resi_patch[0,0] = 0.8
    visualize_results(valid_patch, resi_patch, "Reco")

if __name__ == "__main__":
    #validation()
    validation_reco("08")
