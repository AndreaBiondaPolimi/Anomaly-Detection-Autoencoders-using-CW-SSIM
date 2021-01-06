import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from AutoencoderModels import Model_ssim_skip, Model_noise_skip,  Model_noise_mod
from DataLoader import load_patches, load_patches_from_file_fixed
from Steerables.metrics_TF import Metric, Metric_win
from skimage.metrics import structural_similarity as ssim

def visualize_results (img, res, txt):
    f = plt.figure(figsize=(17, 7))

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


patch_size = 128

def validation ():
    #valid_patches = load_patches_from_file_fixed('Dataset\\SEM_Data\\Anomalous\\images\\ITIA1107.tif', patch_size=patch_size, 
        #positions = ((84,174),(251,509),(387,861),(563,520),(215,871))) 
    
    #valid_patches = load_patches_from_file_fixed('Dataset\\SEM_Data\\Anomalous\\images\\ITIA1103.tif', patch_size=patch_size, 
        #positions = ((35,515),(254,447),(174,822),(401,440)))

    valid_patches = load_patches_from_file_fixed('Dataset\\SEM_Data\\Anomalous\\images\\ITIA1113.tif', patch_size=patch_size, 
        positions = ((568,452),(338,262), (163,763)))

    x_valid = preprocess_data(valid_patches)

    tf.keras.backend.set_floatx('float64')
    autoencoder = Model_noise_skip(input_shape=(patch_size,patch_size,1), latent_dim=500)
    autoencoder.load_weights('Weights\\cwssim_loss\\check_epoch130.h5')
    #autoencoder.load_weights('Weights\\stsim1_loss\\check_epoch100.h5')
    decoded_imgs = autoencoder(x_valid).numpy() 

    #print(tf.image.ssim(tf.constant(x_valid), tf.constant(decoded_imgs), 1.0))
    
    metric_tf = Metric_win (window_size=7, patch_size=patch_size)

    for i in range(len(decoded_imgs)):
        visualize_results (np.reshape(x_valid[i],(patch_size,patch_size)), np.reshape(decoded_imgs[i],(patch_size,patch_size)), "aa")
        #diff = np.square(np.reshape(x_valid[i],(patch_size,patch_size)) - np.reshape(decoded_imgs[i],(patch_size,patch_size)))
        #diff = np.abs(np.reshape(x_valid[i],(patch_size,patch_size)) - np.reshape(-1 * decoded_imgs[i],(patch_size,patch_size)))
        #diff = ssim(np.reshape(x_valid[i],(patch_size,patch_size)), np.reshape(decoded_imgs[i],(patch_size,patch_size)), win_size=9, full=True)[1]
        #diff = np.reshape(x_valid[i],(patch_size,patch_size)) - np.reshape(decoded_imgs[i],(patch_size,patch_size))
        #diff = np.reshape(decoded_imgs[i],(patch_size,patch_size)) - np.reshape(x_valid[i],(patch_size,patch_size))
            
        diff = metric_tf.CWSSIM(x_valid, decoded_imgs, height=5, orientations=5, full=True).numpy()[0]
        plt.imshow(np.squeeze(diff))
        plt.show()

        diff = 1. - diff

        plt.imshow(diff)
        plt.show()

        diff[diff < 0.7] = 0
        diff[diff >= 0.7] = 1
        visualize_results(np.reshape(x_valid[i],(patch_size,patch_size)), np.reshape(diff,(patch_size,patch_size)), "aa")




if __name__ == "__main__":
    validation()