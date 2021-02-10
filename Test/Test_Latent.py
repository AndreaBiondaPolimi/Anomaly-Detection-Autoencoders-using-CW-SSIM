import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from AutoencoderModels import Model_ssim_skip, Model_noise_skip, Model_noise_skip_small, Model_noise_skip_wide
from DataLoader import load_patches, load_patches_from_file_fixed, load_gt_from_file, load_patches_from_file, check_preprocessing
from Steerables.metrics_TF import Metric_win
from skimage.metrics import structural_similarity as ssim
import albumentations as A
from Utils import batch_evaluation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

training_size = 128

validation_size = 128
stride = 32

cut_size = (0, 688, 0, 1024)
preprocess_limit = 0

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

def image_reconstruction (y_valid, valid_flags, mean_score):
    reconstrunction = np.zeros((cut_size[1]-cut_size[0], cut_size[3]-cut_size[2]))
    normalizator = np.zeros((cut_size[1]-cut_size[0], cut_size[3]-cut_size[2]))
    
    i=0; j=0
    for idx in range (len(y_valid)):
        if (valid_flags[idx]):
            #reconstrunction [j:j+validation_size, i:i+validation_size] += y_valid[idx]
            reconstrunction [j:j+validation_size, i:i+validation_size] += np.full((validation_size,validation_size), y_valid[idx])
        else:
            reconstrunction [j:j+validation_size, i:i+validation_size] += mean_score
        
        normalizator [j:j+validation_size, i:i+validation_size] += np.ones((validation_size,validation_size))
        if (i+validation_size < cut_size[3]-cut_size[2]):
            i=i+stride
        else:
            i=0; j=j+stride
    reconstrunction =  reconstrunction/normalizator

    return reconstrunction


tf.keras.backend.set_floatx('float64')



def validation ():
    n_img = "08"
    weights = 'Weights\\cwssim_reg_loss\\check_epoch200.h5'
    

    ###TRAIN###
    train_patches = load_patches('Dataset\\SEM_Data\\Normal', patch_size=training_size, n_patches=500, random=True, preprocess_limit=preprocess_limit)
    split = int(len(train_patches) * 0.8)

    autoencoder = Model_noise_skip_wide(input_shape=(training_size,training_size,1), latent_dim=500)
    #autoencoder = Model_noise_skip(input_shape=(training_size,training_size,1), latent_dim=500)
    autoencoder.load_weights(weights)
    extractor = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer("Conv_lat").output)

    train = train_patches[:split]
    _, x_latent = batch_evaluation(train, extractor, 10)

    if (len(np.shape(x_latent)) > 2):
        x_latent = x_latent.reshape((x_latent.shape[0] * x_latent.shape[1] * x_latent.shape[2]),  x_latent.shape[3] )
    
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(x_latent)
    #gm = GaussianMixture(n_components=2, random_state=450, n_init=5).fit(x_latent)
    #clf = IsolationForest(random_state=0).fit(x_latent)
    #lof = LocalOutlierFactor(n_neighbors=25, novelty=True).fit(x_latent)

    valid = train_patches[split:]
    _, x_latent = batch_evaluation(valid, extractor, 10)

    if (len(np.shape(x_latent)) > 2):
        x_latent = x_latent.reshape((x_latent.shape[0] * x_latent.shape[1] * x_latent.shape[2]),  x_latent.shape[3] )

    scores = kde.score_samples(x_latent)
    mean_score = np.mean(scores)

    print ("Training shape", np.shape(x_latent))

    ###VALID###
    valid_patches, valid_img = load_patches_from_file('Dataset\\SEM_Data\\Anomalous\\images\\ITIA11' + n_img + '.tif', patch_size=validation_size, stride=stride,
        random=False, cut_size=cut_size, preprocess_limit=preprocess_limit) 

    flags = [None] * len(valid_patches)
    #Damn flags
    for i in range(len(valid_patches)):
        flags[i] = check_preprocessing(valid_patches[i], preprocess_limit=preprocess_limit)

    autoencoder = Model_noise_skip_wide(input_shape=(training_size,training_size,1), latent_dim=500)
    #autoencoder = Model_noise_skip(input_shape=(training_size,training_size,1), latent_dim=500)
    autoencoder.load_weights(weights)
    extractor = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer("Conv_lat").output)


    #_, y_valid = batch_evaluation(valid_patches, autoencoder, 10)
    _, y_latent = batch_evaluation(valid_patches, extractor, 10)

    print ("Latent shape", np.shape(y_latent))

    if (len(np.shape(y_latent)) > 2):
        scores = []
        for i in range (len(y_latent)):
            aa = y_latent[i].reshape((y_latent[i].shape[0] * y_latent[i].shape[1], y_latent[i].shape[2]) )
            
            kde_scores = kde.score_samples(aa)
            #kde_scores = gm.predict(y_latent)
            #kde_scores = clf.predict(y_latent)
            #kde_scores = lof.predict(y_latent)
            kde_scores = kde_scores.reshape(y_latent[i].shape[0] , y_latent[i].shape[1])

            scores.append(kde_scores)
    else:
        scores = kde.score_samples(y_latent)
        

    scores = np.array(scores)
    print ("Scores shape", np.shape(scores))    
    
    upsampling = []
    for i in range (len(scores)):
        upsampled = tf.image.resize(scores[i][..., tf.newaxis], size=[validation_size,validation_size], method='lanczos5', antialias=True)
        upsampling.append (np.squeeze(upsampled))

    print ("Scores upsampled shape", np.shape(upsampling))    

    reconstruction = image_reconstruction (upsampling, flags, mean_score)
    visualize_results (valid_img, reconstruction, "")

if __name__ == "__main__":
    validation()




"""
layer_names = [layer.name for layer in autoencoder.layers]
layer_idx = layer_names.index("ConvT4")
input_shape = autoencoder.layers[layer_idx].get_input_shape_at(0)[1:]
layer_input = Input(shape=input_shape)   
x = layer_input
for layer in autoencoder.layers[layer_idx:]:
    x = layer(x)


reconstructor = Model(inputs=layer_input, outputs=x)
reconstruction = reconstructor(latent_representation).numpy()

reconstruction = np.squeeze(reconstruction)
plt.imshow(reconstruction)
plt.show()
"""