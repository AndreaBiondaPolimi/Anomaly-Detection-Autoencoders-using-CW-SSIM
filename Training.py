import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from Steerables.SCFpyr_TF import SCFpyr_TF
import Steerables.utils as utils
from DataLoader import load_patches, load_patches_from_file_fixed
import matplotlib.pyplot as plt
from Steerables.metrics_TF import Metric_win

from Utils import preprocess_data, visualize_results
from AutoencoderModels import Model_ssim_skip, Model_noise_skip, Model_noise_mod

latent_dim = 500
lr = 1e-3
decay_fac = 0.5
decay_step = 20
epoch = 100
batch_size = 4
patch_size = 128

def scheduler(epoch):
    return lr * decay_fac ** (np.floor(epoch / decay_step))

def stsim1_loss(y_true, y_pred):
    metric_tf = Metric_win (patch_size)
    stsim_scores_tf = metric_tf.STSIM_1(y_pred, y_true, height=5, orientations=5)   
    loss = tf.math.reduce_mean(1. - stsim_scores_tf) 
    return loss #+ tf.keras.losses.MSE(y_true, y_pred)

def stsim2_loss(y_true, y_pred):
    metric_tf = Metric_win (patch_size)
    stsim_scores_tf = metric_tf.STSIM_2(y_pred, y_true, height=5, orientations=5)  
    loss = tf.math.reduce_mean(1. - stsim_scores_tf) 
    return loss #+ tf.keras.losses.MSE(y_true, y_pred)

def cwssim_loss(y_true, y_pred):
    metric_tf = Metric_win (patch_size, window_size=7)
    stsim_scores_tf = metric_tf.CWSSIM(y_pred, y_true, height=5, orientations=5)  
    loss = tf.math.reduce_mean(1. - stsim_scores_tf) 
    return loss #+ tf.keras.losses.MSE(y_true, y_pred)

def ssim_loss (y_true, y_pred):
    return tf.reduce_mean (1. - tf.image.ssim(y_true, y_pred, 1.0))

def ms_ssim_loss (y_true, y_pred):
    return tf.reduce_mean (1. - tf.image.ssim_multiscale(y_true, y_pred, 1.0))
    
def l2_loss (y_true, y_pred):
    return tf.keras.losses.MSE(y_true, y_pred)


if __name__ == "__main__":
    train_patches = load_patches('Dataset\\SEM_Data\\Normal', patch_size=patch_size, n_patches=100, random=True, preprocess_limit=0, resize=0.8)
    x_train = preprocess_data(train_patches)
    print (np.shape(x_train))

    tf.keras.backend.set_floatx('float64')
    
    for loss in [cwssim_loss, ]:
        print (loss.__name__)

        callbacks = []
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath='Weights\\' + loss.__name__ + '\\check_epoch{epoch:02d}.h5', 
                                    save_weights_only=True, period=5))

        autoencoder = Model_noise_skip(input_shape=(patch_size,patch_size,1), latent_dim=latent_dim)
        autoencoder.summary()

        autoencoder.compile(optimizer='adam', 
                            loss=loss)

        autoencoder.fit(x_train, x_train,
                    epochs=epoch,
                    shuffle=True,
                    batch_size=batch_size,
                    callbacks=callbacks)



    


