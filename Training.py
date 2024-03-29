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
from AutoencoderModels import Model_ssim_skip, Model_noise_skip, Model_noise_skip_small, Model_noise_skip_wide, Model_noise_skip_01

import argparse
import configparser
import os

dataset = 'MVTec_Data'
category = 'wood'

loss_type = 'cwssim_loss'
window_size = 7
scales = 5
orients = 5


n_patches = 25
lr = 1e-3
decay_fac = 0.5
decay_step = 20
epoch = 200
batch_size = 20
patch_size = 256
save_period = 5

def scheduler(epoch):
    return lr * decay_fac ** (np.floor(epoch / decay_step))

def cwssim_loss(y_true, y_pred):
    metric_tf = Metric_win (patch_size, window_size=window_size)
    stsim_scores_tf = metric_tf.CWSSIM(y_pred, y_true, height=scales, orientations=orients)  
    loss = tf.math.reduce_mean(1. - stsim_scores_tf) 
    return loss #+ tf.keras.losses.MSE(y_true, y_pred)

def ssim_loss (y_true, y_pred):
    return tf.reduce_mean (1. - tf.image.ssim(y_true, y_pred, 1.0))

def ms_ssim_loss (y_true, y_pred):
    return tf.reduce_mean (1. - tf.image.ssim_multiscale(y_true, y_pred, 1.0))
    
def l2_loss (y_true, y_pred):
    return tf.keras.losses.MSE(y_true, y_pred)


def train ():
    current_folder = os.getcwd()
    train_patches = load_patches(os.path.join('Dataset',dataset,category,'Normal'), patch_size=patch_size, n_patches=n_patches, random=True, preprocess_limit=0, resize=None)
    x_train = preprocess_data(train_patches)
    print (np.shape(x_train))

    #for x in x_train:
        #plt.imshow(np.squeeze(x))
        #plt.show()

    tf.keras.backend.set_floatx('float64')
    
    loss_function = None
    for loss in [cwssim_loss, ssim_loss, ms_ssim_loss, l2_loss]:
        if (loss.__name__ == loss_type):
            loss_function = loss

    callbacks = []
    callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('Weights','new_weights','check_epoch{epoch:02d}.h5'), save_weights_only=True, period=save_period))

    autoencoder = Model_noise_skip(input_shape=(patch_size,patch_size,1))
    autoencoder.summary()
    #autoencoder.load_weights('Weights\\new_weights\\check_epoch95.h5')

    autoencoder.compile(optimizer='adam', loss=loss_function)

    #autoencoder.fit(x_train, x_train, epochs=epoch, shuffle=True, batch_size=batch_size, callbacks=callbacks, initial_epoch=95)
    autoencoder.fit(x_train, x_train, epochs=epoch, shuffle=True, batch_size=batch_size, callbacks=callbacks)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action="store", help="dataset name", dest="dataset", default='SEM_Data')
    parser.add_argument('-c', action="store", help="category name", dest="category", default='Nanofibrous')
    parser.add_argument('-n', action="store", help="number of patches", dest="n_patches", default=25)
    parser.add_argument('-t', action="store", help="loss_type", dest="loss_type", default='cwssim_loss')
    parser.add_argument('-e', action="store", help="number of epochs", dest="epochs", default=200)
    parser.add_argument('-b', action="store", help="batch size", dest="batch_size", default=20)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    dataset = args.dataset
    category = args.category
    
    loss_type = args.loss_type
    n_patches = args.n_patches
    epoch = args.epochs
    batch_size = args.batch_size  

    train()

