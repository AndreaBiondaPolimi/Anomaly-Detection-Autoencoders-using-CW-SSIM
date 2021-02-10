from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Reshape, ZeroPadding2D, Cropping2D, LeakyReLU, Layer, InputSpec
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Input, Add, Concatenate, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.models import Model
from tensorflow.keras.backend import int_shape

import tensorflow as tf
import numpy as np


def Model_ssim_skip(input_shape=(128,128,1), latent_dim=300):
    flc = 32
    input_img = Input(input_shape)

    h = Conv2D(flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(input_img)
    h = Conv2D(flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    #if cfg.patch_size==256:
        #h = Conv2D(cfg.flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(flc, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(flc*2, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(flc*2, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(flc*4, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(flc*2, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(flc, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)

    encoded = Conv2D(latent_dim, (8, 8), strides=1, activation='linear', padding='valid')(h)

    h = Conv2DTranspose(flc, (8, 8), strides=1, activation=LeakyReLU(alpha=0.2), padding='valid')(encoded)
    h = Conv2D(flc*2, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(flc*4, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2DTranspose(flc*2, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(flc*2, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2DTranspose(flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(flc, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2DTranspose(flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    #if cfg.patch_size==256:
        #h = Conv2DTranspose(cfg.flc, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)

    decoded = Conv2DTranspose(1, (4, 4), strides=2, activation='sigmoid', padding='same')(h)

    return Model(input_img, decoded)


# add batch norm / reduce filter size
def Model_noise_skip(input_shape=(128,128,1), latent_dim=300):
    input = Input(input_shape)

    X_skip_1 = Conv2D(32, (4, 4),activation=LeakyReLU(),  padding='same' , strides=2, kernel_initializer=glorot_normal(seed=None), name="Conv1")(input) 
    X_skip_2 = Conv2D(64, (4, 4),activation=LeakyReLU() , padding='same',strides=2, kernel_initializer=glorot_normal(seed=None), name="Conv2")(X_skip_1)
    X_skip_3 = Conv2D(128, (4, 4),activation=LeakyReLU() , padding='same',strides=2, kernel_initializer=glorot_normal(seed=None), name="Conv3")(X_skip_2)
    X_skip_4 = Conv2D(256, (4, 4),activation=LeakyReLU() , padding='same',strides=4, kernel_initializer=glorot_normal(seed=None), name="Conv4")(X_skip_3)
    
    X_lat = Conv2D(512, (4, 4),activation=LeakyReLU() , padding='same',strides=4, kernel_initializer=glorot_normal(seed=None), name="Conv_lat")(X_skip_4)

    X = Conv2DTranspose(256, (4, 4),activation=LeakyReLU() , padding='same',strides=4, kernel_initializer=glorot_normal(seed=None), name="ConvT4")(X_lat)
    X = Conv2DTranspose(128, (4, 4),activation=LeakyReLU() , padding='same',strides=4, kernel_initializer=glorot_normal(seed=None), name="ConvT3")(X)
    X = Conv2DTranspose(64, (4, 4),activation=LeakyReLU() , padding='same',strides=2, kernel_initializer=glorot_normal(seed=None), name="ConvT2")(X)
    X = Conv2DTranspose(32, (4, 4),activation=LeakyReLU() , padding='same',strides=2, kernel_initializer=glorot_normal(seed=None), name="ConvT1")(X)
    X = Conv2DTranspose(1, (4, 4),activation='linear',  padding='same' ,strides=2, kernel_initializer=glorot_normal(seed=None), name="ConvT0")(X)
    
    return Model(inputs=input, outputs=X)


def Model_noise_skip_wide(input_shape=(128,128,1), latent_dim=300):
    input = Input(input_shape)

    X_skip_1 = Conv2D(16, (4, 4),activation=LeakyReLU(),  padding='same' , strides=2, kernel_initializer=glorot_normal(seed=None), name="Conv1")(input) 
    X_skip_3 = Conv2D(32, (4, 4),activation=LeakyReLU() , padding='same',strides=2, kernel_initializer=glorot_normal(seed=None), name="Conv3")(X_skip_1)
    X_skip_4 = Conv2D(64, (4, 4),activation=LeakyReLU() , padding='same',strides=2, kernel_initializer=glorot_normal(seed=None), name="Conv4")(X_skip_3)
    
    X_lat = Conv2D(128, (4, 4),activation=LeakyReLU() , padding='same',strides=2, kernel_initializer=glorot_normal(seed=None), name="Conv_lat", kernel_regularizer=l2(1e-3))(X_skip_4)

    X = Conv2DTranspose(64, (4, 4),activation=LeakyReLU() , padding='same',strides=2, kernel_initializer=glorot_normal(seed=None), name="ConvT4")(X_lat)
    X = Conv2DTranspose(32, (4, 4),activation=LeakyReLU() , padding='same',strides=2, kernel_initializer=glorot_normal(seed=None), name="ConvT3")(X)
    X = Conv2DTranspose(16, (4, 4),activation=LeakyReLU() , padding='same',strides=2, kernel_initializer=glorot_normal(seed=None), name="ConvT2")(X)
    X = Conv2DTranspose(1, (4, 4),activation='linear',  padding='same' ,strides=2, kernel_initializer=glorot_normal(seed=None), name="ConvT0")(X)
    
    return Model(inputs=input, outputs=X)


def Model_noise_skip_small(input_shape=(128,128,1), latent_dim=300):
    input = Input(input_shape)

    X_skip_1 = Conv2D(32, (4, 4),activation=LeakyReLU(),  padding='same' , strides=2, kernel_initializer=glorot_normal(seed=None), name="Conv1")(input) 
    X_skip_2 = Conv2D(64, (4, 4),activation=LeakyReLU() , padding='same',strides=2, kernel_initializer=glorot_normal(seed=None), name="Conv2")(X_skip_1)
    X_skip_3 = Conv2D(128, (4, 4),activation=LeakyReLU() , padding='same',strides=2, kernel_initializer=glorot_normal(seed=None), name="Conv3")(X_skip_2)
    X_skip_4 = Conv2D(256, (4, 4),activation=LeakyReLU() , padding='same',strides=2, kernel_initializer=glorot_normal(seed=None), name="Conv4")(X_skip_3)
    
    X_lat = Conv2D(512, (4, 4),activation=LeakyReLU() , padding='same',strides=2, kernel_initializer=glorot_normal(seed=None), name="Conv_lat", 
                            kernel_regularizer=l2(1e-3))(X_skip_4)

    X = Conv2DTranspose(256, (4, 4),activation=LeakyReLU() , padding='same',strides=2, kernel_initializer=glorot_normal(seed=None), name="ConvT4")(X_lat)
    X = Conv2DTranspose(128, (4, 4),activation=LeakyReLU() , padding='same',strides=2, kernel_initializer=glorot_normal(seed=None), name="ConvT3")(X)
    X = Conv2DTranspose(64, (4, 4),activation=LeakyReLU() , padding='same',strides=2, kernel_initializer=glorot_normal(seed=None), name="ConvT2")(X)
    X = Conv2DTranspose(32, (4, 4),activation=LeakyReLU() , padding='same',strides=2, kernel_initializer=glorot_normal(seed=None), name="ConvT1")(X)
    X = Conv2DTranspose(1, (4, 4),activation='linear',  padding='same' ,strides=2, kernel_initializer=glorot_normal(seed=None), name="ConvT0")(X)
    
    return Model(inputs=input, outputs=X)




if __name__ == "__main__":
    autoencoder = Model_ssim_skip(input_shape=(128,128,1), latent_dim=500)
    autoencoder.summary()