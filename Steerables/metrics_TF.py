import numpy as np
import tensorflow as tf
import itertools

from tensorflow.keras.backend import int_shape
from Steerables.SCFpyr_TF import SCFpyr_TF, SCFpyr_TF_nosub

class Metric_win():
    def __init__(self, patch_size=128, window_size=7):
        self.patch_size = patch_size
        self.win = window_size

    def conv(self, a, b, is_complex, padding="VALID"):
        a = tf.expand_dims(a, axis=3)
        #return signal.correlate2d(a, b, mode = 'valid')
        if (is_complex):
            real = tf.math.real(a)
            imag = tf.math.imag (a)
            real_conv = tf.nn.conv2d(real, b, strides=[1,1,1,1], padding=padding)
            imag_conv = tf.nn.conv2d(imag, b, strides=[1,1,1,1], padding=padding)
            return tf.complex(real_conv, imag_conv)
        return tf.nn.conv2d(a, b, strides=[1,1,1,1], padding=padding)

    def _tf_fspecial_gauss(self, win = 11, sigma = 1.5):
        """Function to mimic the 'fspecial' gaussian MATLAB function
        """
        x_data, y_data = np.mgrid[-win//2 + 1:win//2 + 1, -win//2 + 1:win//2 + 1]

        x_data = np.expand_dims(x_data, axis=-1)
        x_data = np.expand_dims(x_data, axis=-1)

        y_data = np.expand_dims(y_data, axis=-1)
        y_data = np.expand_dims(y_data, axis=-1)

        x = tf.constant(x_data, dtype=tf.float64)
        y = tf.constant(y_data, dtype=tf.float64)

        g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g / tf.reduce_sum(g)


        ### STSIM-1 Section ###
    
    
    ### STSIM 1 ###
    def STSIM_1 (self, imgs_batch_1, imgs_batch_2, height, orientations):
        score = self.STSIM_1_score (imgs_batch_1, imgs_batch_2, height, orientations)

        return tf.math.reduce_mean(tf.stack([f for f in score], axis=1), axis=1)

    def STSIM_1_score (self, imgs_batch_1, imgs_batch_2, height, orientations):
        if (not tf.is_tensor(imgs_batch_1)):
            imgs_batch_1 = tf.convert_to_tensor(imgs_batch_1, tf.complex128)
        if (not tf.is_tensor(imgs_batch_2)):
            imgs_batch_2 = tf.convert_to_tensor(imgs_batch_2, tf.complex128)

        pyr = SCFpyr_TF(
            height= height, 
            nbands= orientations,
            scale_factor= 2,
            precision = 64,
            patch_size=self.patch_size
        )
        coeff_1 = pyr.build(imgs_batch_1)
        coeff_2 = pyr.build(imgs_batch_2)

        score = []

        score.append(self.pooling(coeff_1[0], coeff_2[0], False))
        for i in range(len(coeff_1[1:-1])):
            for j in range(len(coeff_1[i+1])):
                score.append(self.pooling(coeff_1[i+1][j], coeff_2[i+1][j], True))
        score.append(self.pooling(coeff_1[-1], coeff_2[-1], False))   

        return score

    def pooling (self, s1, s2, is_complex):    
        tmp = tf.math.pow(self.compute_L_term(s1, s2, is_complex) * self.compute_C_term(s1, s2, is_complex) * \
                            self.compute_Cx_term(s1, s2, is_complex, 1) * self.compute_Cx_term(s1, s2, is_complex, 2), 0.25)

        tmp = tf.math.reduce_mean(tmp, axis = (1,2,3))
        print (tmp.numpy())
        return tmp

    def compute_L_term(self, s1, s2, is_complex):
        C = 0.001
        win = self.win
        window = (tf.constant(np.ones((win,win,1,1))/(win*win))) # average filter

        mu1 = tf.abs(self.conv(s1, window, is_complex))
        mu2 = tf.abs(self.conv(s2, window, is_complex))

        Lmap = (2 * mu1 * mu2 + C)/(mu1*mu1 + mu2*mu2 + C)
        return Lmap  
    
    def compute_C_term(self, s1, s2, is_complex):
        C = 0.001
        win = self.win
        window = (tf.constant(np.ones((win,win,1,1))/(win*win))) # average filter

        mu1 = tf.abs(self.conv(s1, window, is_complex))
        mu2 = tf.abs(self.conv(s2, window, is_complex))

        sigma1_sq = self.conv(tf.abs(s1)*tf.abs(s1), window, False) - mu1 * mu1
        sigma2_sq = self.conv(tf.abs(s2)*tf.abs(s2), window, False) - mu2 * mu2

        sigma1 = tf.math.sqrt(sigma1_sq)
        sigma2 = tf.math.sqrt(sigma2_sq)

        Cmap = (2*sigma1*sigma2 + C)/(sigma1_sq + sigma2_sq + C)
        return Cmap

    def compute_Cx_term(self, im1, im2, is_complex, shift_axes):
        C = 0.001
        win = self.win
        window = (tf.constant(np.ones((win,win,1,1))/(win*win))) # average filter

        im11 = tf.roll(im1, shift=1, axis=shift_axes) 
        im21 = tf.roll(im2, shift=1, axis=shift_axes) 

        #X=im1 (o im2), Y=im11 (o im21)
        mu1 = self.conv(im1, window, is_complex)
        mu2 = self.conv(im2, window, is_complex)
        mu11 = self.conv(im11, window, is_complex)
        mu21 = self.conv(im21, window, is_complex)

        #Sx = E(X*X) - E(x)*E(X)
        sigma1_sq = self.conv(tf.abs(im1)*tf.abs(im1), window, False) - tf.abs(mu1)*tf.abs(mu1)
        sigma11_sq = self.conv(tf.abs(im11)*tf.abs(im11), window, False) - tf.abs(mu11)*tf.abs(mu11)
        sigma2_sq = self.conv(tf.abs(im2)*tf.abs(im2), window, False) - tf.abs(mu2)*tf.abs(mu2)
        sigma21_sq = self.conv(tf.abs(im21)*tf.abs(im21), window, False) - tf.abs(mu21)*tf.abs(mu21)

        #COV(X,Y) = E(X*Yc) - E(X)*E(Yc)
        sigma1_cross = self.complex_sub(self.conv(self.complex_multiply(im1, self.complex_conj(im11)), window, True), self.complex_multiply(mu1, self.complex_conj(mu11)))
        sigma2_cross = self.complex_sub(self.conv(self.complex_multiply(im2, self.complex_conj(im21)), window, True), self.complex_multiply(mu2, self.complex_conj(mu21)))

        #P(X,Y) = COV(X,Y) / sqrt(Sx)*sqrt(Sy)
        rho1 = self.complex_div(sigma1_cross + C , tf.math.sqrt(sigma1_sq)*tf.math.sqrt(sigma11_sq) + C)
        rho2 = self.complex_div(sigma2_cross + C , tf.math.sqrt(sigma2_sq)*tf.math.sqrt(sigma21_sq) + C)
        
        Cxmap = 1 - 0.5*tf.abs(rho1 - rho2)
        return Cxmap


    ### STSIM 2 ###
    def STSIM_2 (self, imgs_batch_1, imgs_batch_2, height, orientations):
        score = self.STSIM_1_score (imgs_batch_1, imgs_batch_2, height, orientations)

        pyr = SCFpyr_TF_nosub(
            height= height, 
            nbands= orientations,
            scale_factor= 2,
            precision = 64,
            patch_size=self.patch_size
        )
        coeff_1 = pyr.build(imgs_batch_1)
        coeff_2 = pyr.build(imgs_batch_2)

        # Accross scale, same orientation
        for scale in range(2, len(coeff_1) - 1):
            for orient in range(len(coeff_1[1])):
                im11 = tf.abs(coeff_1[scale - 1][orient])
                im12 = tf.abs(coeff_1[scale][orient])
                im21 = tf.abs(coeff_2[scale - 1][orient])
                im22 = tf.abs(coeff_2[scale][orient])
                score.append(self.compute_cross_term(im11, im12, im21, im22))
        
        # Accross orientation, same scale
        Nor = len(coeff_1[1])
        for scale in range(1, len(coeff_1) - 1):
            for orient in range(Nor - 1):
                im11 = tf.abs(coeff_1[scale][orient])
                im21 = tf.abs(coeff_2[scale][orient])  
                for orient2 in range(orient + 1, Nor):
                    im13 = tf.abs(coeff_1[scale][orient2])
                    im23 = tf.abs(coeff_2[scale][orient2])
                    score.append(self.compute_cross_term(im11, im13, im21, im23))
            
        return tf.math.reduce_mean(tf.stack([f for f in score], axis=1), axis=1)

    def compute_cross_term(self, im11, im12, im21, im22):
        C = 0.001
        win = self.win
        window2 = 1/(win**2)*np.ones((win, win))
        window2 = tf.expand_dims(window2, axis=2)
        window2 = tf.expand_dims(window2, axis=3)

        mu11 = self.conv(im11, window2, True)
        mu12 = self.conv(im12, window2, True)
        mu21 = self.conv(im21, window2, True)
        mu22 = self.conv(im22, window2, True)

        sigma11_sq = self.complex_sub(self.conv(self.complex_multiply(im11,im11), window2, True) , self.complex_multiply(mu11,mu11))
        sigma12_sq = self.complex_sub(self.conv(self.complex_multiply(im12,im12), window2, True) , self.complex_multiply(mu12,mu12))
        sigma21_sq = self.complex_sub(self.conv(self.complex_multiply(im21,im21), window2, True) , self.complex_multiply(mu21,mu21))
        sigma22_sq = self.complex_sub(self.conv(self.complex_multiply(im22,im22), window2, True) , self.complex_multiply(mu22,mu22))
        
        sigma1_cross = self.complex_sub(self.conv(self.complex_multiply(im11,im12), window2, True) , self.complex_multiply(mu11,mu12))
        sigma2_cross = self.complex_sub(self.conv(self.complex_multiply(im21,im22), window2, True) , self.complex_multiply(mu21,mu22))

        rho1 = self.complex_div(sigma1_cross + C , tf.math.sqrt(sigma11_sq)*tf.math.sqrt(sigma12_sq) + C)
        rho2 = self.complex_div(sigma2_cross + C , tf.math.sqrt(sigma21_sq)*tf.math.sqrt(sigma22_sq) + C)

        Crossmap = 1 - 0.5*tf.abs(rho1 - rho2)
        return tf.math.reduce_mean(Crossmap, axis = (1,2,3))


    ### CW SSIM ###
    def CWSSIM (self, imgs_batch_1, imgs_batch_2, height, orientations, full=False):
        pyr = SCFpyr_TF(
            height= height, 
            nbands= orientations,
            scale_factor= 2,
            precision = 64,
            patch_size=self.patch_size
        )
        coeff_1 = pyr.build(imgs_batch_1)
        coeff_2 = pyr.build(imgs_batch_2)

        score = []

        score.append(self.cw_ssim_score(coeff_1[0], coeff_2[0], False, full))
        for i in range(len(coeff_1[1:-1])):
            for j in range(len(coeff_1[i+1])):
                score.append(self.cw_ssim_score(coeff_1[i+1][j], coeff_2[i+1][j], True, full))
        score.append(self.cw_ssim_score(coeff_1[-1], coeff_2[-1], False, full))

        return tf.math.reduce_mean(tf.stack([f for f in score], axis=1), axis=1)

    def cw_ssim_score (self, s1, s2, is_complex, full=False):
        C = 0.001
        win = self.win
        padding = "VALID" if full==False else "SAME"
        
        #window = (tf.constant(np.ones((win,win,1,1))/(win*win))) # average filter
        window = (tf.constant(np.ones((win,win,1,1)))) # sum filter

        num = self.complex_multiply(s1, self.complex_conj(s2))
        num = self.conv(num, window, True, padding) 
        
        den = tf.abs(s1)**2 + tf.abs(s2)**2
        den = self.conv(den, window, False, padding) 
        
        cwssim = (2*tf.abs(num) + C) / (den + C) 
        
        if (not full):
            cwssim = tf.math.reduce_mean(cwssim, axis = (1,2))
        else:
            if (int_shape(cwssim)[1] < self.patch_size):
                cwssim = tf.image.resize(cwssim, [self.patch_size,self.patch_size], method='nearest')
    
        return cwssim


    ### GMSD ###
    def GMSD (self, imgs_batch_1, imgs_batch_2, win=3):
        C = 0.0001
        dx = tf.expand_dims(tf.expand_dims(tf.constant([[1/3,0,-1/3],[1/3,0,-1/3],[1/3,0,-1/3]], tf.float64), 2) , 3)
        dy = tf.expand_dims(tf.expand_dims(tf.constant([[1/3,1/3,1/3],[0,0,0],[-1/3,-1/3,-1/3]], tf.float64), 2) , 3)
        avg = tf.expand_dims(tf.expand_dims(tf.constant(np.ones((2,2))/4, tf.float64), 2) , 3)

        x1 = tf.nn.conv2d(imgs_batch_1, avg, strides=2, padding='VALID')
        x2 = tf.nn.conv2d(imgs_batch_2, avg, strides=2, padding='VALID')

        m_x1_dx = tf.nn.conv2d(x1, dx, strides=1, padding='SAME')
        m_x1_dy = tf.nn.conv2d(x1, dy, strides=1, padding='SAME')
        m_x2_dx = tf.nn.conv2d(x2, dx, strides=1, padding='SAME')
        m_x2_dy = tf.nn.conv2d(x2, dy, strides=1, padding='SAME')

        m_x1 = tf.sqrt(m_x1_dx**2 + m_x1_dy**2)
        m_x2 = tf.sqrt(m_x2_dx**2 + m_x2_dy**2)

        gms = (2*m_x1*m_x2 + C) / (m_x1**2 + m_x2**2 + C)
        gmsd = tf.math.reduce_std(gms, axis=(1,2))
        return gmsd


    ### UTILS ###
    def complex_multiply (self, s1, s2):
        real = tf.math.real(s1) * tf.math.real(s2) - tf.math.imag(s1) * tf.math.imag(s2)
        imag = tf.math.real(s1) * tf.math.imag(s2) + tf.math.imag(s1) * tf.math.real(s2)
        return tf.complex (real, imag)

    def complex_conj (self, s1):
        return tf.complex(tf.math.real(s1), - tf.math.imag(s1))

    def complex_sub (self, s1, s2):
        return tf.complex(tf.math.real(s1) - tf.math.real(s2), tf.math.imag(s1) - tf.math.imag(s2))

    def complex_sum (self, s1, s2):
        return tf.complex(tf.math.real(s1) + tf.math.real(s2), tf.math.imag(s1) + tf.math.imag(s2))

    def complex_div (self, s1, s2):
        num = self.complex_multiply(s1, self.complex_conj(s2))
        den = tf.math.real(s2)**2 + tf.math.imag(s2)**2
        return tf.complex(tf.math.real(num) / den, tf.math.imag(num) / den)




#Future implementation of STSIM1 and 2 globally
"""
class Metric(Metric_win):
    def __init__(self, patch_size=128):
        self.patch_size = patch_size
    
    ### STSIM-M ###
    def STSIM_M (self, imgs_batch, height, orientations):
        im_batch_numpy = np.expand_dims(imgs_batch, axis=-1)
        im_batch_tf = tf.convert_to_tensor(im_batch_numpy, tf.complex128)

        pyr = SCFpyr_TF(
            height= height, 
            nbands= orientations,
            scale_factor= 2,
            precision = 64,
            patch_size=self.patch_size
        )
        coeff = pyr.build(imgs_batch)


        features = []

        # Extract mean / var / H_autocorr/ V_autocorr from every subband
        features = self.extract_basic(coeff[0], False, features)
        for orients in coeff[1:-1]:
            for band in orients:
                features = self.extract_basic(band, True, features)
        features = self.extract_basic(coeff[-1], False, features)            

        # Extract correlation across orientations
        for orients in coeff[1:-1]:
            for (s1, s2) in list(itertools.combinations(orients, 2)):
                s1 = tf.math.real(s1)
                s2 = tf.math.real(s2)
                #features.append(tf.reduce_mean(s1 * s2, axis = (1,2)))
                features.append(tf.reduce_mean(s1 * s2, axis = (1,2)) / ((tf.math.reduce_std(s1, axis = (1,2)) * tf.math.reduce_std(s2, axis = (1,2))))) #(why not this?)

        # Extract correlation across heigth
        for orient in range(len(coeff[1])):
            for height in range(len(coeff) - 3):
                s1 = tf.math.real(coeff[height + 1][orient])
                s2 = tf.math.real(coeff[height + 2][orient])

                new_shape = tf.dtypes.cast(tf.shape(s1)[1:3] / 2, tf.int32)
                s1 = tf.image.resize(tf.expand_dims(s1, axis=-1), new_shape)
                s1 = tf.cast(tf.squeeze(s1), tf.float64)
                s2 = tf.cast(tf.squeeze(s2), tf.float64)

                features.append(tf.reduce_mean(s1 * s2, axis = (1,2)) / ((tf.math.reduce_std(s1, axis = (1,2)) * tf.math.reduce_std(s2, axis = (1,2)))))
        

        return tf.stack([f for f in features], axis=1)

    def extract_basic(self,band, is_complex , features):
        if (is_complex):
            band = tf.math.real(band)

        shiftx = tf.roll(band, 1, axis = 1)
        shifty = tf.roll(band, 1, axis = 2)
        
        features.append(tf.reduce_mean(band, axis = (1,2)))
        features.append(tf.math.reduce_variance(band, axis = (1,2)))
        features.append(tf.math.reduce_mean(shiftx * band, axis = (1,2)) / tf.math.reduce_variance(band, axis = (1,2)))
        features.append(tf.math.reduce_mean(shifty * band, axis = (1,2)) / tf.math.reduce_variance(band, axis = (1,2)))
    
        return features


    ### STSIM 1 global ###
    def compute_L_term(self, s1, s2, is_complex):
        self.win = int_shape(s1)[1]
        return super().compute_L_term(s1, s2, is_complex)

    def compute_C_term(self, s1, s2, is_complex):
        self.win = int_shape(s1)[1]
        return super().compute_C_term(s1, s2, is_complex)

    def compute_Cx_term(self, im1, im2, is_complex, shift_axes):
        self.win = int_shape(im1)[1]
        return super().compute_Cx_term(im1, im2, is_complex, shift_axes)

    ### STSIM 2 global ###
    def compute_cross_term(self, im11, im12, im21, im22):
        self.win = int_shape(im11)[1] - 1
        return super().compute_cross_term(im11, im12, im21, im22)
"""
