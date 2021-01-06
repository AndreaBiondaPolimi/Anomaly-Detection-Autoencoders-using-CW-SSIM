# MIT License
#
# Copyright (c) 2020 Didan Deng
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Didan Deng
# Date Created: 2020-03-31

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np

import Steerables.math_utils as math_utils
pointOp = math_utils.pointOp
factorial = math_utils.factorial
class SCFpyr_TF():
    '''
    This is a modified version of buildSFpyr, that constructs a
    complex-valued steerable pyramid  using Hilbert-transform pairs
    of filters. Note that the imaginary parts will *not* be steerable.

    Description of this transform appears in: Portilla & Simoncelli,
    International Journal of Computer Vision, 40(1):49-71, Oct 2000.
    Further information: http://www.cns.nyu.edu/~eero/STEERPYR/

    Modified code from the perceptual repository:
      https://github.com/andreydung/Steerable-filter

    This code looks very similar to the original Matlab code:
      https://github.com/LabForComputationalVision/matlabPyrTools/blob/master/buildSCFpyr.m

    Also looks very similar to the original Python code presented here:
      https://github.com/LabForComputationalVision/pyPyrTools/blob/master/pyPyrTools/SCFpyr.py

    '''
    def __init__(self, height=5, nbands=4, scale_factor=2, precision=32, patch_size=256):#, device=None):
        self.height = height  # including low-pass and high-pass
        self.nbands = nbands  # number of orientation bands
        self.scale_factor = scale_factor
        self.precision = precision
        assert self.precision in [32, 64]
        self.dtype = eval('tf.complex{}'.format(self.precision*2))
        self.PI = np.pi
        # Cache constants
        self.lutsize = 1024
        self.Xcosn = np.pi * np.array(range(-(2*self.lutsize+1), (self.lutsize+2)))/self.lutsize
        self.alpha = (self.Xcosn + self.PI) % (2*self.PI) - self.PI
        self.patch_size = patch_size
    ################################################################################
    # Construction of Steerable Pyramid

    def build(self, im_batch):
        ''' Decomposes a batch of images into a complex steerable pyramid. 
        The pyramid typically has ~4 levels and 4-8 orientations. 
        
        Args:
            im_batch (tf.Tensor or np.ndarray): Batch of images of shape [N,C,H,W]
        
        Returns:
            pyramid: list containing tf.Tensor objects storing the pyramid
        '''
        assert len(im_batch.shape)  == 4, 'Image batch must be of shape [N,H,W, C]'
        #assert im_batch.shape[-1] == 1, 'final dimension must be 1 encoding grayscale image'
        if type(im_batch)==np.ndarray:
            im_batch = tf.convert_to_tensor(im_batch, self.dtype)
        #assert im_batch.dtype == self.dtype
        im_batch = tf.squeeze(im_batch, -1)  # flatten channels dim
        
        # height, width = im_batch.shape[2], im_batch.shape[1] 
        # Check whether image size is sufficient for number of levels
        #if self.height > int(np.floor(np.log2(min(width, height))) - 2):
            #raise RuntimeError('Cannot build {} levels, image too small.'.format(self.height))
        # Prepare a grid
        log_rad, angle = math_utils.prepare_grid(self.patch_size, self.patch_size)

        # Radial transition function (a raised cosine in log-frequency):
        Xrcos, Yrcos = math_utils.rcosFn(1, -0.5)
        Yrcos = np.sqrt(Yrcos)

        YIrcos = np.sqrt(1 - Yrcos**2)

        lo0mask = pointOp(log_rad, YIrcos, Xrcos)
        hi0mask = pointOp(log_rad, Yrcos, Xrcos)
        # Note that we expand dims to support broadcasting later
        lo0mask = tf.convert_to_tensor(lo0mask[None,:,:], self.dtype)
        hi0mask = tf.convert_to_tensor(hi0mask[None,:,:], self.dtype)

        im_batch = tf.dtypes.cast(im_batch, tf.complex128)

        # Fourier transform (2D) and shifting
        imdft = tf.signal.fft2d(im_batch)
        imdft = tf.signal.fftshift(imdft)
        # Low-pass
        lo0dft = imdft * lo0mask

        # Recursive build the steerable pyramid
        coeff = self._build_levels(lo0dft, log_rad, angle, Xrcos, Yrcos, self.height-1)
        # High-pass
        hi0dft = imdft * hi0mask
        hi0 = tf.signal.ifft2d(tf.signal.ifftshift(hi0dft))
        coeff.insert(0, tf.math.real(hi0))
        return coeff
    def _build_levels(self, lodft, log_rad, angle, Xrcos, Yrcos, height):

        if height <= 1:

            # Low-pass
            lo0 = tf.signal.ifftshift(lodft)
            lo0 = tf.signal.ifft2d(lo0)
            coeff = [tf.math.real(lo0)]

        else:
            
            Xrcos = Xrcos - np.log2(self.scale_factor)

            ####################################################################
            ####################### Orientation bandpass #######################
            ####################################################################

            himask = pointOp(log_rad, Yrcos, Xrcos)
            himask = tf.convert_to_tensor(himask, self.dtype)

            order = self.nbands - 1
            const = np.power(2, 2*order) * np.square(factorial(order)) / (self.nbands * factorial(2*order))
            Ycosn = 2*np.sqrt(const) * np.power(np.cos(self.Xcosn), order) * (np.abs(self.alpha) < np.pi/2)

            # Loop through all orientation bands
            orientations = []
            for b in range(self.nbands):
                anglemask = pointOp(angle, Ycosn, self.Xcosn + np.pi*b/self.nbands)
                anglemask = tf.convert_to_tensor(anglemask, self.dtype)
                alpha = tf.convert_to_tensor(np.power(np.complex(0, -1), self.nbands - 1), self.dtype)
                banddft = alpha * lodft * anglemask * himask
                band = tf.signal.ifft2d(tf.signal.ifftshift(banddft))
                orientations.append(band)
            ####################################################################
            ######################## Subsample lowpass #########################
            ####################################################################

            dims = np.array(lodft.shape)[-2:].astype(int)

            # Both are tuples of size 2
            low_ind_start = (np.ceil((dims+0.5)/2) - np.ceil((np.ceil((dims-0.5)/2)+0.5)/2)).astype(int)
            low_ind_end = (low_ind_start + np.ceil((dims-0.5)/2)).astype(int)
          
            # Selection
            log_rad = log_rad[low_ind_start[0]:low_ind_end[0], low_ind_start[1]:low_ind_end[1]]
            angle   = angle[low_ind_start[0]:low_ind_end[0], low_ind_start[1]:low_ind_end[1]]
            lodft   = lodft[..., low_ind_start[0]:low_ind_end[0], low_ind_start[1]:low_ind_end[1]]# first dimension is batch size

            # Subsampling in frequency domain
            YIrcos = np.abs(np.sqrt(1 - Yrcos**2))
            lomask = pointOp(log_rad, YIrcos, Xrcos)
            lomask = tf.convert_to_tensor(lomask[None,:,:], self.dtype) # size:1, W, H

            # Convolution in spatial domain
            lodft = lomask * lodft
            ####################################################################
            ####################### Recursion next level #######################
            ####################################################################

            coeff = self._build_levels(lodft, log_rad, angle, Xrcos, Yrcos, height-1)
            coeff.insert(0, orientations)
        return coeff





    def reconstruct(self, coeff):

        if self.nbands != len(coeff[1]):
            raise Exception("Unmatched number of orientations")
        height, width = coeff[0].shape[2], coeff[0].shape[1] 
        log_rad, angle = math_utils.prepare_grid(height, width)

        Xrcos, Yrcos = math_utils.rcosFn(1, -0.5)
        Yrcos  = np.sqrt(Yrcos)
        YIrcos = np.sqrt(np.abs(1 - Yrcos**2))

        lo0mask = pointOp(log_rad, YIrcos, Xrcos)
        hi0mask = pointOp(log_rad, Yrcos, Xrcos)
        # Note that we expand dims to support broadcasting later
        lo0mask = tf.convert_to_tensor(lo0mask[None,:,:], self.dtype)
        hi0mask = tf.convert_to_tensor(hi0mask[None,:,:], self.dtype)

        tempdft = self._reconstruct_levels(coeff[1:], log_rad, Xrcos, Yrcos, angle)
        if coeff[0].dtype != self.dtype:
            hidft = tf.signal.fftshift(tf.signal.fft2d(tf.cast(coeff[0], self.dtype)))
        else:
            hidft = tf.signal.fftshift(tf.signal.fft2d(coeff[0]))
        outdft = tempdft * lo0mask + hidft * hi0mask

        reconstruction = tf.signal.ifftshift(outdft)
        reconstruction = tf.signal.ifft2d(reconstruction)
        reconstruction = tf.math.real(reconstruction)

        return reconstruction
    def _reconstruct_levels(self, coeff, log_rad, Xrcos, Yrcos, angle):

        if len(coeff) == 1:
            #width, height = coeff[0].shape[-2:]
            if coeff[0].dtype != self.dtype:
                dft = tf.signal.fft2d(tf.cast(coeff[0], self.dtype))
            else:
                dft = tf.signal.fft2d(coeff[0])
            dft = tf.signal.fftshift(dft)
            return dft

        Xrcos = Xrcos - np.log2(self.scale_factor)
        ####################################################################
        ####################### Orientation Residue ########################
        ####################################################################

        himask = pointOp(log_rad, Yrcos, Xrcos)

        lutsize = 1024
        Xcosn = np.pi * np.array(range(-(2*lutsize+1), (lutsize+2)))/lutsize
        order = self.nbands - 1
        const = np.power(2, 2*order) * np.square(factorial(order)) / (self.nbands * factorial(2*order))
        Ycosn = np.sqrt(const) * np.power(np.cos(Xcosn), order)

        orientdft = tf.convert_to_tensor(np.zeros(coeff[0][0].shape), self.dtype)
        for b in range(self.nbands):

            anglemask = pointOp(angle, Ycosn, Xcosn + np.pi * b/self.nbands)
            anglemask = anglemask[None,:,:]  # for broadcasting
            anglemask = tf.convert_to_tensor(anglemask, self.dtype)
            if coeff[0][b].dtype != self.dtype:
                banddft = tf.signal.fft2d(tf.cast(coeff[0][b], self.dtype))
            else:
                banddft = tf.signal.fft2d(coeff[0][b])

            banddft = tf.signal.fftshift(banddft)
            alpha = tf.convert_to_tensor(np.power(np.complex(0, 1), order), self.dtype)

            orientdft = orientdft + alpha * banddft * anglemask * himask

        ####################################################################
        ########## Lowpass component are upsampled and convoluted ##########
        ####################################################################

        dims = np.array(coeff[0][0].shape[1:3])

        lostart = (np.ceil((dims+0.5)/2) - np.ceil((np.ceil((dims-0.5)/2)+0.5)/2)).astype(np.int32)
        loend = lostart + np.ceil((dims-0.5)/2).astype(np.int32)

        nlog_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
        nangle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
        YIrcos = np.sqrt(np.abs(1 - Yrcos**2))
        lomask = pointOp(nlog_rad, YIrcos, Xrcos)
        lomask = tf.convert_to_tensor(lomask[None,:,:], self.dtype) # size:1, W, H

        ################################################################################

        # Recursive call for image reconstruction
        nresdft = self._reconstruct_levels(coeff[1:], nlog_rad, Xrcos, Yrcos, nangle)

        resdft = np.zeros(coeff[0][0].shape, 'complex')
        resdft = tf.Variable(tf.convert_to_tensor(resdft)) #tf.complex128
        dft = tf.cast(nresdft * lomask, tf.complex128)
        resdft = resdft[:, lostart[0]:loend[0], lostart[1]:loend[1]].assign(dft)

        return tf.cast(resdft, self.dtype) + tf.cast(orientdft, self.dtype)



class SCFpyr_TF_nosub(SCFpyr_TF):
    def _build_levels(self, lodft, log_rad, angle, Xrcos, Yrcos, height):

        if height <= 1:

            # Low-pass
            lo0 = tf.signal.ifftshift(lodft)
            lo0 = tf.signal.ifft2d(lo0)
            coeff = [tf.math.real(lo0)]

        else:
            
            Xrcos = Xrcos - np.log2(self.scale_factor)

            ####################################################################
            ####################### Orientation bandpass #######################
            ####################################################################

            himask = pointOp(log_rad, Yrcos, Xrcos)
            himask = tf.convert_to_tensor(himask, self.dtype)

            order = self.nbands - 1
            const = np.power(2, 2*order) * np.square(factorial(order)) / (self.nbands * factorial(2*order))
            Ycosn = 2*np.sqrt(const) * np.power(np.cos(self.Xcosn), order) * (np.abs(self.alpha) < np.pi/2)

            # Loop through all orientation bands
            orientations = []
            for b in range(self.nbands):
                anglemask = pointOp(angle, Ycosn, self.Xcosn + np.pi*b/self.nbands)
                anglemask = tf.convert_to_tensor(anglemask, self.dtype)
                alpha = tf.convert_to_tensor(np.power(np.complex(0, -1), self.nbands - 1), self.dtype)
                banddft = alpha * lodft * anglemask * himask
                band = tf.signal.ifft2d(tf.signal.ifftshift(banddft))
                orientations.append(band)
            ####################################################################
            ######################## Subsample lowpass #########################
            ####################################################################
            low_ind_start = (0, 0)
            low_ind_end = (self.patch_size, self.patch_size)
            #low_ind_end = tf.shape(lodft)
          
            # Selection
            log_rad = log_rad[low_ind_start[0]:low_ind_end[0], low_ind_start[1]:low_ind_end[1]]
            angle   = angle[low_ind_start[0]:low_ind_end[0], low_ind_start[1]:low_ind_end[1]]
            lodft   = lodft[..., low_ind_start[0]:low_ind_end[0], low_ind_start[1]:low_ind_end[1]]# first dimension is batch size

            # Subsampling in frequency domain
            YIrcos = np.abs(np.sqrt(1 - Yrcos**2))
            lomask = pointOp(log_rad, YIrcos, Xrcos)
            lomask = tf.convert_to_tensor(lomask[None,:,:], self.dtype) # size:1, W, H

            # Convolution in spatial domain
            lodft = lomask * lodft
            ####################################################################
            ####################### Recursion next level #######################
            ####################################################################

            coeff = self._build_levels(lodft, log_rad, angle, Xrcos, Yrcos, height-1)
            coeff.insert(0, orientations)
        return coeff