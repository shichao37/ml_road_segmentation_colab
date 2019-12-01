# -*- coding: utf-8 -*-

import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
import re

def load_image(infilename):
    """ Load an image from disk. """
    return mpimg.imread(infilename)

def pad_image(data, padding):
    """
    Extend the canvas of an image. Mirror boundary conditions are applied.
    """
    if len(data.shape) < 3:
        # Greyscale image (ground truth)
        data = np.lib.pad(data, ((padding, padding), (padding, padding)), 'reflect')
    else:
        # RGB image
        data = np.lib.pad(data, ((padding, padding), (padding, padding), (0,0)), 'reflect')
    return data
    
def img_crop_gt(im, w, h, stride):
    """ Crop an image into patches (this method is intended for ground truth images). """
    assert len(im.shape) == 2, 'Expected greyscale image.'
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    for i in range(0,imgheight,stride):
        for j in range(0,imgwidth,stride):
            im_patch = im[j:j+w, i:i+h]
            list_patches.append(im_patch)
    return list_patches

def img_crop_stride(im, w, h, stride):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight-h+1,stride):
        for j in range(0,imgwidth-w+1,stride):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches
    
def img_crop_update(im, w, h, stride, padding):
    """ Crop an image into patches, taking into account mirror boundary conditions. """
    assert len(im.shape) == 3, 'Expected RGB image.'
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    im = np.lib.pad(im, ((padding, padding), (padding, padding), (0,0)), 'reflect')
    for i in range(padding,imgheight+padding,stride):
        for j in range(padding,imgwidth+padding,stride):
            im_patch = im[j-padding:j+w+padding, i-padding:i+h+padding, :]
            list_patches.append(im_patch)
    return list_patches
    
def create_patches(X, patch_size, stride, padding):
    img_patches = np.asarray([img_crop_update(X[i], patch_size, patch_size, stride, padding) for i in range(X.shape[0])])
    # Linearize list
    img_patches = img_patches.reshape(-1, img_patches.shape[2], img_patches.shape[3], img_patches.shape[4])
    return img_patches
    
def create_patches_gt(X, patch_size, stride):
    img_patches = np.asarray([img_crop_gt(X[i], patch_size, patch_size, stride) for i in range(X.shape[0])])
    # Linearize list
    img_patches = img_patches.reshape(-1, img_patches.shape[2], img_patches.shape[3])
    return img_patches
    

