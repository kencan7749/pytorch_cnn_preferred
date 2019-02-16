#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''generate preferred image for the target uints'''

# Author: Shen Guo-Hua <shen-gh@atr.jp>

__author__ = 'sgh'

# version: v1

# import
import os
import numpy as np
import PIL.Image
import caffe
from scipy.optimize import minimize
from datetime import datetime

from .utils import img_preprocess, img_deprocess, normalise_img

# main function
def generate_image(net, layer, feature_mask,
                   feature_weight = 1., initial_image = None, maxiter = 500, disp = True, save_intermediate = False, save_intermediate_every = 1, save_intermediate_path = None):
    
    ''' Generate preferred image for the target uints using L-BFGS-B.
    
    Parameters
    ----------
    net: caffe.Classifier or caffe.Net object
        CNN model coresponding to the target CNN features.
    layer: str
        The name of the layer for the target units.
    feature_mask: ndarray
        The mask used to select the target units.
        The shape of the mask should be the same as that of the CNN features in that layer.
        The values of the mask array are binary, (1: target uint; 0: irrelevant unit)
    
    Optional Parameters
    ----------
    feature_weight: float or ndarray
        The weight for each target unit.
        If it is scalar, the scalar will be used as the universal weight for all units.
        If it is numpy array, it allows to specify different weights for different uints.
    initial_image: ndarray
        Initial image for the optimization.
        Use random noise as initial image by setting to None.
    maxiter: int
        The maximum number of iterations.
    disp: bool
        Display the optimization information or not.
    save_intermediate: bool
        Save the intermediate reconstruction or not.
    save_intermediate_every: int
        Save the intermediate reconstruction for every n iterations.
    save_intermediate_path: str
        The path to save the intermediate reconstruction.
    
    Returns
    -------
    img: ndarray
        The preferred image [227x227x3]. 
    '''
    
    # make dir for saving intermediate
    if save_intermediate:
        if save_intermediate_path is None:
            save_intermediate_path = os.path.join('.','prefer_img_lbfgs_' + datetime.now().strftime('%Y%m%dT%H%M%S'))
        if not os.path.exists(save_intermediate_path):
            os.makedirs(save_intermediate_path)
    
    # image size
    img_size = net.blobs['data'].data.shape[-3:]
    
    # num of pixel
    num_of_pix = np.prod(img_size)
    
    # image mean
    img_mean = net.transformer.mean['data']
    
    # img bounds
    img_min = -img_mean
    img_max = img_min + 255.
    img_bounds = [(img_min[0],img_max[0])]*(num_of_pix/3) + [(img_min[1],img_max[1])]*(num_of_pix/3) + [(img_min[2],img_max[2])]*(num_of_pix/3)
    
    # initial image
    if initial_image is None:
        initial_image = np.random.randint(0,256,(img_size[1],img_size[2],img_size[0]))
    if save_intermediate:
        save_name = 'initial_image.jpg'
        PIL.Image.fromarray(np.uint8(initial_image)).save(os.path.join(save_intermediate_path,save_name))
    
    # preprocess initial img
    initial_image = img_preprocess(initial_image,img_mean)
    initial_image = initial_image.flatten()
    
    # optimization params
    iter = [0]
    opt_params = {
                'args': (net, layer, feature_mask, feature_weight, save_intermediate, save_intermediate_every, save_intermediate_path, iter),
                
                'method': 'L-BFGS-B',
                
                'jac': True,
                
                'bounds': img_bounds,
                
                'options': {'maxiter': maxiter, 'disp': disp},  # 'ftol': 0, 'gtol': 0, 'maxls': 50
                }
    
    # optimization
    res = minimize(obj_fun,initial_image,**opt_params)
    
    # recon img
    img = res.x
    img = img.reshape(img_size)
    
    # return img
    return img_deprocess(img,img_mean)

# objective function
def obj_fun(img, net, layer, feature_mask, feature_weight, save_intermediate, save_intermediate_every, save_intermediate_path, iter=[0]):
    #
    #global loss_list
    
    # reshape img
    img_size = net.blobs['data'].data.shape[-3:]
    img = img.reshape(img_size)
    
    # save intermediate image
    t = iter[0]
    if save_intermediate and (t%save_intermediate_every==0):
        img_mean = net.transformer.mean['data']
        save_name = '%05d.jpg'%t
        PIL.Image.fromarray(normalise_img(img_deprocess(img,img_mean))).save(os.path.join(save_intermediate_path,save_name))
    t = t + 1
    iter[0] = t
    
    # cnn forward
    net.blobs['data'].data[0] = img.copy()
    net.forward(end=layer)
    
    # loss
    loss = - (net.blobs[layer].data[0] * feature_weight)[feature_mask==1]  # since we use gradient descent, we minimize the negative value of the target units;
    
    # grad
    feat_grad = np.zeros_like(net.blobs[layer].diff[0])
    feat_grad[feature_mask==1] = -1.  # here we use gradient descent, so the gradient is negative, in order to make the target units have high positive activation;
    feat_grad = feat_grad * feature_weight
    
    # cnn backward
    net.blobs[layer].diff[0] = feat_grad.copy()
    net.backward(start=layer)
    net.blobs[layer].diff.fill(0.)
    grad = net.blobs['data'].diff[0].copy()
    
    # reshape gradient
    grad = grad.flatten().astype(np.float64)
    
    return loss, grad

