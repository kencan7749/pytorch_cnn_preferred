#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''generate preferred image for the target uints'''

# Author: Shen Guo-Hua <shen-gh@atr.jp>

__author__ = 'sgh'

# version: v1

# import
import os
import numpy as np
import scipy.io as sio
import PIL.Image
import caffe
from datetime import datetime

from .utils import img_deprocess, normalise_img

# main function
def generate_image(net_gen, net, layer, feature_mask,
                      feature_weight = 1.,
                      initial_gen_feat = None,
                      feat_upper_bound = 100., feat_lower_bound = 0.,
                      input_layer_gen = None, output_layer_gen = None,
                      iter_n = 200,
                      lr_start = 2., lr_end = 1e-10,
                      momentum_start = 0.9, momentum_end = 0.9,
                      decay_start = 0.01, decay_end = 0.01,
                      disp_every = 1,
                      save_intermediate = False, save_intermediate_every = 1, save_intermediate_path = None
                      ):
    
    ''' Generate preferred image for the target uints using gradient descent with momentum.
        Constrain the generated image via a deep generator net.
    
    Parameters
    ----------
    net_gen: caffe.Net object
        Deep generator net.
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
    initial_gen_feat: ndarray
        Initial features of the input layer of the generator.
        Use random noise as initial features by setting to None.
    feat_upper_bound: ndarray
        Upper boundary for the input layer of the generator.
    feat_lower_bound: ndarray
        Lower boundary for the input layer of the generator.
    input_layer_gen: str
        The name of the input layer of the generator.
    output_layer_gen: str
        The name of the output layer of the generator.
    iter_n: int
        The total number of iterations.
    lr_start: float
        The learning rate at start of the optimization.
        The learning rate will linearly decrease from lr_start to lr_end during the optimization.
    lr_end: float
        The learning rate at end of the optimization.
        The learning rate will linearly decrease from lr_start to lr_end during the optimization.
    momentum_start: float
        The momentum (gradient descend with momentum) at start of the optimization.
        The momentum will linearly decrease from momentum_start to momentum_end during the optimization.
    momentum_end: float
        The momentum (gradient descend with momentum) at the end of the optimization.
        The momentum will linearly decrease from momentum_start to momentum_end during the optimization.
    decay_start: float
        The decay rate of the features of the input layer of the generator at start of the optimization.
        The decay rate will linearly decrease from decay_start to decay_end during the optimization.
    decay_end: float
        The decay rate of the features of the input layer of the generator at the end of the optimization.
        The decay rate will linearly decrease from decay_start to decay_end during the optimization.
    disp_every: int
        Display the optimization information for every n iterations.
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
    
    # make save dir
    if save_intermediate:
        if save_intermediate_path is None:
            save_intermediate_path = os.path.join('.','prefer_img_dgn_gd_' + datetime.now().strftime('%Y%m%dT%H%M%S'))
        if not os.path.exists(save_intermediate_path):
            os.makedirs(save_intermediate_path)
    
    # input and output layers of the generator
    gen_layer_list = net_gen.blobs.keys()
    if input_layer_gen is None:
        input_layer_gen = gen_layer_list[0]
    if output_layer_gen is None:
        output_layer_gen = gen_layer_list[-1]
    
    # feature size
    feat_size_gen = net_gen.blobs[input_layer_gen].data.shape[1:]
    
    # initial feature
    if initial_gen_feat is None:
        initial_gen_feat = np.random.normal(0, 1, feat_size_gen)
        initial_gen_feat = np.float32(initial_gen_feat)
        initial_gen_feat[initial_gen_feat<0] = 0.
        initial_gen_feat = initial_gen_feat * 10.
    if save_intermediate:
        save_name = 'initial_gen_feat.mat'
        sio.savemat(os.path.join(save_intermediate_path,save_name),{'initial_gen_feat':initial_gen_feat})
    
    # image size
    img_size = net.blobs['data'].data.shape[-3:]
    img_size_gen = net_gen.blobs[output_layer_gen].data.shape[-3:]
    
    # top left offset for cropping the output image to get the 227x227 image
    top_left = ((img_size_gen[1] - img_size[1])/2,(img_size_gen[2] - img_size[2])/2)
    
    # image mean
    img_mean = net.transformer.mean['data']
    
    # iteration for gradient descent
    feat_gen = initial_gen_feat.copy()
    delta_feat_gen = np.zeros_like(feat_gen)
    feat_grad = np.zeros_like(net.blobs[layer].diff[0])
    feat_grad[feature_mask==1] = -1.  # here we use gradient descent, so the gradient is negative, in order to make the target units have high positive activation;
    feat_grad = feat_grad * feature_weight
    for t in xrange(iter_n):
        
        # parameters
        lr = lr_start + t * (lr_end - lr_start) / iter_n
        decay = decay_start + t * (decay_end - decay_start) / iter_n
        momentum = momentum_start + t * (momentum_end - momentum_start) / iter_n
        
        # forward for generator
        net_gen.blobs[input_layer_gen].data[0] = feat_gen.copy()
        net_gen.forward()
        #print('feat_gen='+str(np.mean(np.abs(feat_gen))))
        
        # generated image
        img0 = net_gen.blobs[output_layer_gen].data[0].copy()
        
        # crop image
        img = img0[:,top_left[0]:top_left[0]+img_size[1],top_left[1]:top_left[1]+img_size[2]].copy()
        if t==0 and save_intermediate:
            save_name = 'initial_img.jpg'
            PIL.Image.fromarray(np.uint8(img_deprocess(img,img_mean))).save(os.path.join(save_intermediate_path,save_name))
        
        # forward for net
        net.blobs['data'].data[0] = img.copy()
        net.forward(end=layer)
        feat = net.blobs[layer].data[0][feature_mask==1]
        feat_abs_mean = np.mean(np.abs(feat))
        
        # backward for net
        net.blobs[layer].diff[0] = feat_grad.copy()
        net.backward(start=layer)
        net.blobs[layer].diff.fill(0.)
        g = net.blobs['data'].diff[0].copy()
        
        # backward for generator
        g0 = np.zeros_like(net_gen.blobs[output_layer_gen].diff[0])
        g0[:,top_left[0]:top_left[0]+img_size[1],top_left[1]:top_left[1]+img_size[2]] = g.copy()
        net_gen.blobs[output_layer_gen].diff[0] = g0.copy()
        net_gen.backward()
        net_gen.blobs[output_layer_gen].diff.fill(0.)
        g = net_gen.blobs[input_layer_gen].diff[0].copy()
        
        # normalize gradient
        g_mean = np.abs(g).mean()
        if g_mean>0:
            g = g / g_mean
        
        # gradient with momentum
        delta_feat_gen = delta_feat_gen * momentum + g
        
        # feat update
        feat_gen = feat_gen - lr * delta_feat_gen
        
        # L_2 decay
        feat_gen = (1-decay) * feat_gen
        
        # clip feat
        if feat_lower_bound is not None:
            feat_gen = np.maximum(feat_gen,feat_lower_bound)
        
        if feat_upper_bound is not None:
            feat_gen = np.minimum(feat_gen,feat_upper_bound)
        
        # disp info
        if (t+1)%disp_every==0:
            print('iter=%d; mean(abs(feat))=%g;'%(t+1,feat_abs_mean))
        
        # save image
        if save_intermediate and ((t+1)%save_intermediate_every==0):
            save_name = '%05d.jpg'%(t+1)
            PIL.Image.fromarray(normalise_img(img_deprocess(img,img_mean))).save(os.path.join(save_intermediate_path,save_name))
        
    # return img
    return img_deprocess(img,img_mean)

