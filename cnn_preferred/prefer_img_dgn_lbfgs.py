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
from scipy.optimize import minimize
from datetime import datetime

from .utils import img_preprocess, img_deprocess, normalise_img


# main function
def generate_image(net_gen, net, layer, feature_mask,
                   feature_weight = 1.,
                   initial_gen_feat = None,
                   gen_feat_bounds = None,
                   input_layer_gen = None, output_layer_gen = None,
                   maxiter = 500, disp = True,
                   save_intermediate = False, save_intermediate_every = 1, save_intermediate_path = None
                   ):
    
    ''' Generate preferred image for the target uints using L-BFGS-B.
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
    gen_feat_bounds: list
        (min, max) pairs for each unit in the input layer of the generator, defining the boundary on the units.
        Use None or +-inf for one of min or max when there is no bound in that direction.
        Use (0, 100) as default bounds for each unit if gen_feat_bounds=None.
    input_layer_gen: str
        The name of the input layer of the generator.
    output_layer_gen: str
        The name of the output layer of the generator.
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
            save_intermediate_path = os.path.join('.','prefer_img_dgn_lbfgs_' + datetime.now().strftime('%Y%m%dT%H%M%S'))
        if not os.path.exists(save_intermediate_path):
            os.makedirs(save_intermediate_path)
    
    # input and output layers of the generator
    gen_layer_list = net_gen.blobs.keys()
    if input_layer_gen is None:
        input_layer_gen = gen_layer_list[0]
    if output_layer_gen is None:
        output_layer_gen = gen_layer_list[-1]
    
    # gen feature size
    feat_gen_size = net_gen.blobs[input_layer_gen].data.shape[1:]
    
    # initial gen feature
    if initial_gen_feat is None:
        initial_gen_feat = np.random.normal(0, 1, feat_gen_size)
        initial_gen_feat = np.float32(initial_gen_feat)
        initial_gen_feat[initial_gen_feat<0] = 0.
        initial_gen_feat = initial_gen_feat * 10.
    if save_intermediate:
        save_name = 'initial_gen_feat.mat'
        sio.savemat(os.path.join(save_intermediate_path,save_name),{'initial_gen_feat':initial_gen_feat})
    
    # gen feature bounds
    if gen_feat_bounds is None:
        num_of_unit = np.prod(feat_gen_size)
        gen_feat_bounds = []
        for j in xrange(num_of_unit):
            gen_feat_bounds.append((0.,100.)) # as default, lower bound is 0, upper bound is 100.
    
    # image size
    img_size = net.blobs['data'].data.shape[-3:]
    img_size_gen = net_gen.blobs[output_layer_gen].data.shape[-3:]
    
    # top left offset for cropping the output image to get the 224x224 image
    top_left = ((img_size_gen[1] - img_size[1])/2,(img_size_gen[2] - img_size[2])/2)
    
    # image mean
    img_mean = net.transformer.mean['data']
    
    # optimization params
    iter = [0]
    opt_params = {
                'args': (net, layer, feature_mask, feature_weight, net_gen, input_layer_gen, output_layer_gen, save_intermediate, save_intermediate_every, save_intermediate_path, iter),
                
                'method': 'L-BFGS-B',
                
                'jac': True,
                
                'bounds': gen_feat_bounds,
                
                'options': {'maxiter': maxiter, 'disp': disp},                
                }
    
    # optimization
    res = minimize(obj_fun,initial_gen_feat.flatten(),**opt_params)
    
    # feat_gen
    feat_gen = res.x
    
    # reshape gen feat
    feat_gen = feat_gen.reshape(feat_gen_size)
    
    # generator forward
    net_gen.blobs[input_layer_gen].data[0] = feat_gen.copy()
    net_gen.forward()
    
    # generated image
    img0 = net_gen.blobs[output_layer_gen].data[0].copy()
    
    # crop image
    img = img0[:,top_left[0]:top_left[0]+img_size[1],top_left[1]:top_left[1]+img_size[2]].copy()
    
    # return img
    return img_deprocess(img,img_mean)

# objective function
def obj_fun(feat_gen, net, layer, feature_mask, feature_weight, net_gen, input_layer_gen, output_layer_gen, save_intermediate, save_intermediate_every, save_intermediate_path, iter = [0]): 
    
    # reshape feat_gen
    feat_gen_size = net_gen.blobs[input_layer_gen].data.shape[1:]
    feat_gen = feat_gen.reshape(feat_gen_size)
    
    # generator forward
    net_gen.blobs[input_layer_gen].data[0] = feat_gen.copy()
    net_gen.forward()
    
    # generated image
    img0 = net_gen.blobs[output_layer_gen].data[0].copy()
    
    # crop image
    img_size = net.blobs['data'].data.shape[-3:]
    img_size_gen = net_gen.blobs[output_layer_gen].data.shape[-3:]
    top_left = ((img_size_gen[1] - img_size[1])/2,(img_size_gen[2] - img_size[2])/2)
    img = img0[:,top_left[0]:top_left[0]+img_size[1],top_left[1]:top_left[1]+img_size[2]].copy()
    
    # save intermediate image
    t = iter[0]
    print('t='+str(t))
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
    loss = - (net.blobs[layer].data[0] * feature_weight)[feature_mask==1] # since we use gradient descent, we minimize the negative value of the target units;
    
    # grad
    feat_grad = np.zeros_like(net.blobs[layer].diff[0])
    feat_grad[feature_mask==1] = -1.  # here we use gradient descent, so the gradient is negative, in order to make the target units have high positive activation;
    feat_grad = feat_grad * feature_weight
    
    # cnn backward
    net.blobs[layer].diff[0] = feat_grad.copy()
    net.backward(start=layer)
    net.blobs[layer].diff.fill(0.)
    grad = net.blobs['data'].diff[0].copy()
    
    # generator backward
    grad0 = np.zeros_like(net_gen.blobs[output_layer_gen].diff[0])
    grad0[:,top_left[0]:top_left[0]+img_size[1],top_left[1]:top_left[1]+img_size[2]] = grad.copy()
    net_gen.blobs[output_layer_gen].diff[0] = grad0.copy()
    net_gen.backward()
    net_gen.blobs[output_layer_gen].diff.fill(0.)
    grad = net_gen.blobs[input_layer_gen].diff[0].copy()
    
    # reshape gradient
    grad = grad.flatten().astype(np.float64)
    
    return loss, grad

