#! /usr/bin/env python
# -*- coding: utf-8 -*-


'''generate preferred image/video for the target uints'''

# Author: Ken SHIRAKAWA <shirakawa.ken.38w@st.kyoto-u.ac.jp>

__author__ = 'ks'

# version: v0.1

# import
import os
import numpy as np
import PIL.Image
import torch
from datetime import datetime

from utils import img_preprocess, img_deprocess, normalise_img, p_norm, TV_norm,TV_norm_vid, image_norm, gaussian_blur, \
    clip_extreme_pixel, clip_small_norm_pixel, clip_small_contribution_pixel,save_video, save_gif, normalise_vid, vid_preprocess, vid_deprocess,get_cnn_features, create_feature_mask




class minusLoss(torch.nn.Module):
    """
    loss function for hand made
    In pytorch, hand made loss function requires two method, __init__ and forward
    """

    def __init__(self):
        super(minusLoss, self).__init__()

    def forward(self, act):
        return -torch.sum(act)


# main function
def generate_preferred(net, exec_code, channel=None,
                   feature_mask = None,
                   img_mean = (0,0,0),
                   img_std = (1,1,1),
                   norm = 255,
                   input_size=(224, 224, 3), bgr = False,
                   feature_weight=1.,
                   initial_input=None,
                   iter_n=200,
                   lr_start=1., lr_end=1.,
                   momentum_start=0.001, momentum_end=0.001,
                   decay_start=0.001, decay_end=0.001,
                   grad_normalize=True,
                   image_jitter=True, jitter_size=32,
                   image_blur=True, sigma_start=2.5, sigma_end=0.5,
                   use_p_norm_reg=False, p=2, lamda_start=0.5, lamda_end=0.5,
                   use_TV_norm_reg=False, TVbeta1=2, TVbeta2=2, TVlamda_start=0.5, TVlamda_end=0.5,
                   clip_extreme=False, clip_extreme_every=4, e_pct_start=1, e_pct_end=1,
                   clip_small_norm=False, clip_small_norm_every=4, n_pct_start=5., n_pct_end=5.,
                   clip_small_contribution=False, clip_small_contribution_every=4, c_pct_start=5., c_pct_end=5.,
                   disp_every=1,
                   save_intermediate=False, save_intermediate_every=1, save_intermediate_path=None
                   ):
    '''Generate preferred image/video for the target uints using gradient descent with momentum.

        Parameters
        ----------
        net: torch.nn.Module
            CNN model coresponding to the target CNN features.

        feature_mask: ndarray
            The mask used to select the target units.
            The shape of the mask should be the same as that of the CNN features in that layer.
            The values of the mask array are binary, (1: target uint; 0: irrelevant unit)

        exec_code: list
           The code to extract intermidiate layer. This code is run in the 'get_cnn_feature' function
        img_mean: np.ndarray
            set the mean in rgb order to pre/de-process to input/output image/video
        img_std : np.ndarray
            set the std in rgb order to pre/de-process to input/output image/video

        input_size: np.ndarray
            the shape correspond to the CNN available input
        Optional Parameters
        ----------
        feature_weight: float or ndarray
            The weight for each target unit.
            If it is scalar, the scalar will be used as the universal weight for all units.
            If it is numpy array, it allows to specify different weights for different uints.
        initial_input: ndarray
            Initial image for the optimization.
            Use random noise as initial image by setting to None.
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
            The decay rate of the image pixels at start of the optimization.
            The decay rate will linearly decrease from decay_start to decay_end during the optimization.
        decay_end: float
            The decay rate of the image pixels at the end of the optimization.
            The decay rate will linearly decrease from decay_start to decay_end during the optimization.
        grad_normalize: bool
            Normalise the gradient or not for each iteration.
        image_jitter: bool
            Use image jittering or not.
            If true, randomly shift the intermediate reconstructed image for each iteration.
        jitter_size: int
            image jittering in number of pixels.
        image_blur: bool
            Use image smoothing or not.
            If true, smoothing the image for each iteration.
        sigma_start: float
            The size of the gaussian filter for image smoothing at start of the optimization.
            The sigma will linearly decrease from sigma_start to sigma_end during the optimization.
        sigma_end: float
            The size of the gaussian filter for image smoothing at the end of the optimization.
            The sigma will linearly decrease from sigma_start to sigma_end during the optimization.
        use_p_norm_reg: bool
            Use p-norm loss for image or not as regularization term.
        p: float
            The order of the p-norm loss of image
        lamda_start: float
            The weight for p-norm loss at start of the optimization.
            The lamda will linearly decrease from lamda_start to lamda_end during the optimization.
        lamda_end: float
            The weight for p-norm loss at the end of the optimization.
            The lamda will linearly decrease from lamda_start to lamda_end during the optimization.
        use_TV_norm_reg: bool
            Use TV-norm or not as regularization term.
        TVbeta: float
            The order of the TV-norm.
        TVlamda_start: float
            The weight for TV-norm regularization term at start of the optimization.
            The TVlamda will linearly decrease from TVlamda_start to TVlamda_end during the optimization.
        TVlamda_end: float
            The weight for TV-norm regularization term at the end of the optimization.
            The TVlamda will linearly decrease from TVlamda_start to TVlamda_end during the optimization.
        clip_extreme: bool
            Clip or not the pixels with extreme high or low value.
        clip_extreme_every: int
            Clip the pixels with extreme value every n iterations.
        e_pct_start: float
            the percentage of pixels to be clipped at start of the optimization.
            The percentage will linearly decrease from e_pct_start to e_pct_end during the optimization.
        e_pct_end: float
            the percentage of pixels to be clipped at the end of the optimization.
            The percentage will linearly decrease from e_pct_start to e_pct_end during the optimization.
        clip_small_norm: bool
            Clip or not the pixels with small norm of RGB valuse.
        clip_small_norm_every: int
            Clip the pixels with small norm every n iterations
        n_pct_start: float
            The percentage of pixels to be clipped at start of the optimization.
            The percentage will linearly decrease from n_pct_start to n_pct_end during the optimization.
        n_pct_end: float
            The percentage of pixels to be clipped at start of the optimization.
            The percentage will linearly decrease from n_pct_start to n_pct_end during the optimization.
        clip_small_contribution: bool
            Clip or not the pixels with small contribution: norm of RGB channels of (img*grad).
        clip_small_contribution_every: int
            Clip the pixels with small contribution every n iterations.
        c_pct_start: float
            The percentage of pixels to be clipped at start of the optimization.
            The percentage will linearly decrease from c_pct_start to c_pct_end during the optimization.
        c_pct_end: float
            The percentage of pixels to be clipped at the end of the optimization.
            The percentage will linearly decrease from c_pct_start to c_pct_end during the optimization.
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
            The preferred image/video same shape as input_size.

     '''

    # make save dir
    if save_intermediate:
        if save_intermediate_path is None:
            save_intermediate_path = os.path.join('.', 'preferred_gd_' + datetime.now().strftime('%Y%m%dT%H%M%S'))
        if not os.path.exists(save_intermediate_path):
            os.makedirs(save_intermediate_path, exist_ok=True)

    # initial input
    if initial_input is None:
        initial_input = np.random.randint(0, 256, (input_size))
    else:
        input_size = initial_input.shape
    # image mean
    img_mean = img_mean
    img_std = img_std
    # image norm
    noise_vid = np.random.randint(0, 256, (input_size))
    img_norm0 = np.linalg.norm(noise_vid)
    img_norm0 = img_norm0 / 2.


    if save_intermediate:
        if len(input_size) == 3:
            #image
            save_name = 'initial_video.jpg'
            if bgr:
                PIL.Image.fromarray(np.uint8(initial_input[...,[2,1,0]])).save(os.path.join(save_intermediate_path, save_name))
            else:
                PIL.Image.fromarray(np.uint8(initial_input)).save(os.path.join(save_intermediate_path, save_name))
        elif len(input_size) == 4:
            # video
            # if you install cv2 and ffmpeg, you can use save_video function which save preferred video as video format
            #save_name = 'initial_video.avi'
            #save_video(initial_input, save_name, save_intermediate_path, bgr)

            save_name = 'initial_video.gif'
            save_gif(initial_input, save_name, save_intermediate_path, bgr,
                     fr_rate=150)

        else:
            print('Input size is not appropriate for save')
            assert len(input_size) not in [3,4]

    # create feature mask if not define
    if feature_mask is None:
        feature_mask = create_feature_mask(net, exec_code, input_size, channel)

    # iteration for gradient descent
    init_input = initial_input.copy()
    if len(input_size) == 3:
        #Image
        input = img_preprocess(init_input, img_mean, img_std, norm)
    else:
        #Video
        input = vid_preprocess(init_input, img_mean, img_std, norm)
    delta_input = np.zeros_like(input)
    feat_grad = np.zeros_like(feature_mask)
    feat_grad[feature_mask == 1] = -1.  # here we use gradient descent, so the gradient is negative, in order to make the target units have high positive activation;
    feat_grad = feat_grad * feature_weight

    # Loss function (minus Loss)
    loss_fun = minusLoss()


    for t in range(iter_n):

        # parameters
        lr = lr_start + t * (lr_end - lr_start) / iter_n
        momentum = momentum_start + t * (momentum_end - momentum_start) / iter_n
        decay = decay_start + t * (decay_end - decay_start) / iter_n
        sigma = sigma_start + t * (sigma_end - sigma_start) / iter_n

        # shift
        if image_jitter:
            ox, oy = np.random.randint(-jitter_size, jitter_size + 1, 2)
            input = np.roll(np.roll(input, ox, -1), oy, -2)
            delta_input = np.roll(np.roll(delta_input, ox, -1), oy, -2)
        # create Tensor
        input = torch.Tensor(input[np.newaxis])
        input.requires_grad_()
        # forward
        fw = get_cnn_features(net, input, exec_code)[0]


        feat = torch.masked_select(fw, torch.ByteTensor(feature_mask))
        feat_abs_mean = np.mean(np.abs(feat[0].detach().numpy()))

        #for the first time iteration, input.grad is None
        if input.grad is not None:
            input.grad.data.zero_()
        # zero grad
        net.zero_grad()

        # backward for net
        loss = loss_fun(feat)
        loss.backward()


        grad = input.grad.numpy()
        input = input.detach().numpy()
        # normalize gradient
        if grad_normalize:
            grad_mean = np.abs(grad).mean()
            if grad_mean > 0:
                grad = grad / grad_mean

        # gradient with momentum
        delta_input = delta_input * momentum + grad

        # p norm regularization
        if use_p_norm_reg:
            lamda = lamda_start + t * (lamda_end - lamda_start) / iter_n
            _, grad_r = p_norm(input, p)
            grad_r = grad_r / (img_norm0 ** 2)
            if grad_normalize:
                grad_mean = np.abs(grad_r).mean()
                if grad_mean > 0:
                    grad_r = grad_r / grad_mean
            delta_input = delta_input + lamda * grad_r

        # TV norm regularization
        if use_TV_norm_reg:
            TVlamda = TVlamda_start + t * (TVlamda_end - TVlamda_start) / iter_n
            if len(input_size) == 3:
                loss_r, grad_r = TV_norm(input, TVbeta1)
            else:
                loss_r, grad_r = TV_norm_vid(input, TVbeta1, TVbeta2)
            loss_r = loss_r / (img_norm0 ** 2)
            grad_r = grad_r / (img_norm0 ** 2)
            if grad_normalize:
                grad_mean = np.abs(grad_r).mean()
                if grad_mean > 0:
                    grad_r = grad_r / grad_mean
            delta_input = delta_input + TVlamda * grad_r

        # input update [0] means remove the newaxis
        input = np.add(input, - lr * delta_input, dtype=np.float32)[0]
        grad = grad[0]
        delta_input = delta_input[0]
        # clip pixels with extreme value
        if clip_extreme and (t + 1) % clip_extreme_every == 0:
            e_pct = e_pct_start + t * (e_pct_end - e_pct_start) / iter_n
            input= clip_extreme_pixel(input, e_pct)

        # clip pixels with small norm
        if clip_small_norm and (t + 1) % clip_small_norm_every == 0:
            n_pct = n_pct_start + t * (n_pct_end - n_pct_start) / iter_n
            input = clip_small_norm_pixel(input, n_pct)

        # clip pixels with small contribution
        if clip_small_contribution and (t + 1) % clip_small_contribution_every == 0:
            c_pct = c_pct_start + t * (c_pct_end - c_pct_start) / iter_n
            input = clip_small_contribution_pixel(input, grad, c_pct)

        # unshift
        if image_jitter:
            input = np.roll(np.roll(input, -ox, -1), -oy, -2)
            delta_input = delta_input - grad
            delta_input = np.roll(np.roll(delta_input, -ox, -1), -oy, -2)
            delta_input = delta_input + grad

        # L_2 decay
        input = (1 - decay) * input

        # gaussian blur
        if image_blur:
            if len(input_size) == 3:
                input = gaussian_blur(input, sigma)
            else:
                for i  in range(input.shape[1]):
                    input[:,i] = gaussian_blur(input[:,i], sigma)

        # disp info
        if (t + 1) % disp_every == 0:
            print('iter=%d; mean(abs(feat))=%g;' % (t + 1, feat_abs_mean))

        # save image
        if save_intermediate and ((t + 1) % save_intermediate_every == 0):
            if len(input_size) == 3:
                save_name = '%05d.jpg' % (t + 1)
                if bgr:
                    PIL.Image.fromarray(
                        normalise_img(img_deprocess(input, img_mean, img_std, norm)[..., [2, 1, 0]])).save(
                        os.path.join(save_intermediate_path, save_name))
                else:
                    PIL.Image.fromarray(normalise_img(img_deprocess(input, img_mean, img_std,norm))).save(
                    os.path.join(save_intermediate_path, save_name))

            else:
                # if you install cv2 and ffmpeg, you can use save_video function which save preferred video as video format
                #save_name = '%05d.avi' % (t + 1)
                #save_video(normalise_vid(vid_deprocess(input, img_mean, img_std,norm)), save_name, save_intermediate_path, bgr,fr_rate = 10)
                save_name = '%05d.gif' % (t + 1)
                save_gif(normalise_vid(vid_deprocess(input, img_mean, img_std,norm)), save_name, save_intermediate_path,bgr, fr_rate = 150)

    # return input
    if len(input_size) == 3:
        return img_deprocess(input, img_mean, img_std, norm)
    else:
        return vid_deprocess(input, img_mean, img_std, norm)



