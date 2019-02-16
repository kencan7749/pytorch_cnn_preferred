import torch
import torchvision
# import
import os
import pickle
import numpy as np
import PIL.Image
import cv2 

import scipy.io as sio
from datetime import datetime
import sys


sys.path.append('../model')
from resnet3d import resnet, mean
from C3D import C3D
                
sys.path.append('../icnn_torch')
from utils_upd import normalise_img, clip_extreme_pixel, save_video, normalise_vid, vid_preprocess,vid_deprocess, get_cnn_features
from icnn_lbfgs_upd import reconstruct_video_upd


#load model
net = resnet.resnet50(num_classes=400, shortcut_type='B', sample_size=112, sample_duration=90)
net = torch.nn.DataParallel(net, device_ids=None)
param_file = os.path.join('../model','resnet3d', 'resnet-50-kinetics.pth')
param = torch.load(param_file, map_location='cpu')
net.load_state_dict(param['state_dict'])
net.eval()

img_mean = np.array(mean.get_mean())
img_std = np.array(mean.get_std())
img_std = np.ones(3)
norm = 255


#save_dir
save_dir = '../result'
save_folder = 'jupyter_demo_torch_resnet3D_icnn_lbfgs'#__file__.split('.')[0]
save_folder = save_folder + '_' + datetime.now().strftime('%Y%m%dT%H%M%S')
save_path = os.path.join(save_dir,save_folder)
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture('v009_0050.mp4')

org_video = []
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        org_video.append(cv2.resize(frame, (112, 112)).astype(np.float32))

    else:
        cap.release()
        break
org_video = np.array(org_video)
# preprocessing (mean substruction)
org_vid = vid_preprocess(org_video, img_mean, img_std,norm=norm)

inputs = torch.Tensor(org_vid[np.newaxis])

layer_list = ['conv1', 'layer1[1].iden', 'layer4[2].iden']
#layer_list = ['layer1[1].iden']
exec_str_list = ["net.module."+layer +".register_forward_hook(hook)" for layer in layer_list]

features = get_cnn_features(net, inputs, exec_str_list)

feat_norm_list = np.array([np.linalg.norm(features[i].detach().numpy().astype(np.float)) for i in range(len(features))], dtype= np.float32)

# Use the inverse of the squared norm of the CNN features as the weight for each layer
weights = 1. / (feat_norm_list**2)
# Normalise the weights such that the sum of the weights = 1
weights = weights / weights.sum()

#layer_weight = dict(zip(layer_list, weights))
layer_weight= dict(zip(exec_str_list, weights))

#TODO:  Reconstrucion ------------------------
# Reconstruction options
opts = {
    # Loss function type: {'l2', 'l1', 'inner', 'gram'}
    'loss_type': 'l2',
    'img_mean': img_mean,
    'img_std' : img_std,

    # The maximum number of iterations
    'maxiter': 500,

    # Display the information on the terminal or not
    'disp': True,

    # Save the intermediate reconstruction or not
    'save_intermediate': True,
    # Save the intermediate reconstruction for every n iterations
    'save_intermediate_every': 1,
    # Path to the directory saving the intermediate reconstruction
    'save_intermediate_path': save_path,

    # A python dictionary consists of weight parameter of each layer in the
    # loss function, arranged in pairs of layer name (key) and weight (value);
    'layer_weight': layer_weight,

    # The initial image for the optimization (setting to None will use random
    # noise as initial image)
    'initial_input': None,#initial_video,

    # A python dictionary consists of channels to be selected, arranged in
    # pairs of layer name (key) and channel numbers (value); the channel
    # numbers of each layer are the channels to be used in the loss function;
    # use all the channels if some layer not in the dictionary; setting to None
    # for using all channels for all layers;
    'channel': None,
    
    'exec_code': exec_str_list,
    
    'bgr': False,
    'norm': norm,

    # A python dictionary consists of masks for the traget CNN features,
    # arranged in pairs of layer name (key) and mask (value); the mask selects
    # units for each layer to be used in the loss function (1: using the uint;
    # 0: excluding the unit); mask can be 3D or 2D numpy array; use all the
    # units if some layer not in the dictionary; setting to None for using all
    #units for all layers;
    'mask': None,
}

recon_mov, loss_list = reconstruct_video_upd(features, net, org_video.shape,**opts)