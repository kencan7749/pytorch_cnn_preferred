'''Demonstration code for icnn_gd

This script will do the followings:

1. extract cnn features from a test image,
2. reconstruct the test image from the CNN features.
'''
import os
import sys
import pickle
from datetime import datetime

import numpy as np
import cv2

import torch

sys.path.append('../')
from icnn_torch.icnn_lbfgs import reconstruct_video
from icnn_torch.utils import vid_preprocess

sys.path.append('./model')

from C3D import C3D_conv


# Load the average image of ImageNet
img_mean_file = './data/ilsvrc_2012_mean.npy'
img_mean = np.load(img_mean_file)
img_mean = np.float32([img_mean[0].mean(), img_mean[1].mean(), img_mean[2].mean()])
#img_mean = np.float32(np.zeros(3))

# Load CNN model
# This is better for reconstructon since all feature can extract like caffe manner

model = C3D_conv()
weight_path = torch.load('/home/shirakawa/movie/c3d-pytorch-master/c3d.pickle')
model.load_state_dict(weight_path)
model.eval()

# Layer list
# Example: layer_list = ['conv1_1','conv2_1','conv3_1']

# Use all conv and fc layers
layer_list = [layer[0]
              for layer in model.named_children()
              if 'conv' in layer[0] ]
#layer_list = [layer[0]
#              for layer in model.named_children()
#              if 'conv' in layer[0]]
layer_list = layer_list[4:]
print(layer_list)

# Setup directories -----------------------------------------------------------
# Make directory for saving the results
save_dir = './result_first_attempt'
save_subdir = __file__.split('.')[0] + '_' + datetime.now().strftime('%Y%m%dT%H%M%S')
save_path = os.path.join(save_dir, save_subdir)
print(save_path)
os.makedirs(save_path)

# Setup the test image and image features -------------------------------------

# Test video ---------------
cap = cv2.VideoCapture('/home/shirakawa/movie/data/mov_10_IJ/file_frame16/0/IJ_vol0001_fs16_run01.avi')
cap = cv2.VideoCapture('/home/shirakawa/movie/data/mov_10_IJ/file_frame16/0/IJ_vol0502_fs16_run02.avi')
cap = cv2.VideoCapture('/home/shirakawa/movie/data/mov_10_IJ/file_frame16/0/IJ_vol1507_fs16_run06.avi')
cap = cv2.VideoCapture('/home/shirakawa/movie/data/mov_10_IJ/file_frame16/0/IJ_vol1525_fs16_run06.avi')
cap = cv2.VideoCapture('/home/shirakawa/movie/data/contents_shared/MITtest_v1/source/v181_0031.mp4')
cap = cv2.VideoCapture('/home/shirakawa/movie/data/contents_shared/MITtest_v1/source/v005_0057.mp4')
#cap = cv2.VideoCapture('/home/shirakawa/movie/data/mov_10_BM/file_frame16/0/BM_vol0001_fs16_run01.avi')
#cap = cv2.VideoCapture('/home/shirakawa/movie/data/mov_10_BM/file_frame16/0/BM_vol0502_fs16_run02.avi')
#cap = cv2.VideoCapture('/home/shirakawa/movie/data/mov_10_BM/file_frame16/0/BM_vol0557_fs16_run02.avi')

#cap = cv2.VideoCapture('/home/shirakawa/movie/data/MIT_exp/test/calling/v207_0026.mp4')

org_video = []
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        org_video.append(cv2.resize(frame, (64, 64)).astype(np.float32))

    else:
        cap.release()
        break
org_video = np.array(org_video)
# preprocessing (mean substruction)
org_vid = vid_preprocess(org_video, img_mean)

inputs = torch.Tensor(org_vid[np.newaxis,])
# Extract CNN features from the test video
features = model(inputs, layer_list)


#TODO: Save the test image
fr, height, width, ch = org_video.shape
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = cv2.VideoWriter(os.path.join(save_path,'sample.avi'),fourcc,30,(width,height))
for j in range(fr):
    writer.write(org_video[j].astype(np.uint8))
writer.release()

#TODO:  Setup layer weights (optional) ----------------------------------------------
# Norm of the CNN features for each layer
try:
    feat_norm_list = np.array([np.linalg.norm(features[layer].detach().numpy()) for layer in layer_list],
                          dtype='float32')
except:
    features = dict(zip(features.keys(), [torch.Tensor(features[layer]) for layer in features.keys()]))
    feat_norm_list = np.array([np.linalg.norm(features[layer].detach().numpy()) for layer in layer_list],
                              dtype='float32')

# Use the inverse of the squared norm of the CNN features as the weight for each layer
weights = 1. / (feat_norm_list**2)

# Normalise the weights such that the sum of the weights = 1
weights = weights / weights.sum()

layer_weight = dict(zip(layer_list, weights))

initial_image = np.random.randint(0,255,(112,112,3))
initial_video = np.tile(initial_image[...,np.newaxis],16)
initial_video = initial_video.transpose(3,0,1,2)


#TODO:  Reconstrucion ------------------------
# Reconstruction options
opts = {
    # Loss function type: {'l2', 'l1', 'inner', 'gram'}
    'loss_type': 'l2',

    # The maximum number of iterations
    'maxiter': 500,

    # Display the information on the terminal or not
    'disp': True,

    # Save the intermediate reconstruction or not
    'save_intermediate': True,
    # Save the intermediate reconstruction for every n iterations
    'save_intermediate_every': 10,
    # Path to the directory saving the intermediate reconstruction
    'save_intermediate_path': save_path,

    # A python dictionary consists of weight parameter of each layer in the
    # loss function, arranged in pairs of layer name (key) and weight (value);
    'layer_weight': layer_weight,

    # The initial image for the optimization (setting to None will use random
    # noise as initial image)
    'initial_video': None,#initial_video,

    # A python dictionary consists of channels to be selected, arranged in
    # pairs of layer name (key) and channel numbers (value); the channel
    # numbers of each layer are the channels to be used in the loss function;
    # use all the channels if some layer not in the dictionary; setting to None
    # for using all channels for all layers;
    'channel': None,

    # A python dictionary consists of masks for the traget CNN features,
    # arranged in pairs of layer name (key) and mask (value); the mask selects
    # units for each layer to be used in the loss function (1: using the uint;
    # 0: excluding the unit); mask can be 3D or 2D numpy array; use all the
    # units if some layer not in the dictionary; setting to None for using all
    #units for all layers;
    'mask': None,
}

# Save the optional parameters
save_name = 'options.pkl'
with open(os.path.join(save_path, save_name), 'wb') as f:
    pickle.dump(opts, f)





recon_mov, loss_list = reconstruct_video(features, model, org_video.shape, img_mean,layer_list,**opts)


results = {
    # the image reconstracted after iter_n iterations
    'raw_video': recon_mov,

    # The total number of iterations for gradient descend
    'iter_n': 200,

    # loss list
    'loss_list': loss_list,
    'layer_list': layer_list

}

save_name = 'results.pkl'
with open(os.path.join(save_path, save_name), 'wb') as f:
    pickle.dump(results, f)