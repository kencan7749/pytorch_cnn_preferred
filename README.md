# pytorch cnn_preferred

  Generate preferred image/video for the target units in arbitrary CNN model written in pytorch

| features[12]                                                 | features[14]                                                 | features[17]                                                 | features[21]                                                 |                         features[28]                         | classifier[6]                                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | :----------------------------------------------------------: | ------------------------------------------------------------ |
| ![preferred_img_layer_features[12]_channel_111](image_gallery/preferred_img_layer_features[12]_channel_111.jpg) | ![preferred_img_layer_features[14]_channel_10](image_gallery/preferred_img_layer_features[14]_channel_10.jpg) | ![preferred_img_layer_features[17]_channel_174](image_gallery/preferred_img_layer_features[17]_channel_174.jpg) | ![preferred_img_layer_features[21]_channel_95](image_gallery/preferred_img_layer_features[21]_channel_95.jpg) | ![preferred_img_layer_features[28]_channel_5](image_gallery/preferred_img_layer_features[28]_channel_5.jpg) | ![preferred_img_layer_classifier[6]_channel_168](image_gallery/preferred_img_layer_classifier[6]_channel_168.jpg) |
| ![preferred_img_layer_features[12]_channel_188](image_gallery/preferred_img_layer_features[12]_channel_188.jpg) | ![preferred_img_layer_features[14]_channel_102](image_gallery/preferred_img_layer_features[14]_channel_102.jpg) | ![preferred_img_layer_features[17]_channel_260](image_gallery/preferred_img_layer_features[17]_channel_260.jpg) | ![preferred_img_layer_features[21]_channel_183](image_gallery/preferred_img_layer_features[21]_channel_183.jpg) | ![preferred_img_layer_features[28]_channel_311](image_gallery/preferred_img_layer_features[28]_channel_311.jpg) | ![preferred_img_layer_classifier[6]_channel_410](image_gallery/preferred_img_layer_classifier[6]_channel_410.jpg) |
| ![preferred_img_layer_features[12]_channel_244](image_gallery/preferred_img_layer_features[12]_channel_244.jpg) | ![preferred_img_layer_features[14]_channel_164](image_gallery/preferred_img_layer_features[14]_channel_164.jpg) | ![preferred_img_layer_features[17]_channel_362](image_gallery/preferred_img_layer_features[17]_channel_362.jpg) | ![preferred_img_layer_features[21]_channel_406](image_gallery/preferred_img_layer_features[21]_channel_406.jpg) | ![preferred_img_layer_features[28]_channel_427](image_gallery/preferred_img_layer_features[28]_channel_427.jpg) | ![preferred_img_layer_classifier[6]_channel_601](image_gallery/preferred_img_layer_classifier[6]_channel_601.jpg) |

These are the preferred images of some channels in the layer of VGG16 trained on ImageNet dataset.

## Description

  This repository contains Python codes for generating preferred image/video of the target units in a CNN model. The preferred image is based on the "activation maximum" method which generates images such that target unit(s) can have high activation value (vectors).

  Here, we can generate preferred image in arbitrary CNN model written in pytorch. I already tried  pretrained AlexNet, VGG19, ResNet50, Densenet121, Inception-v3, provided by torchvision. I also tried 3D CNN optimized for video input such as C3D ([pretrained on Sports1M dataset](https://github.com/DavideA/c3d-pytorch)

), and 3D ResNet50 inflated from 2D RGB model  ([pertained on Moments in time dataset](https://github.com/metalbubble/moments_models )



). Please try to your own CNN. Enjoy!

## Requirements

I checked this repository work correctly in this environment:

- Python 3.6
- Numpy
- Scipy
- PIL
- Pytorch 0.4.1
- Torchvison

## Usage

The key component is extracting intemediate layer activation. See `example/Instruct_extracting_intermediate_feature.ipynb` and try to run it.

Generating preferred image code is at `example/preferred_image_shortest_demo.ipynb`. You can generate preferred image without concerning any parameters. If you consider the parameters carefully, check `example/preferred_image_demo_simpleCNN_conv` or `example/preferred_image_demo_complexCNN_conv.ipynb`.



## Image Gallery

Here area the example results of  variaous network. First one is preferred image of maximized channel. Second one is that of maximized unit.

### Maximimize channel

- ResNet50 (pretrained on ImageNet)

| layer3[5].conv3                                              | layer4[0].conv1                                              | layer4[2].conv3                                              | fc                                                           |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![preferred_img_layer_layer3[5].conv3_channel_422](image_gallery/preferred_img_layer_layer3[5].conv3_channel_422.jpg) | ![preferred_img_layer_layer4[0].conv1_channel_315](image_gallery/preferred_img_layer_layer4[0].conv1_channel_315.jpg) | ![preferred_img_layer_layer4[2].conv3_channel_1192](image_gallery/preferred_img_layer_layer4[2].conv3_channel_1192.jpg) | ![preferred_img_layer_fc_channel_244](image_gallery/preferred_img_layer_fc_channel_244.jpg) |
| ![preferred_img_layer_layer3[5].conv3_channel_471](image_gallery/preferred_img_layer_layer3[5].conv3_channel_471.jpg) | ![preferred_img_layer_layer4[0].conv1_channel_363](image_gallery/preferred_img_layer_layer4[0].conv1_channel_363.jpg) | ![preferred_img_layer_layer4[2].conv3_channel_1305](image_gallery/preferred_img_layer_layer4[2].conv3_channel_1305.jpg) | ![preferred_img_layer_fc_channel_931](image_gallery/preferred_img_layer_fc_channel_931.jpg) |



- DenseNet121 (pretrained on ImageNet)

| db2.dl12.conv2                                               | db3.dl16.conv2                                               | db4.dl16.conv2                                               | classifier                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![preferred_img_layer_features.denseblock2.denselayer12.conv2_channel_4](image_gallery/preferred_img_layer_features.denseblock2.denselayer12.conv2_channel_4.jpg) | ![preferred_img_layer_features.denseblock3.denselayer16.conv2_channel_6](image_gallery/preferred_img_layer_features.denseblock3.denselayer16.conv2_channel_6.jpg) | ![preferred_img_layer_features.denseblock4.denselayer16.conv2_channel_24](image_gallery/preferred_img_layer_features.denseblock4.denselayer16.conv2_channel_24.jpg) | ![preferred_img_layer_classifier_channel_637](image_gallery/preferred_img_layer_classifier_channel_637.jpg) |
| ![preferred_img_layer_features.denseblock2.denselayer12.conv2_channel_7](image_gallery/preferred_img_layer_features.denseblock2.denselayer12.conv2_channel_7.jpg) | ![preferred_img_layer_features.denseblock3.denselayer16.conv2_channel_25](image_gallery/preferred_img_layer_features.denseblock3.denselayer16.conv2_channel_25.jpg) | ![preferred_img_layer_features.denseblock4.denselayer16.conv2_channel_26](image_gallery/preferred_img_layer_features.denseblock4.denselayer16.conv2_channel_26.jpg) | ![preferred_img_layer_classifier_channel_225](image_gallery/preferred_img_layer_classifier_channel_225.jpg) |

(db… denseblock, ds… denselayer)

- Inception-v3 (pretrained on ImageNet)

| Mx_5d.bc5x5_2.conv                                           | Mx_6a_bc3x3dbl_3.conv                                        | Mx_7b_bc_pool                                                | fc                                                           |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![preferred_img_layer_Mixed_5d.branch5x5_2.conv_channel_17](image_gallery/preferred_img_layer_Mixed_5d.branch5x5_2.conv_channel_17.jpg) | ![preferred_img_layer_Mixed_6a.branch3x3dbl_3.conv_channel_59](image_gallery/preferred_img_layer_Mixed_6a.branch3x3dbl_3.conv_channel_59.jpg) | ![preferred_img_layer_Mixed_7b.branch_pool.conv_channel_127](image_gallery/preferred_img_layer_Mixed_7b.branch_pool.conv_channel_127.jpg) | ![preferred_img_layer_fc_channel_179](image_gallery/preferred_img_layer_fc_channel_179.jpg) |
| ![preferred_img_layer_Mixed_5d.branch5x5_2.conv_channel_40](image_gallery/preferred_img_layer_Mixed_5d.branch5x5_2.conv_channel_40.jpg) | ![preferred_img_layer_Mixed_6a.branch3x3dbl_3.conv_channel_92](image_gallery/preferred_img_layer_Mixed_6a.branch3x3dbl_3.conv_channel_92.jpg) | ![preferred_img_layer_Mixed_7b.branch_pool.conv_channel_74](image_gallery/preferred_img_layer_Mixed_7b.branch_pool.conv_channel_74.jpg) | ![preferred_img_layer_fc_channel_448](image_gallery/preferred_img_layer_fc_channel_448.jpg) |

(Mx… Mixed, bc… branch)

- 3D ResNet50 (pretrained on Moments in Time dataset)

  These results are preferred video of 16 frames input (shown as gif). Note that this model trained on 16 frame inputs at 5 fps.

| layer2[3].conv3 | layer3[5].conv3 | layer4[2].conv3 | fc (last_linear) |
| --------------- | --------------- | --------------- | ---------------- |
|                 |                 |                 |                  |
|                 |                 |                 |                  |



### Maximize unit

- Resnet50

|      |      |      |
| ---- | ---- | ---- |
|      |      |      |
|      |      |      |
|      |      |      |



- DenseNet121

|      |      |      |
| ---- | ---- | ---- |
|      |      |      |
|      |      |      |
|      |      |      |



- Inception-v3

|      |      |      |
| ---- | ---- | ---- |
|      |      |      |
|      |      |      |
|      |      |      |

### Version

version 1.0 #2019/03/03

### Copyright and license



### Author

Ken Shirakawa

Master student at Kamitani lab, Kyoto University (http://kamitani-lab.ist.i.kyoto-u.ac.jp)



### Acknowledgement





