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

  Here, we can generate preferred image in arbitrary CNN model written in pytorch. I already tried  pretrained AlexNet, VGG19, ResNet50, Densenet, Inception-v3, provided by torchvision. I also tried 3D CNN optimized for video input such as C3D (

[pretrained on Sports1M dataset]: https://github.com/DavideA/c3d-pytorch

), and 3D resnet (

[pertained on Moments in time dataset]: https://github.com/metalbubble/moments_models

). Please try to your own CNN. Enjoy!

## Requirements

- Python 3.6

- Numpy
- Scipy
- PIL
- Pytorch
- Torchvison

## Usage

The key point is extracting intemediate layer activation. See `example/Instruct_extracting_intermediate_feature.ipynb` and try to run it.

Generating preferred image code is at `example/preferred_image_shortest_demo.ipynb`. You can generate preferred image without concerning any parameters. If you consider the parameters carefully, check `example/preferred_image_demo_simpleCNN_conv` or `example/preferred_image_demo_complexCNN_conv.ipynb`.

### Version

version 1.0 #2019/03/03

### Copyright and license



### Author

Ken Shirakawa

Master student at Kamitani lab, Kyoto University (http://kamitani-lab.ist.i.kyoto-u.ac.jp)



### Acknowledgement





