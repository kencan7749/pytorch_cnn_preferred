# pytorch cnn_preferred

  Generate preferred image/video for the target units in arbitrary CNN model written in pytorch

| features[12]                                                 | features[14]                                                 | features[17]                                                 | features[21]                                                 | features[28]                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![preferred_img_layer_features[12]_channel_111](/Users/admin/Downloads/features[12]/preferred_img_layer_features[12]_channel_111.jpg) | ![preferred_img_layer_features[14]_channel_10](/Users/admin/Downloads/features[14]/preferred_img_layer_features[14]_channel_10.jpg) | ![preferred_img_layer_features[17]_channel_174](/Users/admin/Downloads/features[17]/preferred_img_layer_features[17]_channel_174.jpg) | ![preferred_img_layer_features[21]_channel_95](/Users/admin/Downloads/features[21]/preferred_img_layer_features[21]_channel_95.jpg) | ![preferred_img_layer_features[28]_channel_5](/Users/admin/Downloads/features[28]/preferred_img_layer_features[28]_channel_5.jpg) |
| ![preferred_img_layer_features[12]_channel_188](/Users/admin/Downloads/features[12]/preferred_img_layer_features[12]_channel_188.jpg) | ![preferred_img_layer_features[14]_channel_102](/Users/admin/Downloads/features[14]/preferred_img_layer_features[14]_channel_102.jpg) | ![preferred_img_layer_features[17]_channel_260](/Users/admin/Downloads/features[17]/preferred_img_layer_features[17]_channel_260.jpg) | ![preferred_img_layer_features[21]_channel_183](/Users/admin/Downloads/features[21]/preferred_img_layer_features[21]_channel_183.jpg) | ![preferred_img_layer_features[28]_channel_427](/Users/admin/Downloads/features[28]/preferred_img_layer_features[28]_channel_427.jpg) |
| ![preferred_img_layer_features[12]_channel_244](/Users/admin/Downloads/features[12]/preferred_img_layer_features[12]_channel_244.jpg) | ![preferred_img_layer_features[14]_channel_164](/Users/admin/Downloads/features[14]/preferred_img_layer_features[14]_channel_164.jpg) | ![preferred_img_layer_features[17]_channel_362](/Users/admin/Downloads/features[17]/preferred_img_layer_features[17]_channel_362.jpg) | ![preferred_img_layer_features[21]_channel_406](/Users/admin/Downloads/features[21]/preferred_img_layer_features[21]_channel_406.jpg) | ![preferred_img_layer_features[28]_channel_311](/Users/admin/Downloads/features[28]/preferred_img_layer_features[28]_channel_311.jpg) |

These are the preferred images of some channels in the layer of VGG16 trained on the ImageNet dataset.

## Description

  This repository contains Python codes for generating preferred image/video of the target units in a CNN model. The preferred image is based on the "activation maximum" method which generates images such that target unit(s) can have high activation value (vectors).

  Here, we can generate preferred image in arbitrary CNN model written in pytorch. I already tried  pretrained AlexNet, VGG19, ResNet50, Densenet, Inception-v3, provided by torchvision. I also tried 3D CNN optimized for video input such as C3D (), and 3D resnet (). Please try to your own CNN. Enjoy!

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

version 

### Copyright and license



### Author

Ken Shirakawa

Master student at Kamitani lab, Kyoto University (http://kamitani-lab.ist.i.kyoto-u.ac.jp)



### Acknowledgement





