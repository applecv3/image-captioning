# Image Captioning using EfficientNet and Transformer
Implemented on COCO 2014 dataset: https://cocodataset.org/#download

Used EfficientNet to extract features from the images and Transformer(Decoder) to generate the captions.
As for EfficientNet, I slightly modified this tensorflow implementation code: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet


Here's the architecture. It's pretty simple and intuitive.

![Image of Yaktocat](https://github.com/applecv3/image-captioning/blob/master/pictures/Screenshot%20from%202021-02-27%2000-12-26.png)

# How to Use
First of all, you might want to create TFrecord files with create_tfrecord.py

Second, train the model through train.py

Third, freeze and evaluate the model with Beam Search algorithm excuting freeze.py and test.py

Last, the images with caption implementing inference.py

# Examples
