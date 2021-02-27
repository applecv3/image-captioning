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

Last, get the images with captions implementing inference.py

# Examples
Its performance was measured on Beam Search with k=3 and here are some generated examples by the trained model.

![Image of Yaktocat](https://github.com/applecv3/image-captioning/blob/master/pictures/Screenshot%20from%202021-02-27%2005-13-09.png)

![Image of Yaktocat](https://github.com/applecv3/image-captioning/blob/master/pictures/example1.jpg)
![Image of Yaktocat](https://github.com/applecv3/image-captioning/blob/master/pictures/example2.jpg)
![Image of Yaktocat](https://github.com/applecv3/image-captioning/blob/master/pictures/example3.jpg)
![Image of Yaktocat](https://github.com/applecv3/image-captioning/blob/master/pictures/example4.jpg)
![Image of Yaktocat](https://github.com/applecv3/image-captioning/blob/master/pictures/example5.jpg)
![Image of Yaktocat](https://github.com/applecv3/image-captioning/blob/master/pictures/example6.jpg)

# References
[1] Attention Is All You Need
https://arxiv.org/abs/1706.03762

[2] EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
https://arxiv.org/abs/1905.11946
