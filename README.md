# Extreme-Image-Compression Tensorflow Eager Implementation
An implementation of [Extreme Image Compression via Multiscale Autoencoders With Generative Adversarial Optimization](https://arxiv.org/abs/1904.03851) written in tensorflow.

## Requirements
 - Tensorflow Eager
 - PIL

## Training
In order to train, you'll have to do a few things...
 - Download a dataset of images (This compression algorithm is applied to almost all kinds of images)
 - run `python train.py` where train_dataset_dir is the directory containing your images

## Training Details
Since the network is sensitive to the image shape, the input of the network is either cropped or resized. These operations are set at teh function "_load_data". 

## Using Trained Network
In order to use trained weights you just need to run this command `python test.py`. By default, this will take every image from your test dataset, compute their output, and save it in the `sample/test` directory. 

## Results
<br />
Updates coming soon.......
<br />

| Original image | Shrunk image | EDSR Output |
| -------------- | ------------ | ----------- |
| ![alt-text](https://github.com/wikichao/Extreme-Image-Compression/blob/master/result/400.png "Original")          | ![alt-text](https://github.com/jmiller656/EDSR-Tensorflow/blob/master/results/input0.png "input")         | ![alt-text](https://github.com/jmiller656/EDSR-Tensorflow/blob/master/results/output0.png "shrunk")        |


## Future Work
- Train and post results on more datasets(either specific or general)

