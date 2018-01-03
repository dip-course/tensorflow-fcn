# tensorflow-fcn
This is a one file Tensorflow implementation of [Fully Convolutional Networks](http://arxiv.org/abs/1411.4038) in Tensorflow. The code can easily be integrated in your semantic segmentation pipeline. The network can be applied directly or finetuned to perform semantic segmentation using tensorflow training code.

Deconvolution Layers are initialized as bilinear upsampling. Conv and FCN layer weights using VGG weights. Numpy load is used to read VGG weights. No Caffe or Caffe-Tensorflow is required to run this. **The .npy file for [VGG16] to be downloaded before using this needwork**. You can find the file here: ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy

No Pascal VOC finetuning was applied to the weights. The model is meant to be finetuned on your own data. The model can be applied to an image directly (see `test_fcn.py`) but the result will be rather coarse.

## Requirements

In addition to tensorflow the following packages are required:

numpy
scipy
pillow
matplotlib

Those packages can be installed by running `pip install -r requirements.txt` or `pip install numpy scipy pillow matplotlib`.

### Tensorflow 1.0rc

This code requires `Tensorflow Version >= 1.0rc` to run. If you want to use older Version you can try using commit `bf9400c6303826e1c25bf09a3b032e51cef57e3b`. This Commit has been tested using the pip version of `0.12`, `0.11` and `0.10`.

Tensorflow 1.0 comes with a large number of breaking api changes. If you are currently running an older tensorflow version, I would suggest creating a new `virtualenv` and install 1.0rc using:

```bash
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.0rc0-cp27-none-linux_x86_64.whl
pip install --upgrade $TF_BINARY_URL
```

Above commands will install the linux version with gpu support. For other versions follow the instructions [here](https://www.tensorflow.org/versions/r1.0/get_started/os_setup).

## Usage

`python test_fcn.py` to test the implementation.

## Predecessors

Weights were generated using [Caffe to Tensorflow](https://github.com/ethereon/caffe-tensorflow). The VGG implementation is based on [tensorflow-vgg16](https://github.com/ry/tensorflow-vgg16) and numpy loading is based on [tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg). You do not need any of the above cited code to run the model, not do you need caffe.

## Install

Installing matplotlib from pip requires the following packages to be installed `libpng-dev`, `libjpeg8-dev`, `libfreetype6-dev` and `pkg-config`. On Debian, Linux Mint and Ubuntu Systems type:

`sudo apt-get install libpng-dev libjpeg8-dev libfreetype6-dev pkg-config` <br>
`pip install -r requirements.txt`
