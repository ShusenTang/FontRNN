# FontRNN

This repository contains the implementation in tensorflow of FontRNN described in our paper FontRNN: Generating Large-scale Chinese Fonts via Recurrent Neural Network (accepted to Computer Graphics Forum (proc. of Pacific Graphics 2019)).

## Overview of Paper
Despite the recent impressive development of deep neural networks, using deep learning based methods to generate large-scale Chinese fonts is still a rather challenging task due to the huge number of intricate Chinese glyphs, e.g., the official standard Chinese charset GB18030-2000 consists of 27,533 Chinese characters. Until now, most existing models for this task adopt Convolutional Neural Networks (CNNs) to generate bitmap images of Chinese characters due to CNN based models’ remarkable success in various application fields. However, CNN based models focus more on image-level features while usually ignore stroke order information when writing characters. Instead, we treat Chinese characters as sequences of points (i.e., writing trajectories) and propose to handle this task via an effective Recurrent Neural Network (RNN) model with monotonic attention mechanism, which can learn from as few as hundreds of training samples and then synthesize glyphs for remaining thousands of characters with the same style. Experiments show that our proposed FontRNN can be used for synthesizing large-scale Chinese fonts as well as generating realistic Chinese handwritings efficiently.

## Quickstart
### code
The code of FontRNN that contains model defination (model.py), training script (train.py) and testing jupyter notebook (test.ipynb). 
> Some scripts are borrowed from [Sketch-RNN](https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn).

training example:
``` shell

CUDA_VISIBLE_DEVICES=0 python train.py -hparams="log_root=../log/demo, batch_size=128, attention_method=LM"
```

testing:

Make sure training was done before testing. We've provided a simple jupyter notebook (test.ipynb) to show you how to load a trained model and generate the generate results.


### data
For copyright reasons, we only provide one font data (FZTLJW.npz) for research (commercial use prohibited), that contains three parts of train, validation and test sets. The train set contains 775 samples described in paper, the remaining samples are randomly divided into validation and test sets.

## Requirements
* python=3.6
* tensorflow-gpu=1.11.0
* tensorboard

## Citation
If you use this code or data in your research, please cite us as follows:
``` shell
# TODO
```


