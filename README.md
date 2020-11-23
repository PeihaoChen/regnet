# Generating Visually Aligned Sound from Videos

This is the official pytorch implementation of the TIP paper "[Generating Visually Aligned Sound from Videos][REGNET]" and the corresponding Visually Aligned Sound (VAS) dataset. 

Demo videos containing sound generation results can be found [here][demo].

![](https://github.com/PeihaoChen/regnet/blob/master/overview.png)


# Contents
----

* [Usage Guide](#usage-guide)
   * [Getting Started](#getting-started)
      * [Installation](#installation)
      * [Download Datasets](#download-datasets)
      * [Data Preprocessing](#data-preprocessing)
   * [Training REGNET](#training-regnet)
   * [Generating Sound](#generating-sound)
* [Other Info](#other-info)
   * [Citation](#citation)
   * [Contact](#contact)


----
# Usage Guide

## Getting Started
[[back to top](#Generating-Visually-Aligned-Sound-from-Videos)]

### Installation

Clone this repository into a directory. We refer to that directory as *`REGNET_ROOT`*.

```bash
git clone https://github.com/PeihaoChen/regnet
cd regnet
```
Create a new Conda environment.
```bash
conda create -n regnet python=3.7.1
conda activate regnet
```
Install [PyTorch][pytorch] and other dependencies.
```bash
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0
conda install ffmpeg -n regnet -c conda-forge
pip install -r requirements.txt
```

### Download Datasets

In our paper, we collect 8 sound types (Dog, Fireworks, Drum, Baby form [VEGAS][vegas] and Gun, Sneeze, Cough, Hammer from [AudioSet][audioset]) to build our [Visually Aligned Sound (VAS)][VAS] dataset.
Please first download VAS dataset and unzip the data to *`$REGNET_ROOT/data/`*  folder.

For each sound type in AudioSet, we download all videos from Youtube and clean data on Amazon Mechanical Turk (AMT) using the same way as [VEGAS][visual_to_sound].


```bash
unzip ./data/VAS.zip -d ./data
```



### Data Preprocessing

Run `data_preprocess.sh` to preprocess data and extract RGB and optical flow features. 

Notice: The script we provided to calculate optical flow is easy to run but is resource-consuming and will take a long time. We strongly recommend you to refer to [TSN repository][TSN] and their built [docker image][TSN_docker] (our paper also uses this solution)  to speed up optical flow extraction and to restrictly reproduce the results.
```bash
source data_preprocess.sh
```


## Training REGNET

Training the REGNET from scratch. The results will be saved to `ckpt/dog`.

```bash
CUDA_VISIBLE_DEVICES=7 python train.py \
save_dir ckpt/dog \
auxiliary_dim 32 \ 
rgb_feature_dir data/features/dog/feature_rgb_bninception_dim1024_21.5fps \
flow_feature_dir data/features/dog/feature_flow_bninception_dim1024_21.5fps \
mel_dir data/features/dog/melspec_10s_22050hz \
checkpoint_path ''
```

In case that the program stops unexpectedly, you can continue training.
```bash
CUDA_VISIBLE_DEVICES=7 python train.py \
-c ckpt/dog/opts.yml \
checkpoint_path ckpt/dog/checkpoint_018081
```

## Generating Sound


During inference, our RegNet will generate visually aligned spectrogram, and then use [WaveNet][wavenet] as vocoder to generate waveform from spectrogram. You should first download our trained WaveNet model for different sound categories (
[Dog](https://github.com/PeihaoChen/regnet/releases/download/WaveNet_model/dog_checkpoint_step000200000_ema.pth),
[Fireworks](https://github.com/PeihaoChen/regnet/releases/download/WaveNet_model/fireworks_checkpoint_step000267000_ema.pth),
[Drum](https://github.com/PeihaoChen/regnet/releases/download/WaveNet_model/drum_checkpoint_step000160000_ema.pth),
[Baby](https://github.com/PeihaoChen/regnet/releases/download/WaveNet_model/baby_checkpoint_step000470000_ema.pth),
[Gun](https://github.com/PeihaoChen/regnet/releases/download/WaveNet_model/gun_checkpoint_step000152000_ema.pth),
[Sneeze](https://github.com/PeihaoChen/regnet/releases/download/WaveNet_model/sneeze_checkpoint_step000071000_ema.pth),
[Cough](https://github.com/PeihaoChen/regnet/releases/download/WaveNet_model/cough_checkpoint_step000079000_ema.pth),
[Hammer](https://github.com/PeihaoChen/regnet/releases/download/WaveNet_model/hammer_checkpoint_step000137000_ema.pth)
). 

The generated spectrogram and waveform will be saved at `ckpt/dog/inference_result`
```bash
CUDA_VISIBLE_DEVICES=7 python test.py \
-c ckpt/dog/opts.yml \ 
aux_zero True \ 
checkpoint_path ckpt/dog/checkpoint_041000 \ 
save_dir ckpt/dog/inference_result \
wavenet_path /path/to/wavenet_dog.pth
```

If you want to train your own WaveNet model, you can use [WaveNet repository][wavenet_repository].
```bash
git clone https://github.com/r9y9/wavenet_vocoder && cd wavenet_vocoder
git checkout 2092a64
```

## Pre-trained Models
You can also use our pre-trained RegNet for generating visually aligned sounds.

First, download and unzip our pre-trained RegNet ([Dog](https://github.com/PeihaoChen/regnet/releases/download/Pretrained_RegNet/RegNet_dog_checkpoint_041000.tar)) to `./ckpt/dog` folder.
```bash
tar -xvf ./ckpt/dog/RegNet_dog_checkpoint_041000.tar # unzip
```


Second, run the inference code.
```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
-c config/dog_opts.yml \ 
aux_zero True \ 
checkpoint_path ckpt/dog/checkpoint_041000 \ 
save_dir ckpt/dog/inference_result \
wavenet_path /path/to/wavenet_dog.pth
```

Enjoy your experiments!


# Other Info
[[back to top](#Generating-Visually-Aligned-Sound-from-Videos)]

## Citation


Please cite the following paper if you feel REGNET useful to your research
```
@Article{chen2020regnet,
  author  = {Peihao Chen, Yang Zhang, Mingkui Tan, Hongdong Xiao, Deng Huang and Chuang Gan},
  title   = {Generating Visually Aligned Sound from Videos},
  journal = {TIP},
  year    = {2020},
}
```

## Contact
For any question, please file an issue or contact
```
Peihao Chen: phchencs@gmail.com
Hongdong Xiao: xiaohongdonghd@gmail.com
```

[REGNET]:https://arxiv.org/abs/2008.00820
[audioset]:https://research.google.com/audioset/index.html
[VEGAS_link]:http://bvision11.cs.unc.edu/bigpen/yipin/visual2sound_webpage/VEGAS.zip
[pytorch]:https://github.com/pytorch/pytorch
[wavenet]:https://arxiv.org/abs/1609.03499
[wavenet_repository]:https://github.com/r9y9/wavenet_vocoder
[opencv]:https://github.com/opencv/opencv
[dense_flow]:https://github.com/yjxiong/dense_flow
[VEGAS]: http://bvision11.cs.unc.edu/bigpen/yipin/visual2sound_webpage/visual2sound.html
[visual_to_sound]: https://arxiv.org/abs/1712.01393
[TSN]: https://github.com/yjxiong/temporal-segment-networks
[VAS]: https://drive.google.com/file/d/14birixmH7vwIWKxCHI0MIWCcZyohF59g/view?usp=sharing
[TSN_docker]: https://hub.docker.com/r/bitxiong/tsn/tags
[demo]: https://youtu.be/fI_h5mZG7bg
