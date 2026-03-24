# vocal-remover

This is a deep-learning-based tool to extract instrumental track from your songs.

## Installation

### Install PyTorch
**See**: [GET STARTED](https://pytorch.org/get-started/locally/)

### Install the other packages
```
cd vocal-remover
pip install -r requirements.txt
```

## Usage
The following command separates the input into instrumental and vocal tracks. They are saved as `*_Instruments.wav` and `*_Vocals.wav`.

### Run on CPU
```
python inference.py --input path/to/an/audio/file -P path/to/an/model/weight
```

### Run on GPU
```
python inference.py --input path/to/an/audio/file -P path/to/an/model/weight --gpu 0
```

## Train your own model

### Place your dataset
```
path/to/dataset/
  +- instruments/
  |    +- aaa.wav
  |    +- bbb.wav
  |    +- ...
  +- mixtures/
       +- aaa.wav
       +- bbb.wav
       +- ...
```

### Train a model
```
# train a mono model 训练单声道模型
python train.py --dataset path/to/dataset --gpu 0 --mono --exp_name your_exp_name
# train a stereo model 训练立体声模型
python train.py --dataset path/to/dataset --gpu 0 --exp_name your_exp_name
```

### Export the model
```
# requirements: PyTorch >= 2.1.0
pip install onnx onnxsim
python export.py path/to/model.pt path/to/model.onnx
```

## References
- [1] Jansson et al., "Singing Voice Separation with Deep U-Net Convolutional Networks", https://ejhumphrey.com/assets/pdf/jansson2017singing.pdf
- [2] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf
- [3] Takahashi et al., "MMDENSELSTM: AN EFFICIENT COMBINATION OF CONVOLUTIONAL AND RECURRENT NEURAL NETWORKS FOR AUDIO SOURCE SEPARATION", https://arxiv.org/pdf/1805.02410.pdf
- [4] Choi et al., "PHASE-AWARE SPEECH ENHANCEMENT WITH DEEP COMPLEX U-NET", https://openreview.net/pdf?id=SkeRTsAcYm
- [5] Jansson et al., "Learned complex masks for multi-instrument source separation", https://arxiv.org/pdf/2103.12864.pdf
- [6] Liutkus et al., "The 2016 Signal Separation Evaluation Campaign", Latent Variable Analysis and Signal Separation - 12th International Conference
