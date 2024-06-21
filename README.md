# [Seg-LSTM: Performance of xLSTM for Semantic Segmentation of Remotely Sensed Images](https://arxiv.org/abs/2406.14086)

This is the official code repository for "Seg-LSTM: Performance of xLSTM for Semantic Segmentation of Remotely Sensed Images". {[Arxiv Paper](https://arxiv.org/abs/2406.14086)}

![架构](https://github.com/zhuqinfeng1999/Seg-LSTM/assets/34743935/85199e03-dfb3-4410-aa6b-663526fa7b3c)


[![GitHub stars](https://badgen.net/github/stars/zhuqinfeng1999/Seg-LSTM)](https://github.com//zhuqinfeng1999/Seg-LSTM)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2406.14086-b31b1b.svg)](https://arxiv.org/abs/2406.14086)

## Abstract

Recent advancements in autoregressive networks with linear complexity have driven significant research progress, demonstrating exceptional performance in large language models. A representative model is the Extended Long Short-Term Memory (xLSTM), which incorporates gating mechanisms and memory structures, performing comparably to Transformer architectures in long-sequence language tasks. Autoregressive networks such as xLSTM can utilize image serialization to extend their application to visual tasks such as classification and segmentation. Although existing studies have demonstrated Vision-LSTM’s impressive results in image classification, its performance in image semantic segmentation remains unverified. Our study represents the first attempt to evaluate the effectiveness of Vision-LSTM in the semantic segmentation of remotely sensed images. This evaluation is based on a specifically designed encoder-decoder architecture named Seg-LSTM, and comparisons with state-of-the-art segmentation networks. Our study found that Vision-LSTM's performance in semantic segmentation was limited and generally inferior to Vision-Transformers-based and Vision-Mamba-based models in most comparative tests. Future research directions for enhancing Vision-LSTM are recommended. The source code is available from https://github.com/zhuqinfeng1999/Seg-LSTM.

## Installation

### Requirements

Requirements: Ubuntu 20.04, CUDA 12.4

* Set up the mmsegmentation environment; we conduct experiments using the mmsegmentation framework. Please refer to https://github.com/open-mmlab/mmsegmentation.


#### LoveDA datasets

* The LoveDA datasets can be found here https://github.com/Junjue-Wang/LoveDA.

* After downloading the dataset, you are supposed to put them into '/mmsegmentation/data/loveDA/'

* '/mmsegmentation/data/loveDA/'
- ann_dir
  - train
    - .png
  - val
    - .png
- img_dir
  - train
    - .png
  - val
    - .png

#### Model file and config file

- The model file zqf_seglstm.py can be found in /mmsegmentation/mmseg/models/backbones/

- The config file zqf_seglstm_'decoder'.py for the combination of backbone and decoder head can be found in /mmsegmentation/configs/_base_/models

- The config file for training can be found in /mmsegmentation/configs/zqf_seglstm/

## Training Seg-LSTM

`bash tools/dist_train.sh 'configfile' 2 --work-dir /mmsegmentation/output/seglstm`


## Testing Seg-LSTM

`bash tools/dist_test.sh 'configfile' \ /mmsegmentation/output/seglstm/iter_15000.pth 2 --out /mmsegmentation/visout/seglstm`


## Citation

If you find this work useful in your research, please consider cite:

```
@article{
zhu2024seglstm,
    title={Seg-LSTM: Performance of xLSTM for Semantic Segmentation of Remotely Sensed Images},
    author={Qinfeng Zhu and Yuanzhi Cai and Lei Fan},
    year={2024},
    eprint={2406.14086},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

We acknowledge all the authors of the employed public datasets, allowing the community to use these valuable resources for research purposes. We also thank the authors of [xLSTM](https://github.com/NX-AI/xlstm), [Vision-LSTM](https://github.com/NX-AI/vision-lstm), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) for making their valuable code publicly available.
