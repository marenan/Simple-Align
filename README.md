# CVPR2024-Navigating Beyond Dropout: An Intriguing Solution towards Generalizable Image Super-Resolution

## Abstract

Deep learning has led to a dramatic leap on Single Image Super-Resolution (SISR) performances in recent years. While most existing work assumes a simple and fixed degradation model (e.g., bicubic downsampling), the research of Blind SR seeks to improve model generalization ability with unknown degradation. Recently, Kong et al. pioneer the investigation of a more suitable training strategy for Blind SR using Dropout. Although such method indeed brings substantial generalization improvements via mitigating overfitting, we argue that Dropout simultaneously introduces undesirable side-effect that compromises model's capacity to faithfully reconstruct fine details. We show both the theoretical and experimental analyses in our paper, and furthermore, we present another easy yet effective training strategy that enhances the generalization ability of the model by simply modulating its first and second-order features statistics. Experimental results have shown that our method could serve as a model-agnostic regularization and outperforms Dropout on seven benchmark datasets including both synthetic and real-world scenarios. 
## Installation

1. Install dependent packages 
- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.7.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install basicsr`
- [option] Python packages: [`pip install tensorboardX`](https://github.com/lanpa/tensorboardX), for visualizing curves.


1. Download the testing datasets (Set5, Set14, B100, Manga109, Urban100) and move them to `./dataset/benchmark`.
[Google Drive](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u) or [Baidu Drive](https://pan.baidu.com/s/1AZDcEAFwwc1OC3KCd7EDnQ) (Password: basr) .

2. Add degradations to testing datasets.
```
cd ./dataset
python add_degradations.py
```

3. Download [pretrained models](https://drive.google.com/drive/folders/1NcNHbsGtD0OHuAf_ATACmZ_cTikL7bB3?usp=sharing) and move them to  `./pretrained_models/` folder. 

   To remain the setting of Real-ESRGAN, we use the GT USM (sharpness) in the paper. But we also provide the models without USM, the improvement is basically same.

4. Run the testing commands.
```
CUDA_VISIBLE_DEVICES=1 python realesrgan/test.py -opt options/test/test_realsrresnet.yml
CUDA_VISIBLE_DEVICES=1 python realesrgan/test.py -opt options/test/test_realsrresnet_reg.yml
```
6. The output results will be sorted in `./results`. 

### How to train Real-SRResNet (w/ or w/o) reg

**Some steps require replacing your local paths.**

1. Move to experiment dir.
```
cd Real-train
```

2. Download the training datasets([DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)), move it to `./dataset` and validation dataset(Set5), move it to `./dataset/benchmark`.

3. Run the training commands.
```
cd codes
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 realesrgan/train.py -opt options/train/train_realsrresnet.yml --launcher pytorch --auto_resume
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 realesrgan/train.py -opt options/train/train_realsrresnet_reg.yml --launcher pytorch --auto_resume
```
4. The experiments will be sorted in `./experiments`. 


### Acknowledgement

Many parts of this code is adapted from:

- [RDSR](https://github.com/XPixelGroup/RDSR/tree/main)
- [BasicSR](https://github.com/XPixelGroup/BasicSR/tree/master)

We thank the authors for sharing codes for their great works.

