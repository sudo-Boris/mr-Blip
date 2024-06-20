# The Surprising Effectiveness of Multimodal Large Language Models for Video Moment Retrieval

* Authors: [Boris Meinardus](https://sudo-boris.github.io/), [Anil Batra](https://anilbatra2185.github.io/), [Anna Rohrbach](https://anna-rohrbach.net/), [Marcus Rohrbach](https://rohrbach.vision/)
* Paper: TBA
<!-- [arXiv](https://example.com/) -->
<!-- * Online Demo: Try our Gradio demo on Hugging Face[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/Shoubin/SeViLA) -->

<img src="./assets/teaser.png" alt="teaser image" width="800"/>

<img src="./assets/model.png" alt="teaser image" width="800"/>

# Code structure

```bash

# data & data preprocessing
./mr_BLIP_data

# pretrained checkpoints
./mr_BLIP_checkpoints

# mr_BLIP code
./lavis/

# running scripts for mr_BLIP training and inference
./run_scripts

```

# Setup

## Install Dependencies

1. (Optional) Creating conda environment

```bash
conda create -n mrBlip python=3.8
conda activate mrBlip
```

2. build from source

```bash
pip install -r requirements.txt
```

## Download Pretrained Models

We train Mr. BLIP on QVHighlights, Charades-STA, ActivityNet Captions, and ActivityNet 1.3 (TAL) and provide the checkpoints.
Download the [checkpoints](https://drive.google.com/drive/folders/1AR-rdUillx0fy7KS4zbEuswFMl7qR9Gj?usp=sharing) and put them under /mr_BLIP_checkpoints.

# Dataset Preparation

We test our model on:

* [Charades-STA](https://github.com/jiyanggao/TALL)
  * [Charades (Videos)](https://prior.allenai.org/projects/charades)

* [QVHighlights](https://github.com/jayleicn/moment_detr)

* [ActivityNet Captions](https://cs.stanford.edu/people/ranjaykrishna/densevid/)
* [ActivityNet 1.3 (TAL)](http://activity-net.org/download.html)

Please download original MR data and preprocess them via our [scripts](mr_BLIP_data/data_preprocess.ipynb).

# Training and Inference

We provide Mr. BLIP training and inference script examples as follows.

And please refer to [dataset page](lavis/configs/datasets/) to customize your data path.

You might want to update the [config files](lavis/projects/mr_BLIP/train/) for the respective runs to fit on your machine. They are currently set to run on 8 A100-80GB GPUs. You can simply reduce the batch size, reduce the number of frames, or apply a frame level embeddings aggregation (32 frame tokens -> 1 token) to fit on a smaller GPU.

## 1) QVH Finetuning

```bash
sh run_scripts/mr_BLIP/train/qvh.sh
```

## 2) Charades-STA Finetuning

```bash
sh run_scripts/mr_BLIP/train/charades.sh
```

## 3) ANet Captions Finetuning

```bash
sh run_scripts/mr_BLIP/train/anet.sh
```

## 4) ANet 1.3 (TAL) Finetuning

```bash
sh run_scripts/mr_BLIP/train/anet_TAL.sh
```

## 5) QVH Evaluation

```bash
sh run_scripts/mr_BLIP/eval/qvh.sh
```

## 6) Charades-STA Evaluation

```bash
sh run_scripts/mr_BLIP/eval/charades.sh
```

## 7) ANet Captions Evaluation

```bash
sh run_scripts/mr_BLIP/eval/anet.sh
```

## 8) ANet 1.3 (TAL) Evaluation

```bash
sh run_scripts/mr_BLIP/eval/anet_TAL.sh
```

# Acknowledgments

We thank the developers of [LAVIS](https://github.com/salesforce/LAVIS) and [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) for their public code release.

<!-- # Reference

Please cite our paper if you use our models in your works:

```bibtex
@inproceedings{meinardus2024mrBLIP,
  title   = {The Surprising Effectiveness of Multimodal Large Language Models for Video Moment Retrieval},
  author  = {Meinardus, Boris and Batra, Anil and Rohrbach, Anna and Rohrbach, Marcus},
  year    = {2024}
} -->
