# doc-llm

## Env Setup

Torch

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Other libs:

```bash
pip install -r requirements.txt
```

```bash
pip install -e .
```

## Code

A large part of the code is copied with small modification from https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-2B-V1-5 as InternVL is not yet available as a library.

## Datasets

### SROIE Dataset

```
https://www.kaggle.com/datasets/urbikn/sroie-datasetv2/data
```