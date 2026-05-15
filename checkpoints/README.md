# Pretrained checkpoints

This directory holds trained model weights. Files here are gitignored
except for this README.

## Expected files after training

| File              | Description                                          | Size   |
|-------------------|------------------------------------------------------|--------|
| `tcae_best.pth`   | TCAE backbone trained on PTB-XL training folds       | ~3 MB  |
| `W_global.npy`    | Population-mean transform matrix (12, 3) for I,II,V3 | <1 KB  |

## How to obtain

### Option 1: train from scratch

```bash
python experiments/train_tcae.py
```

This writes both files into this directory.

### Option 2: download pretrained

A pretrained checkpoint trained on PTB-XL folds 1–8 is hosted on
[Zenodo](https://zenodo.org/) (link to be added once released).

```bash
# placeholder — update once published
wget https://zenodo.org/record/<RECORD>/files/tcae_best.pth -O checkpoints/tcae_best.pth
wget https://zenodo.org/record/<RECORD>/files/W_global.npy -O checkpoints/W_global.npy
```
