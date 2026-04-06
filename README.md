# Aotearoa Bird Classifier

Deep learning project based on original [Aotearoa Species Classifier](https://github.com/Waikato/aotearoa-species-classifier). Reproduced model on birds subset of original dataset and investigated backbones, losses, and augmentations.

## Installation

### Clone Repository
```bash
git clone https://github.com/aimeexlin/aotearoa-bird-classifier.git
cd aotearoa-bird-classifier
```

### conda (Linux/CUDA)
```bash
conda env create -f environment.yml python=3.10
conda activate species
```

### pip
1. Create a Python 3.10 environment
2. Install PyTorch 2.0.0: https://pytorch.org/get-started
   ```bash
   # Example: Linux CUDA 11.7
   pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu117
   ```
3. Install remaining dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Setup

### Download Data Files
```bash
gdown 1L74V_Fqsvj1ku7drcHpBkS2imYLytsZD
gdown 1TXnETXa2do8jMDITqOf4FIBGZME0p1xc
gdown 1eSdpfSNjnh42FFLo3Y4cxZ3025N-YK7a
```

### Prepare Data
```bash
python download_res_grade.py                # Download bird subset (1/2)
python download_cap_cul.py                  # Download bird subset (2/2)
python perform_sanitise_instructions.py     # Clean data
python split.py                             # Split into train/test
```

## Training

```bash
python fine_tune.py \
  --backbone <backbone> \
  --loss <loss> \
  --augment <augment>
```

**Backbone options:**
- `env2` - EfficientNetV2-S (ImageNet21k, default)
- `cnx_i` - ConvNeXt-S (ImageNet22k)
- `cnx_d` - ConvNeXt-S (DINOv3)
- `vit_i` - ViT-B (ImageNet21k)
- `vit_inat` - ViT-B (iNaturalist finetuned)
- `vit_d` - ViT-B (DINOv3)

**Loss options:**
- `ce` - Cross-Entropy (default)
- `wce` - Weighted Cross-Entropy
- `focal` - Focal Loss
- `ldam` - LDAM + Deferred Re-weighting

**Augment options:**
- `auto` - AutoAugment (default)
- `mixup` - Mixup augmentation
- `cutmix` - CutMix augmentation

**Example:**
```bash
python fine_tune.py --backbone cnx_i --loss ldam --augment mixup
```

Models are saved to `models/{backbone}_{loss}_{augment}/` with TensorBoard logs in `tb/`.

## Evaluation

```bash
python validate.py \
  --backbone <backbone> \
  --name <model dir>
```

## Monitoring

View training progress with TensorBoard:
```bash
tensorboard --logdir tb
```