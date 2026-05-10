# Aotearoa Bird Classifier

Deep learning project based on [Aotearoa Species Classifier](https://github.com/Waikato/aotearoa-species-classifier). Reproduced model on bird subset of original dataset and investigated loss functions, backbone architectures, pretraining, and data augmentation strategies.

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
gdown "https://drive.google.com/file/d/1L74V_Fqsvj1ku7drcHpBkS2imYLytsZD/view" -O multimedia.txt
gdown "https://drive.google.com/file/d/1TXnETXa2do8jMDITqOf4FIBGZME0p1xc/view" -O NZ-Species.csv
gdown "https://drive.google.com/file/d/1eSdpfSNjnh42FFLo3Y4cxZ3025N-YK7a/view" -O captive_cultivated.csv
```

### Prepare Bird Subset
```bash
python download_res_grade.py                # Download bird subset (1/2)
python download_cap_cul.py                  # Download bird subset (2/2)
python perform_sanitise_instructions.py     # Clean data
python split.py                             # Split into train/val
```

## Trained Models
`models.zip` contains the best checkpoint for each combination investigated, preserving the `models/{backbone}_{loss}_{augment}` directory structure.
```bash
gdown "https://drive.google.com/file/d/1K7ejJeEZvT1N1dtI37Xlg_WHZ_D_e4oO/view" -0 models.zip
unzip -qq models.zip
```

## Training
```bash
python fine_tune.py --backbone <backbone> --loss <loss> --augment <augment>
```

**Backbone options:**
- `env2` - EfficientNetV2-S (ImageNet-21k, default)
- `cnx_i` - ConvNeXt-S (ImageNet-22k)
- `cnx_d` - ConvNeXt-S (DINOv3)
- `vit_i` - ViT-B (ImageNet-21k)
- `vit_inat` - ViT-B (ImageNet-21k + iNaturalist)
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
python fine_tune.py --backbone vit_d --loss ldam --augment mixup
```

Models are saved to `models/{backbone}_{loss}_{augment}/` with TensorBoard logs in `tb/`.

## Evaluation

```bash
python validate.py --model <model dir>
```

## Monitoring

View training progress with TensorBoard:
```bash
tensorboard --logdir tb
```
