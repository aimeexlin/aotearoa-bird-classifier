import argparse
import json
import math
import os
import random
import pickle
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader, has_file_allowed_extension
from torch.optim import RMSprop
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def get_logits(output):
    """Support both timm tensor outputs and HF ImageClassifierOutput."""
    return output.logits if hasattr(output, "logits") else output

# backbones configs
BACKBONES = {
    #            timm name                                                        batch  train  val_resize  val_crop
    "env2":     ("tf_efficientnetv2_s.in21k",                                       256,   300,    416,        384),
    "cnx_i":    ("convnext_small.fb_in22k",                                         256,   224,    256,        224),
    "cnx_d":    ("convnext_small.dinov3_lvd1689m",                                  256,   224,    256,        224),
    "vit_d":    ("vit_base_patch16_dinov3.lvd1689m",                                 64,   224,    256,        224),
    "vit_i":    ("google/vit-base-patch16-224-in21k",                                64,   224,    256,        224), 
    "vit_inat": ("bryanzhou008/vit-base-patch16-224-in21k-finetuned-inaturalist",    64,   224,    256,        224),
}

# training stages
STAGES = [5, 495]

# optimiser hyperparams
LR_BASE      = 1e-6
WEIGHT_DECAY = 1e-5
MOMENTUM     = 0.9
VAL_BATCH    = 64

# dataset
class ImageFolderAllowEmpty(torch.utils.data.Dataset):
    """ImageFolder that tolerates empty class directories in the val/test split."""
    def __init__(self, root, transform=None):
        self.transform = transform
        self.loader = default_loader

        classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"No class folders found under: {root}")
        self.classes = classes
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        samples = []
        for cls_name in classes:
            cls_dir = os.path.join(root, cls_name)
            for walk_root, _, fnames in os.walk(cls_dir, followlinks=True):
                for fname in sorted(fnames):
                    path = os.path.join(walk_root, fname)
                    if has_file_allowed_extension(path, IMG_EXTENSIONS):
                        samples.append((path, self.class_to_idx[cls_name]))

        if not samples:
            raise FileNotFoundError(f"Found no valid image files under: {root}")

        self.samples = samples
        self.imgs = samples
        self.targets = [t for _, t in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

# losses
class FocalLoss(nn.Module):
    """Focal loss for class imbalance (Lin et al., 2017)."""
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin loss (Cao et al., 2019)."""
    def __init__(self, class_counts, max_margin=0.5, scale=30.0, weight=None):
        super().__init__()
        margins = max_margin / (class_counts ** 0.25)
        margins = margins * (max_margin / margins.max())
        self.register_buffer("margins", margins)
        self.scale = scale
        self.weight = weight

    def forward(self, inputs, targets):
        inputs_adjusted = inputs.clone()
        inputs_adjusted[torch.arange(len(targets)), targets] -= self.margins[targets]
        return F.cross_entropy(self.scale * inputs_adjusted, targets, weight=self.weight)

# augmentations
def mixup_batch(inputs, targets, alpha=1.0):
    """Mix two batches of samples and their targets."""
    batch_size = inputs.size(0)
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(batch_size).to(inputs.device)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
    targets_a = targets
    targets_b = targets[index]
    return mixed_inputs, targets_a, targets_b, lam

def cutmix_batch(inputs, targets, alpha=1.0):
    """Cut and mix two batches of samples."""
    batch_size = inputs.size(0)
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(batch_size).to(inputs.device)
    
    h, w = inputs.size(2), inputs.size(3)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    
    cx = np.random.randint(0, w)
    cy = np.random.randint(0, h)
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    
    mixed_inputs = inputs.clone()
    mixed_inputs[:, :, bby1:bby2, bbx1:bbx2] = inputs[index, :, bby1:bby2, bbx1:bbx2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (h * w))
    targets_a = targets
    targets_b = targets[index]
    return mixed_inputs, targets_a, targets_b, lam

# args
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning for bird species classification")
    parser.add_argument("--backbone", choices=list(BACKBONES.keys()), default="env2",
                        help="Backbone and pretraining: env2=EfficientNetV2-S IN21k, "
                             "cnx_i=ConvNeXt-S IN22k, cnx_d=ConvNeXt-S DINOv3, vit_d=ViT-B DINOv3")
    parser.add_argument("--loss", choices=["ce", "wce", "focal", "ldam"], default="ce",
                        help="Loss function: ce, weighted ce, focal, LDAM+DRW")
    parser.add_argument("--augment", choices=["auto", "mixup", "cutmix"], default="auto",
                        help="Augmentation strategy: auto=AutoAugment, mixup=Mixup, cutmix=CutMix")
    parser.add_argument("--load", type=int, default=None,
                        help="Resume from checkpoint epoch number")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--checkpoint-every", type=int, default=1,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--early-stop-patience", type=int, default=5,
                        help="Early stop after N epochs without tail metric improvement")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4,
                        help="Minimum improvement in tail metric to count as progress")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible training")
    return parser.parse_args()


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)

# helpers
def stage_end_epochs(stage_lengths):
    ends = []
    running = 0
    for e in stage_lengths:
        running += e
        ends.append(running)
    return ends

def stage_from_epoch(epoch, stage_ends):
    for idx, stage_end in enumerate(stage_ends):
        if epoch < stage_end:
            return idx
    return len(stage_ends) - 1

def apply_stage_freezing(model, backbone, stage_i, stage_count):
    model_ref = model.module if isinstance(model, nn.DataParallel) else model
    model_ref.requires_grad_(False)

    if stage_i == stage_count - 1:
        # unfreeze everything after warmup
        model_ref.requires_grad_(True)
        return

    # unfreeze classifier head only during warmup
    if hasattr(model_ref, "head"):
        model_ref.head.requires_grad_(True)
    elif hasattr(model_ref, "classifier"):
        model_ref.classifier.requires_grad_(True)
    else:
        raise AttributeError(f"Cannot find classifier head for backbone {backbone}")

def normalize_state_dict_for_model(state_dict, model_ref):
    model_keys = set(model_ref.state_dict().keys())
    ckpt_has_module = all(k.startswith("module.") for k in state_dict.keys()) if state_dict else False
    model_has_module = all(k.startswith("module.") for k in model_keys) if model_keys else False
    if ckpt_has_module and not model_has_module:
        return {k[len("module."):]: v for k, v in state_dict.items()}
    if model_has_module and not ckpt_has_module:
        return {f"module.{k}": v for k, v in state_dict.items()}
    return state_dict

def get_sorted_class_names(root):
    classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"No class folders found under: {root}")
    return classes

# eval
def evaluate(model, loader, device, instance_count):
    criterion = nn.CrossEntropyLoss().to(device)
    model.eval()
    running_loss = 0.0
    processed = 0
    top1_correct = 0
    top5_correct = 0
    processed_binned = [0] * 5
    correct_binned = [0] * 5

    def instance_bin(n):
        if n < 5:  return 0
        if n < 10: return 1
        if n < 20: return 2
        if n < 50: return 3
        return 4

    class_bins = None
    if hasattr(loader.dataset, "class_to_idx") and instance_count is not None:
        class_bins = {
            v: instance_bin(instance_count.get(k, 0))
            for k, v in loader.dataset.class_to_idx.items()
        }

    with torch.no_grad():
        progress = tqdm(loader, total=len(loader), desc="Validation", leave=False, dynamic_ncols=True)
        for inputs, targets in progress:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with amp.autocast(enabled=(device.type == "cuda")):
                outputs = get_logits(model(inputs))
                loss = criterion(outputs, targets)

            batch_items = inputs.size(0)
            running_loss += loss.item() * batch_items
            processed += batch_items

            _, top5_pred = torch.topk(outputs, k=5, dim=1)
            correct = top5_pred.eq(targets.unsqueeze(1))
            top1_correct += correct[:, 0].sum().item()
            top5_correct += correct.any(dim=1).sum().item()

            if class_bins is not None:
                for t, c in zip(targets.cpu().tolist(), correct[:, 0].cpu().tolist()):
                    tb = class_bins.get(t)
                    if tb is not None:
                        processed_binned[tb] += 1
                        correct_binned[tb] += c

            progress.set_postfix(loss=f"{running_loss / max(processed, 1):.4f}")

    avg_loss = running_loss / max(processed, 1)
    top1 = 100.0 * top1_correct / max(processed, 1)
    top5 = 100.0 * top5_correct / max(processed, 1)
    tail_accs = [
        100.0 * correct_binned[k] / processed_binned[k]
        for k in range(3) if processed_binned[k] > 0
    ]
    tail_metric = sum(tail_accs) / max(len(tail_accs), 1)
    return avg_loss, top1, top5, tail_metric

# main
def main():
    args = parse_args()

    seed_everything(args.seed)

    model_name, batch_size, train_crop, val_resize, val_crop = BACKBONES[args.backbone]
    lr = LR_BASE * batch_size / 16.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoints_name = f"{args.backbone}_{args.loss}_{args.augment}"
    checkpoints_dir = Path(f"models/{checkpoints_name}")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    class_names = get_sorted_class_names("dataset/train")
    num_classes = len(class_names)
    classes_path = checkpoints_dir / "classes.json"
    classes_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    print("=" * 80, flush=True)
    print("Starting fine-tuning", flush=True)
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        visible_gpu_count = torch.cuda.device_count()
        print(f"Visible CUDA devices: {visible_gpu_count}", flush=True)
        print(f"GPU names: {[torch.cuda.get_device_name(g) for g in range(visible_gpu_count)]}", flush=True)
    print(f"Backbone: {args.backbone} ({model_name})", flush=True)
    print(f"Loss: {args.loss}", flush=True)
    print(f"Augment: {args.augment}", flush=True)
    print(f"Batch size: {batch_size}", flush=True)
    print(f"Workers: {args.workers}", flush=True)
    print(f"Checkpoints: {checkpoints_dir}", flush=True)
    print(f"Early stopping: patience={args.early_stop_patience}, min_delta={args.early_stop_min_delta}", flush=True)
    print("=" * 80, flush=True)

    if args.backbone in {"vit_i", "vit_inat"}:
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except ImportError as exc:
            raise ImportError(
                "Backbones vit_i/vit_inat require transformers. Install with: pip install transformers"
            ) from exc

        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        ).to(device)
        hf_image_processor = AutoImageProcessor.from_pretrained(model_name)
    else:
        model = timm.create_model(
            model_name,
            pretrained=(args.load is None),
            num_classes=num_classes,
        ).to(device)
        hf_image_processor = None

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"DataParallel enabled on {torch.cuda.device_count()} GPUs", flush=True)

    try:
        with open("instance_count.pkl", "rb") as f:
            instance_count = pickle.load(f)
    except FileNotFoundError:
        print("Warning: instance_count.pkl not found, binned metrics unavailable", flush=True)
        instance_count = None

    if hf_image_processor is not None:
        mean = tuple(hf_image_processor.image_mean) if hf_image_processor.image_mean is not None else (0.5, 0.5, 0.5)
        std = tuple(hf_image_processor.image_std) if hf_image_processor.image_std is not None else (0.5, 0.5, 0.5)
    else:
        data_config = timm.data.resolve_model_data_config(model)
        mean = data_config['mean']
        std = data_config['std']
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(train_crop),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = datasets.ImageFolder("dataset/train", transform=train_transform)
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        generator=train_generator,
        worker_init_fn=seed_worker,
    )

    # class weights
    class_counts = torch.bincount(torch.tensor(train_dataset.targets), minlength=num_classes).float()
    class_counts = torch.clamp(class_counts, min=1)
    beta = 0.9999
    effective_num = 1.0 - torch.pow(beta, class_counts)
    weights = (1.0 - beta) / (effective_num + 1e-8)
    weights = weights / weights.sum() * num_classes
    weights = weights.to(device)

    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "wce":
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif args.loss == "focal":
        criterion = FocalLoss(gamma=2.0, weight=weights)
    elif args.loss == "ldam":
        criterion = LDAMLoss(class_counts=class_counts.to(device))
    criterion = criterion.to(device)

    val_transform = transforms.Compose([
        transforms.Resize(val_resize),
        transforms.CenterCrop(val_crop),
        transforms.ToTensor(),
        normalize,
    ])
    val_dataset = ImageFolderAllowEmpty("dataset/test", transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH, shuffle=False, num_workers=args.workers)

    print(f"Training samples: {len(train_dataset)}", flush=True)
    print(f"Steps per epoch: {len(train_loader)}", flush=True)
    print(f"Validation samples: {len(val_dataset)}", flush=True)

    stage_ends = stage_end_epochs(STAGES)
    total_epochs = stage_ends[-1]

    def save_checkpoint(epoch_num, stage_i, reason):
        cp_path = checkpoints_dir / f"checkpoint_epoch{epoch_num}.pth"
        model_ref = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(
            {
                "epoch": epoch_num,
                "stage_i": stage_i,
                "model": model_ref.state_dict(),
                "optim": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epochs_without_improvement": epochs_without_improvement,
                "num_classes": num_classes,
                "class_names": class_names,
                "class_names_file": classes_path.name,
            },
            str(cp_path),
        )
        print(f"Saved checkpoint ({reason}): {cp_path}", flush=True)

    scaler = amp.GradScaler(enabled=(device.type == "cuda"))
    writer = SummaryWriter(f"tb/{checkpoints_name}/train")

    best_tail_metric = -float("inf")
    epochs_without_improvement = 0
    start_epoch = 0
    checkpoint = None

    if args.load is not None:
        cp_path = checkpoints_dir / f"checkpoint_epoch{args.load}.pth"
        checkpoint = torch.load(str(cp_path), map_location=device)
        model_ref = model.module if isinstance(model, nn.DataParallel) else model
        model_state = normalize_state_dict_for_model(checkpoint["model"], model_ref)
        model_ref.load_state_dict(model_state)
        start_epoch = int(checkpoint.get("epoch", args.load))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        print(f"Loaded checkpoint: {cp_path} (epoch={start_epoch})", flush=True)

    stage_i_prev = None
    optimizer = None
    lr_manager = None
    stopped_early = False
    last_completed_epoch = start_epoch

    if start_epoch >= total_epochs:
        print("Start epoch is beyond configured total epochs; nothing to train.", flush=True)
        writer.close()
        return

    for epoch in range(start_epoch, total_epochs):
        stage_i = stage_from_epoch(epoch, stage_ends)
        if stage_i != stage_i_prev:
            apply_stage_freezing(model, args.backbone, stage_i, len(STAGES))
            optimizer = RMSprop(
                params=filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
                alpha=0.9,
                eps=1e-8,
                weight_decay=WEIGHT_DECAY,
                momentum=MOMENTUM,
            )
            lr_manager = LambdaLR(
                optimizer,
                lambda step, e=epoch: 0.99 ** (float(e) + float(step) / float(len(train_loader))),
            )

            if checkpoint is not None and int(checkpoint.get("stage_i", -1)) == stage_i:
                if "optim" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optim"])
                if "scaler" in checkpoint:
                    scaler.load_state_dict(checkpoint["scaler"])
                print(f"Restored optimizer/scaler for stage {stage_i}", flush=True)
                checkpoint = None

            # ldam: ce during warmup, ldam + drw after warmup
            if args.loss == "ldam":
                if stage_i == 0:
                    criterion = nn.CrossEntropyLoss().to(device)
                    print("LDAM: using CE loss during warmup stage", flush=True)
                else:
                    criterion = LDAMLoss(class_counts=class_counts.to(device)).to(device)
                    sample_weights = weights[train_dataset.targets]
                    sampler = torch.utils.data.WeightedRandomSampler(
                        sample_weights, num_samples=len(train_dataset), replacement=True
                    )
                    train_loader = DataLoader(
                        train_dataset, batch_size=batch_size,
                        sampler=sampler, num_workers=args.workers,
                    )
                    print("LDAM: switched to LDAM loss + DRW sampler", flush=True)

            # reset early stopping
            if stage_i_prev is not None:
                epochs_without_improvement = 0
                best_tail_metric = -float("inf")
                print("Reset early stopping counter for new stage", flush=True)

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(
                f"Entering stage {stage_i + 1}/{len(STAGES)} at epoch {epoch + 1} | "
                f"trainable params: {trainable_params:,}",
                flush=True,
            )
            stage_i_prev = stage_i

        model.train()
        epoch_loss_sum = 0.0
        epoch_items = 0
        epoch_start = time.time()

        epoch_retried_for_loader = False
        while True:
            try:
                progress = tqdm(
                    train_loader,
                    total=len(train_loader),
                    desc=f"Epoch {epoch + 1}/{total_epochs}",
                    leave=True,
                    dynamic_ncols=True,
                )
                for step, (inputs, targets) in enumerate(progress):
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)

                    # apply mixup or cutmix augmentation
                    if args.augment == "mixup":
                        inputs, targets_a, targets_b, lam = mixup_batch(inputs, targets, alpha=1.0)
                    elif args.augment == "cutmix":
                        inputs, targets_a, targets_b, lam = cutmix_batch(inputs, targets, alpha=1.0)
                    else:
                        targets_a = targets_b = targets
                        lam = 1.0

                    with amp.autocast(enabled=(device.type == "cuda")):
                        outputs = get_logits(model(inputs))
                        if args.augment in {"mixup", "cutmix"}:
                            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                        else:
                            loss = criterion(outputs, targets)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    lr_manager.step()
                    scaler.update()

                    batch_items = inputs.size(0)
                    epoch_loss_sum += loss.item() * batch_items
                    epoch_items += batch_items

                    progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

                    global_step = epoch * len(train_loader) + step
                    if step % 100 == 99:
                        writer.add_scalar("train/loss_step", loss.item(), global_step)
                        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                break
            except RuntimeError as exc:
                msg = str(exc)
                loader_worker_failed = ("DataLoader worker" in msg) or ("bus error" in msg.lower())
                if loader_worker_failed and args.workers > 0 and not epoch_retried_for_loader:
                    print(
                        "DataLoader worker failure detected. "
                        f"Falling back to workers={max(args.workers - 1, 0)} and restarting epoch.",
                        flush=True,
                    )
                    args.workers = max(args.workers - 1, 0)
                    train_loader = DataLoader(
                        train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=args.workers,
                    )
                    val_loader = DataLoader(
                        val_dataset, batch_size=VAL_BATCH,
                        shuffle=False, num_workers=args.workers,
                    )
                    epoch_loss_sum = 0.0
                    epoch_items = 0
                    epoch_retried_for_loader = True
                    continue
                raise

        epoch_loss = epoch_loss_sum / max(epoch_items, 1)
        writer.add_scalar("train/loss_epoch", epoch_loss, epoch)

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch + 1} complete | loss={epoch_loss:.6f} | "
            f"time={epoch_time:.1f}s | lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )

        val_loss, val_top1, val_top5, tail_metric = evaluate(model, val_loader, device, instance_count)
        writer.add_scalar("val/loss_epoch", val_loss, epoch)
        writer.add_scalar("val/top1", val_top1, epoch)
        writer.add_scalar("val/top5", val_top5, epoch)
        writer.add_scalar("val/tail_metric", tail_metric, epoch)
        print(
            f"Validation | loss={val_loss:.6f} | top1={val_top1:.3f}% | "
            f"top5={val_top5:.3f}% | tail={tail_metric:.3f}%",
            flush=True,
        )

        improved = (tail_metric - best_tail_metric) > args.early_stop_min_delta
        if improved:
            best_tail_metric = tail_metric
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for {epochs_without_improvement} epoch(s). "
                f"Best tail metric: {best_tail_metric:.3f}%",
                flush=True,
            )

        if (epoch + 1) % args.checkpoint_every == 0:
            save_checkpoint(epoch + 1, stage_i, "periodic")

        last_completed_epoch = epoch + 1

        warmup_end = stage_ends[0]
        if epochs_without_improvement >= args.early_stop_patience and epoch + 1 > warmup_end:
            save_checkpoint(epoch + 1, stage_i, "early-stop")
            print(
                f"Early stopping at epoch {epoch + 1}: no tail metric improvement "
                f"greater than {args.early_stop_min_delta} for {args.early_stop_patience} consecutive epochs.",
                flush=True,
            )
            stopped_early = True
            break

    if not stopped_early and last_completed_epoch > 0:
        final_stage = stage_from_epoch(last_completed_epoch - 1, stage_ends)
        save_checkpoint(last_completed_epoch, final_stage, "final")

    writer.close()
    print("Training finished.", flush=True)

if __name__ == "__main__":
    main()