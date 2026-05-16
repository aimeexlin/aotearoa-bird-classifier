import argparse
import csv
import math
import os
import random
import pickle
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader, has_file_allowed_extension
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score


def get_logits(output):
    return output.logits if hasattr(output, "logits") else output

# same as fine_tune.py
BACKBONES = {
    "env2":  ("tf_efficientnetv2_s.in21k",          416, 384),
    "cnx_i": ("convnext_tiny.fb_in22k",             256, 224),
    "cnx_d": ("convnext_tiny.dinov3_lvd1689m",      256, 224),
    "vit_d": ("vit_small_patch16_dinov3.lvd1689m",  256, 224),
    "vit_i": ("vit_small_patch16_224.augreg_in21k", 256, 224),
    "vit_inat": ("bryanzhou008/vit-base-patch16-224-in21k-finetuned-inaturalist", 256, 224),
}

# dataset
class ImageFolderAllowEmpty(torch.utils.data.Dataset):
    """ImageFolder that tolerates empty class directories in the test split."""
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


# helpers
def get_num_classes_from_checkpoint(checkpoint):
    if "num_classes" in checkpoint:
        return int(checkpoint["num_classes"])
    state = checkpoint["model"]
    for prefix in ("", "module."):
        for suffix in ("head.weight", "classifier.weight"):
            key = f"{prefix}{suffix}"
            if key in state:
                return state[key].shape[0]
    raise KeyError("Cannot determine num_classes from checkpoint.")

def infer_backbone_from_model_dir(model_dir: Path):
    parts = model_dir.name.split("_")
    for i in range(1, len(parts) + 1):
        candidate = "_".join(parts[:i])
        if candidate in BACKBONES:
            return candidate
    raise ValueError(f"Could not infer backbone from model directory name: {model_dir.name}")

def instance_bin(n):
    if n < 5:  return 0
    if n < 10: return 1
    if n < 20: return 2
    if n < 50: return 3
    return 4


def discover_checkpoint_epochs(checkpoints_dir):
    epochs = []
    for ckpt_path in checkpoints_dir.glob("checkpoint_epoch*.pth"):
        suffix = ckpt_path.stem.removeprefix("checkpoint_epoch")
        if suffix.isdigit():
            epochs.append(int(suffix))
    if not epochs:
        raise FileNotFoundError(f"No checkpoint_epoch*.pth files found under: {checkpoints_dir}")
    return sorted(set(epochs))


def load_predictions_from_csv(csv_path: Path):
    targets = []
    predictions = []

    with open(csv_path, newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        if "true_label" not in reader.fieldnames or "pred_label" not in reader.fieldnames:
            raise ValueError(
                f"CSV is missing required columns true_label/pred_label: {csv_path}"
            )

        for row in reader:
            targets.append(int(row["true_label"]))
            predictions.append(int(row["pred_label"]))

    if not targets:
        raise ValueError(f"CSV has no prediction rows: {csv_path}")

    return targets, predictions


def compute_metrics_from_predictions(all_targets, all_predictions, class_bins):
    processed = len(all_targets)
    top1 = 100.0 * sum(int(t == p) for t, p in zip(all_targets, all_predictions)) / processed

    processed_binned = [0] * 5
    correct_binned = [0] * 5
    for target, pred in zip(all_targets, all_predictions):
        if target not in class_bins:
            continue
        tb = class_bins[target]
        processed_binned[tb] += 1
        correct_binned[tb] += int(target == pred)

    tail_accs = [
        100.0 * correct_binned[k] / processed_binned[k]
        for k in range(3) if processed_binned[k] > 0
    ]
    tail_metric = sum(tail_accs) / max(len(tail_accs), 1)
    macro_f1 = 100.0 * f1_score(all_targets, all_predictions, average="macro", zero_division=0)

    bin_labels = ["1-4", "5-9", "10-19", "20-49", "50+"]
    bin_accs = [
        f"{bin_labels[k]}: {100.0 * correct_binned[k] / processed_binned[k]:.1f}%"
        if processed_binned[k] > 0 else f"{bin_labels[k]}: n/a"
        for k in range(5)
    ]

    return top1, tail_metric, macro_f1, bin_accs, processed_binned, correct_binned


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# args
def parse_args():
    parser = argparse.ArgumentParser(description="Validate bird species classifier checkpoints")
    parser.add_argument("--model", required=True,
                        help="Model directory name — looks for models/<model>/ and writes to tb/<model>/")
    parser.add_argument("--start", type=int, default=None,
                        help="First epoch to evaluate; defaults to the smallest checkpoint found")
    parser.add_argument("--end", type=int, default=None,
                        help="Last epoch to evaluate; defaults to the largest checkpoint found")
    parser.add_argument("--step", type=int, default=1,
                        help="Epoch step between evaluations")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible validation")
    return parser.parse_args()

# main
def main():
    args = parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoints_dir = Path(f"models/{args.model}")
    discovered_epochs = discover_checkpoint_epochs(checkpoints_dir)

    if args.start is None:
        args.start = discovered_epochs[0]
    if args.end is None:
        args.end = discovered_epochs[-1]
    if args.start > args.end:
        raise ValueError(f"Invalid epoch range: start ({args.start}) is greater than end ({args.end})")

    print("=" * 80, flush=True)
    print("Starting validation", flush=True)
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        visible_gpu_count = torch.cuda.device_count()
        print(f"Visible CUDA devices: {visible_gpu_count}", flush=True)
        print(f"GPU names: {[torch.cuda.get_device_name(g) for g in range(visible_gpu_count)]}", flush=True)
    backbone = infer_backbone_from_model_dir(checkpoints_dir)
    _, val_resize, val_crop = BACKBONES[backbone]
    is_hf_backbone = backbone == "vit_inat"
    print(f"Backbone: {backbone}", flush=True)
    print(f"Checkpoints dir: {checkpoints_dir}", flush=True)
    print(f"Epoch range: {args.start} -> {args.end} (step {args.step})", flush=True)
    print("=" * 80, flush=True)

    first_ckpt_path = checkpoints_dir / f"checkpoint_epoch{args.start}.pth"
    first_ckpt = torch.load(str(first_ckpt_path), map_location="cpu")
    num_classes = get_num_classes_from_checkpoint(first_ckpt)
    del first_ckpt

    model_name, _, _ = BACKBONES[backbone]
    if is_hf_backbone:
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except ImportError as exc:
            raise ImportError(
                "Backbone vit_inat requires transformers. Install with: pip install transformers"
            ) from exc

        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        ).to(device)
        image_processor = AutoImageProcessor.from_pretrained(model_name)
    else:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes).to(device)
        image_processor = None
    criterion = nn.CrossEntropyLoss().to(device)

    try:
        with open("instance_count.pkl", "rb") as f:
            instance_count = pickle.load(f)
    except FileNotFoundError:
        print("Warning: instance_count.pkl not found, binned metrics unavailable", flush=True)
        instance_count = None

    if image_processor is not None:
        mean = tuple(image_processor.image_mean) if image_processor.image_mean is not None else (0.5, 0.5, 0.5)
        std = tuple(image_processor.image_std) if image_processor.image_std is not None else (0.5, 0.5, 0.5)
    else:
        data_config = timm.data.resolve_model_data_config(model)
        mean = data_config["mean"]
        std = data_config["std"]

    normalize = transforms.Normalize(mean=mean, std=std)
    val_transform = transforms.Compose([
        transforms.Resize(val_resize),
        transforms.CenterCrop(val_crop),
        transforms.ToTensor(),
        normalize,
    ])
    val_dataset = ImageFolderAllowEmpty("dataset/test", transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False,
        num_workers=args.workers, pin_memory=False,
    )
    print(f"Validation samples: {len(val_dataset)}", flush=True)

    class_bins = {
        v: instance_bin(instance_count.get(k, 0))
        for k, v in val_dataset.class_to_idx.items()
    } if instance_count is not None else {}

    writer_summary = SummaryWriter(f"tb/{args.model}/test")
    try:
        best_epoch = None
        best_tail_metric = None
        best_top1 = None
        best_bin_accs = None
        best_macro_f1 = None

        epochs_to_eval = list(range(args.start, args.end, args.step))
        if not epochs_to_eval or epochs_to_eval[-1] != args.end:
            epochs_to_eval.append(args.end)

        for epoch in epochs_to_eval:
            out_csv = checkpoints_dir / f"test_{epoch}.csv"

            avg_loss = None

            if out_csv.exists():
                print(f"Found existing predictions, recalculating metrics from: {out_csv}", flush=True)
                all_targets, all_predictions = load_predictions_from_csv(out_csv)
            else:
                ckpt_path = checkpoints_dir / f"checkpoint_epoch{epoch}.pth"
                if not ckpt_path.exists():
                    print(f"Checkpoint not found, skipping: {ckpt_path}", flush=True)
                    continue

                print(f"Loading checkpoint: {ckpt_path}", flush=True)
                checkpoint = torch.load(str(ckpt_path), map_location=device)
                model_state = checkpoint["model"]
                model.load_state_dict(model_state)
                del checkpoint
                model.eval()

                running_loss = 0.0
                processed = 0
                top_correct = 0
                all_targets = []
                all_predictions = []

                with open(out_csv, "w") as fp:
                    fp.write("true_label,pred_label,confidence\n")

                    with torch.no_grad():
                        progress = tqdm(
                            val_loader, total=len(val_loader),
                            desc=f"Epoch {epoch}", leave=True, dynamic_ncols=True,
                        )
                        for step, (inputs, targets) in enumerate(progress):
                            inputs = inputs.to(device, non_blocking=True)
                            targets = targets.to(device, non_blocking=True)
                            outputs = get_logits(model(inputs))
                            loss = criterion(outputs, targets)

                            running_loss += loss.item()
                            processed += targets.size(0)
                            probabilities = torch.softmax(outputs, dim=-1)

                            confidence, predicted = torch.topk(probabilities, 5, 1)

                            for tl_i, pl_i, c_i in zip(
                                targets.cpu(), predicted[:, 0].cpu(), confidence[:, 0].cpu()
                            ):
                                fp.write(f"{int(tl_i)},{int(pl_i)},{float(c_i)}\n")
                                all_targets.append(int(tl_i))
                                all_predictions.append(int(pl_i))

                            correct_labels = predicted == targets.unsqueeze(1)

                            top_correct += correct_labels[:, 0].sum().item()

                            progress.set_postfix(
                                loss=f"{running_loss / (step + 1):.4f}",
                                top1=f"{100.0 * top_correct / processed:.2f}%",
                            )

                avg_loss = running_loss / (step + 1)

            top1, tail_metric, macro_f1, bin_accs, processed_binned, correct_binned = compute_metrics_from_predictions(
                all_targets,
                all_predictions,
                class_bins,
            )

            loss_str = "n/a" if avg_loss is None else f"{avg_loss:.4f}"
            print(
                f"Epoch {epoch} complete | loss={loss_str} | "
                f"top1={top1:.2f}% | tail={tail_metric:.3f}% | macro_f1={macro_f1:.2f}%",
                flush=True,
            )
            print(f"Binned accuracy    | {' | '.join(bin_accs)}", flush=True)

            if best_macro_f1 is None or macro_f1 > best_macro_f1:
                best_epoch = epoch
                best_tail_metric = tail_metric
                best_top1 = top1
                best_bin_accs = bin_accs
                best_macro_f1 = macro_f1

            if avg_loss is not None:
                writer_summary.add_scalar("test/loss", avg_loss, epoch)
            writer_summary.add_scalar("test/tail_metric", tail_metric, epoch)
            writer_summary.add_scalar("test/macro_f1", macro_f1, epoch)
            for k, bin_range in enumerate(["1_4", "5_9", "10_19", "20_49", "50_"]):
                if processed_binned[k] > 0:
                    writer_summary.add_scalar(
                        f"test/{bin_range}",
                        100.0 * correct_binned[k] / processed_binned[k],
                        epoch,
                    )
            writer_summary.add_scalar("test/top1", top1, epoch)
            writer_summary.flush()

        if best_epoch is None:
            raise RuntimeError("No checkpoints were evaluated; cannot report a best epoch")

        print("=" * 80, flush=True)
        print(
            f"Best epoch: {best_epoch} | macro_f1={best_macro_f1:.2f}% | "
            f"tail={best_tail_metric:.3f}% | top1={best_top1:.2f}%",
            flush=True,
        )
        print(f"Best binned accuracy | {' | '.join(best_bin_accs)}", flush=True)
        print("=" * 80, flush=True)
    finally:
        writer_summary.flush()
        writer_summary.close()

if __name__ == "__main__":
    main()