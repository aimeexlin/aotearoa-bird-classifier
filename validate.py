import argparse
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


def get_logits(output):
    return output.logits if hasattr(output, "logits") else output

# same as fine_tune.py
BACKBONES = {
    "env2":  ("tf_efficientnetv2_s.in21k",          416, 384),
    "cnx_i": ("convnext_small.fb_in22k",            256, 224),
    "cnx_d": ("convnext_small.dinov3_lvd1689m",     256, 224),
    "vit_d": ("vit_base_patch16_dinov3.lvd1689m",   256, 224),
    "vit_i": ("google/vit-base-patch16-224-in21k",   256, 224),
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


# abstention masks
def confidence_mask(probabilities, threshold):
    return probabilities.max(1)[0] > threshold

def margin_mask(probabilities, threshold):
    top_probabilities, _ = torch.topk(probabilities, 2, 1)
    return top_probabilities[:, 0] - top_probabilities[:, 1] > threshold

def entropy_mask(probabilities, threshold):
    return (
        1
        + (probabilities * torch.log(probabilities).nan_to_num()).sum(1)
        / math.log(probabilities.shape[1])
        > threshold
    )

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


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# args
def parse_args():
    parser = argparse.ArgumentParser(description="Validate bird species classifier checkpoints")
    parser.add_argument("--backbone", choices=list(BACKBONES.keys()), default="env2",
                        help="Backbone (must match the training run)")
    parser.add_argument("--name", required=True,
                        help="Experiment name — looks for models/<name>/ and writes to tb/<name>/")
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

    _, val_resize, val_crop = BACKBONES[args.backbone]
    is_hf_backbone = args.backbone in {"vit_i", "vit_inat"}

    seed_everything(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    abstentions = {"confidence": confidence_mask, "margin": margin_mask, "entropy": entropy_mask}
    thresholds = [
        0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
        0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999,
    ]

    checkpoints_dir = Path(f"models/{args.name}")
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
    print(f"Backbone: {args.backbone}", flush=True)
    print(f"Checkpoints dir: {checkpoints_dir}", flush=True)
    print(f"Epoch range: {args.start} -> {args.end} (step {args.step})", flush=True)
    print("=" * 80, flush=True)

    first_ckpt_path = checkpoints_dir / f"checkpoint_epoch{args.start}.pth"
    first_ckpt = torch.load(str(first_ckpt_path), map_location="cpu")
    num_classes = get_num_classes_from_checkpoint(first_ckpt)
    del first_ckpt

    model_name, _, _ = BACKBONES[args.backbone]
    if is_hf_backbone:
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

    writer_summary = SummaryWriter(f"tb/{args.name}/test_summary")
    writers_abstention = {
        a: SummaryWriter(f"tb/{args.name}/test_{a}") for a in abstentions
    }

    best_epoch = None
    best_tail_metric = None
    best_top1 = None
    best_top5 = None
    best_bin_accs = None

    epochs_to_eval = list(range(args.start, args.end, args.step))
    if not epochs_to_eval or epochs_to_eval[-1] != args.end:
        epochs_to_eval.append(args.end)

    for epoch in epochs_to_eval:
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

        out_csv = checkpoints_dir / f"test_{epoch}.csv"

        running_loss = 0.0
        processed = 0
        top_correct = [0] * 5
        processed_binned = [0] * 5
        correct_binned = [0] * 5
        abstention_processed = {a: [0] * len(thresholds) for a in abstentions}
        abstention_correct   = {a: [0] * len(thresholds) for a in abstentions}

        with open(out_csv, "w") as fp:
            fp.write("true_label,pred_label,confidence\n")

            with torch.no_grad():
                progress = tqdm(
                    val_loader, total=len(val_loader),
                    desc=f"Epoch {epoch}", leave=True, dynamic_ncols=True,
                )
                for step, (inputs, targets) in enumerate(progress):
                    inputs  = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    outputs = get_logits(model(inputs))
                    loss    = criterion(outputs, targets)

                    running_loss += loss.item()
                    processed    += targets.size(0)
                    probabilities = torch.softmax(outputs, dim=-1)

                    confidence, predicted = torch.topk(probabilities, 5, 1)

                    for tl_i, pl_i, c_i in zip(
                        targets.cpu(), predicted[:, 0].cpu(), confidence[:, 0].cpu()
                    ):
                        fp.write(f"{int(tl_i)},{int(pl_i)},{float(c_i)}\n")

                    correct_labels = predicted == targets.unsqueeze(1)

                    for a in abstentions:
                        for i, threshold in enumerate(thresholds):
                            mask = abstentions[a](probabilities, threshold)
                            abstention_processed[a][i] += mask.sum().item()
                            abstention_correct[a][i]   += (correct_labels[:, 0] * mask).sum().item()

                    top_correct = [
                        correct_labels[:, :k + 1].sum().item() + tc
                        for k, tc in enumerate(top_correct)
                    ]

                    for tb, cl in zip(
                        [class_bins[t] for t in targets.cpu().tolist()],
                        correct_labels[:, 0].cpu().tolist(),
                    ):
                        processed_binned[tb] += 1
                        correct_binned[tb]   += cl

                    progress.set_postfix(
                        loss=f"{running_loss / (step + 1):.4f}",
                        top1=f"{100.0 * top_correct[0] / processed:.2f}%",
                    )

        top_k_accs = [float(f"{100.0 * tc / processed:3.3f}") for tc in top_correct]
        avg_loss = running_loss / (step + 1)
        tail_accs = [
            100.0 * correct_binned[k] / processed_binned[k]
            for k in range(3) if processed_binned[k] > 0
        ]
        tail_metric = sum(tail_accs) / max(len(tail_accs), 1)
        print(
            f"Epoch {epoch} complete | loss={avg_loss:.4f} | "
            f"top1={top_k_accs[0]:.2f}% | top5={top_k_accs[4]:.2f}% | tail={tail_metric:.3f}%",
            flush=True,
        )
        bin_labels = ["1-4", "5-9", "10-19", "20-49", "50+"]
        bin_accs = [
            f"{bin_labels[k]}: {100.0 * correct_binned[k] / processed_binned[k]:.1f}%"
            if processed_binned[k] > 0 else f"{bin_labels[k]}: n/a"
            for k in range(5)
        ]
        print(f"Binned accuracy    | {' | '.join(bin_accs)}", flush=True)

        if best_tail_metric is None or tail_metric > best_tail_metric:
            best_epoch = epoch
            best_tail_metric = tail_metric
            best_top1 = top_k_accs[0]
            best_top5 = top_k_accs[4]
            best_bin_accs = bin_accs

        writer_summary.add_scalar("summary/loss", avg_loss, epoch)
        writer_summary.add_scalar("summary/tail_metric", tail_metric, epoch)
        for k, bin_range in enumerate(["1_4", "5_9", "10_19", "20_49", "50_"]):
            if processed_binned[k] > 0:
                writer_summary.add_scalar(
                    f"summary/{bin_range}",
                    100.0 * correct_binned[k] / processed_binned[k],
                    epoch,
                )
        for k in range(5):
            writer_summary.add_scalar(f"summary/top{k + 1:d}", top_k_accs[k], epoch)
        for a in abstentions:
            for i, threshold in enumerate(thresholds):
                denom = abstention_processed[a][i]
                writers_abstention[a].add_scalar(
                    f"abstention/acc_wrt_predicted_epoch_{epoch:d}",
                    0.0 if denom == 0 else 100.0 * abstention_correct[a][i] / denom,
                    round(100 * denom / processed),
                )

    if best_epoch is None:
        raise RuntimeError("No checkpoints were evaluated; cannot report a best epoch")

    print("=" * 80, flush=True)
    print(
        f"Best epoch: {best_epoch} | tail={best_tail_metric:.3f}% | "
        f"top1={best_top1:.2f}% | top5={best_top5:.2f}%",
        flush=True,
    )
    print(f"Best binned accuracy | {' | '.join(best_bin_accs)}", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()