import argparse
import json
import os
import time
from pathlib import Path

import timm
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader, has_file_allowed_extension
from torch.optim import RMSprop
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


BATCH_SIZES = {"s": 256, "m": 96, "l": 48}


class ImageFolderAllowEmpty(torch.utils.data.Dataset):
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


def build_loader(dataset, batch_size, shuffle, args):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
        kwargs["persistent_workers"] = args.persistent_workers
    return DataLoader(dataset, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning for EfficientNetV2")
    parser.add_argument("model_size", choices=["s", "m", "l"], help="EfficientNetV2 size")
    parser.add_argument("split", help="Dataset split under dataset/<split>")
    parser.add_argument("--load-checkpoint", type=int, default=None, help="Resume from checkpoint epoch number (e.g. 50 for checkpoint_epoch50.pth)")
    parser.add_argument("--epochs", type=int, nargs="+", default=[5, 495], help="Stage-wise epochs. First stage trains classifier only, second trains full model.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override default batch size")
    parser.add_argument("--val-split", type=str, default="test", help="Validation split under dataset/<val-split>. Use None to disable validation.")
    parser.add_argument("--val-batch-size", type=int, default=64, help="Validation batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--pin-memory", action="store_true", help="Enable pin_memory in DataLoader (off by default for safer low-shm environments)")
    parser.add_argument("--prefetch-factor", type=int, default=1, help="Per-worker prefetch batches when num_workers > 0")
    parser.add_argument("--persistent-workers", action="store_true", help="Keep DataLoader workers alive between epochs")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Checkpoint period in epochs")
    parser.add_argument("--early-stop-patience", type=int, default=5, help="Stop after N bad epochs")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4, help="Minimum loss improvement")
    parser.add_argument("--lr-base", type=float, default=1e-6, help="Base LR before batch scaling")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="RMSprop momentum")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    return parser.parse_args()


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


def apply_stage_freezing(model, stage_i, stage_count):
    model_ref = model.module if isinstance(model, nn.DataParallel) else model
    model_ref.requires_grad_(False)
    if stage_i == 0:
        model_ref.classifier.requires_grad_(True)
    elif stage_i == stage_count - 1:
        model_ref.requires_grad_(True)
    else:
        model_ref.classifier.requires_grad_(True)
        if stage_i >= 1:
            model_ref.bn2.requires_grad_(True)
            model_ref.conv_head.requires_grad_(True)
        model_ref.blocks[-stage_i].requires_grad_(True)


def normalize_state_dict_for_model(state_dict, model_ref):
    model_keys = set(model_ref.state_dict().keys())
    ckpt_has_module = all(k.startswith("module.") for k in state_dict.keys()) if state_dict else False
    model_has_module = all(k.startswith("module.") for k in model_keys) if model_keys else False
    if ckpt_has_module and not model_has_module:
        return {k[len("module."):]: v for k, v in state_dict.items()}
    if model_has_module and not ckpt_has_module:
        return {f"module.{k}": v for k, v in state_dict.items()}
    return state_dict


def adapt_classifier_state_dict(state_dict, num_classes):
    """Shrink classifier head if checkpoint was saved with more classes (e.g. 14991 -> 336)."""
    for prefix in ("", "module."):
        w_key = f"{prefix}classifier.weight"
        b_key = f"{prefix}classifier.bias"
        if w_key in state_dict and b_key in state_dict:
            rows = state_dict[w_key].shape[0]
            if rows > num_classes:
                print(f"Adapting classifier head: {rows} -> {num_classes} classes", flush=True)
                state_dict[w_key] = state_dict[w_key][:num_classes].clone()
                state_dict[b_key] = state_dict[b_key][:num_classes].clone()
            return state_dict
    return state_dict


def get_sorted_class_names(root):
    classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"No class folders found under: {root}")
    return classes


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    processed = 0
    top1_correct = 0
    top5_correct = 0
    with torch.no_grad():
        progress = tqdm(
            loader,
            total=len(loader),
            desc="Validation",
            leave=False,
            dynamic_ncols=True,
        )
        for inputs, targets in progress:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            batch_items = inputs.size(0)
            running_loss += loss.item() * batch_items
            processed += batch_items

            _, top5_pred = torch.topk(outputs, k=5, dim=1)
            correct = top5_pred.eq(targets.unsqueeze(1))
            top1_correct += correct[:, 0].sum().item()
            top5_correct += correct.any(dim=1).sum().item()

            progress.set_postfix(loss=f"{running_loss / max(processed, 1):.4f}")

    avg_loss = running_loss / max(processed, 1)
    top1 = 100.0 * top1_correct / max(processed, 1)
    top5 = 100.0 * top5_correct / max(processed, 1)
    return avg_loss, top1, top5


def main():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    batch_size = args.batch_size if args.batch_size is not None else BATCH_SIZES[args.model_size]
    lr = args.lr_base * batch_size / 16.0

    checkpoints_dir = Path(f"{args.model_size}_{args.split}")
    checkpoints_dir.mkdir(exist_ok=True)

    class_names = get_sorted_class_names(f"dataset/{args.split}")
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
    print(f"Model: tf_efficientnetv2_{args.model_size}_in21k", flush=True)
    print(f"Num classes: {num_classes}", flush=True)
    print(f"Class map saved to: {classes_path}", flush=True)
    print(f"Data split: dataset/{args.split}", flush=True)
    val_enabled = args.val_split is not None and str(args.val_split).lower() != "none"
    print(f"Validation split: {('dataset/' + args.val_split) if val_enabled else 'disabled'}", flush=True)
    print(f"Batch size: {batch_size}", flush=True)
    print(
        f"DataLoader: workers={args.num_workers}, pin_memory={args.pin_memory}, "
        f"prefetch_factor={(args.prefetch_factor if args.num_workers > 0 else 'n/a')}, "
        f"persistent_workers={(args.persistent_workers if args.num_workers > 0 else 'n/a')}",
        flush=True,
    )
    print(f"Epoch stages: {args.epochs} (total={sum(args.epochs)})", flush=True)
    print(
        f"Early stopping: patience={args.early_stop_patience}, min_delta={args.early_stop_min_delta}",
        flush=True,
    )
    print(f"Checkpoints: {checkpoints_dir}", flush=True)
    print("=" * 80, flush=True)

    model = timm.create_model(
        f"tf_efficientnetv2_{args.model_size}_in21k",
        pretrained=(args.load_checkpoint is None),
        num_classes=num_classes,
    ).to(device)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"DataParallel enabled on {torch.cuda.device_count()} GPUs", flush=True)

    if not args.no_compile and hasattr(torch, "compile") and not isinstance(model, nn.DataParallel):
        try:
            model = torch.compile(model)
            print("torch.compile enabled", flush=True)
        except Exception as exc:
            print(f"torch.compile unavailable, continuing without it: {exc}", flush=True)
    elif isinstance(model, nn.DataParallel) and not args.no_compile:
        print("torch.compile skipped (DataParallel enabled)", flush=True)

    criterion = nn.CrossEntropyLoss().to(device)

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(300 if args.model_size == "s" else 384),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(),
        transforms.ToTensor(),
        normalize,
    ])
    try:
        train_dataset = datasets.ImageFolder(f"dataset/{args.split}", transform=train_transform, allow_empty=True)
    except TypeError:
        train_dataset = datasets.ImageFolder(f"dataset/{args.split}", transform=train_transform)
    train_loader = build_loader(train_dataset, batch_size, True, args)

    val_loader = None
    if val_enabled:
        val_transform = transforms.Compose([
            transforms.Resize(416 if args.model_size == "s" else 512),
            transforms.CenterCrop(384 if args.model_size == "s" else 480),
            transforms.ToTensor(),
            normalize,
        ])
        val_dataset = ImageFolderAllowEmpty(f"dataset/{args.val_split}", transform=val_transform)
        val_loader = build_loader(val_dataset, args.val_batch_size, False, args)

    print(f"Training samples: {len(train_dataset)}", flush=True)
    print(f"Steps per epoch: {len(train_loader)}", flush=True)
    if val_loader is not None:
        print(f"Validation samples: {len(val_loader.dataset)}", flush=True)

    stage_ends = stage_end_epochs(args.epochs)
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
                "best_epoch_loss": best_epoch_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "num_classes": num_classes,
                "class_names": class_names,
                "class_names_file": classes_path.name,
            },
            str(cp_path),
        )
        print(f"Saved checkpoint ({reason}): {cp_path}", flush=True)

    scaler = amp.GradScaler(enabled=(device.type == "cuda"))
    writer = SummaryWriter(str(checkpoints_dir / "tb"))

    best_epoch_loss = float("inf")
    epochs_without_improvement = 0
    start_epoch = 0
    checkpoint = None

    if args.load_checkpoint is not None:
        cp_path = checkpoints_dir / f"checkpoint_epoch{args.load_checkpoint}.pth"
        checkpoint = torch.load(str(cp_path), map_location=device)
        model_ref = model.module if isinstance(model, nn.DataParallel) else model
        model_state = checkpoint["model"]
        model_state = adapt_classifier_state_dict(model_state, num_classes)
        model_state = normalize_state_dict_for_model(model_state, model_ref)
        model_ref.load_state_dict(model_state)
        start_epoch = int(checkpoint.get("epoch", args.load_checkpoint))
        best_epoch_loss = float(checkpoint.get("best_epoch_loss", float("inf")))
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
            apply_stage_freezing(model, stage_i, len(args.epochs))
            optimizer = RMSprop(
                params=filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
                alpha=0.9,
                eps=1e-8,
                weight_decay=args.weight_decay,
                momentum=args.momentum,
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

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(
                f"Entering stage {stage_i + 1}/{len(args.epochs)} at epoch {epoch + 1} | "
                f"trainable params: {trainable_params}",
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

                    with amp.autocast(enabled=(device.type == "cuda")):
                        outputs = model(inputs)
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
                if loader_worker_failed and args.num_workers > 0 and not epoch_retried_for_loader:
                    print(
                        "DataLoader worker failure detected. "
                        f"Falling back to num_workers={max(args.num_workers - 1, 0)}, pin_memory=False and restarting epoch.",
                        flush=True,
                    )
                    args.num_workers = max(args.num_workers - 1, 0)
                    args.pin_memory = False
                    train_loader = build_loader(train_dataset, batch_size, True, args)
                    if val_enabled:
                        val_loader = build_loader(val_dataset, args.val_batch_size, False, args)
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

        metric_name = "train/loss"
        metric_value = epoch_loss
        if val_loader is not None:
            val_loss, val_top1, val_top5 = evaluate(model, val_loader, criterion, device)
            writer.add_scalar("val/loss_epoch", val_loss, epoch)
            writer.add_scalar("val/top1", val_top1, epoch)
            writer.add_scalar("val/top5", val_top5, epoch)
            print(
                f"Validation | loss={val_loss:.6f} | top1={val_top1:.3f}% | top5={val_top5:.3f}%",
                flush=True,
            )
            metric_name = "val/loss"
            metric_value = val_loss

        improved = (best_epoch_loss - metric_value) > args.early_stop_min_delta
        if improved:
            best_epoch_loss = metric_value
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(
                f"No significant improvement for {epochs_without_improvement} epoch(s). "
                f"Best {metric_name}: {best_epoch_loss:.6f}",
                flush=True,
            )

        if (epoch + 1) % args.checkpoint_every == 0:
            save_checkpoint(epoch + 1, stage_i, "periodic")

        last_completed_epoch = epoch + 1

        if epochs_without_improvement >= args.early_stop_patience:
            save_checkpoint(epoch + 1, stage_i, "early-stop")
            print(
                f"Early stopping at epoch {epoch + 1}: no {metric_name} improvement "
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