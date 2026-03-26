import math
import os
import pickle
import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader, has_file_allowed_extension
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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

# abstention
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

# helper
def get_num_classes_from_checkpoint(checkpoint):
    """Read num_classes stored in checkpoint, or infer from classifier weight shape."""
    if "num_classes" in checkpoint:
        return int(checkpoint["num_classes"])
    state = checkpoint["model"]
    for prefix in ("", "module."):
        key = f"{prefix}classifier.weight"
        if key in state:
            return state[key].shape[0]
    raise KeyError("Cannot determine num_classes from checkpoint.")


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


def instance_bin(n):
    if n < 5:  return 0
    if n < 10: return 1
    if n < 20: return 2
    if n < 50: return 3
    return 4

# main
abstentions = {"confidence": confidence_mask, "margin": margin_mask, "entropy": entropy_mask}
thresholds = [
    0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
    0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999,
]

model_size = sys.argv[1]
checkpoints_dir = sys.argv[2]
start_epoch = int(sys.argv[3])
end_epoch = int(sys.argv[4])

print("=" * 80, flush=True)
print("Starting validation", flush=True)
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)
print(f"Model: tf_efficientnetv2_{model_size}", flush=True)
print(f"Checkpoints dir: {checkpoints_dir}", flush=True)
print(f"Epoch range: {start_epoch} -> {end_epoch} (step 5)", flush=True)
print("=" * 80, flush=True)

# detect num_classes from first checkpoint
first_ckpt_path = Path(checkpoints_dir) / f"checkpoint_epoch{start_epoch}.pth"
first_ckpt = torch.load(str(first_ckpt_path), map_location="cpu")
num_classes = get_num_classes_from_checkpoint(first_ckpt)
del first_ckpt
print(f"✓ Detected num_classes={num_classes} from checkpoint", flush=True)

model = timm.create_model(
    f"tf_efficientnetv2_{model_size}",
    pretrained=False,
    num_classes=num_classes,
).cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
print(f"✓ Model created", flush=True)

# data pipeline
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
val_transform = transforms.Compose([
    transforms.Resize(416 if model_size == "s" else 512),
    transforms.CenterCrop(384 if model_size == "s" else 480),
    transforms.ToTensor(),
    normalize,
])
val_dataset = ImageFolderAllowEmpty("dataset/test", transform=val_transform)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False,
)
print(f"✓ Data loader ready: {len(val_dataset)} images, {len(val_loader)} batches", flush=True)

# instance count bins
instance_count = pickle.load(open("instance_count.pkl", "rb"))
print(f"✓ Loaded instance_count.pkl ({len(instance_count)} classes)", flush=True)

class_bins = {
    v: instance_bin(instance_count.get(k, 0))
    for k, v in val_dataset.class_to_idx.items()
}

# TensorBoard writers
writer_summary = SummaryWriter(f"{checkpoints_dir}/test/summary_test")
writers_abstention = {
    a: SummaryWriter(f"{checkpoints_dir}/test/{a}_test") for a in abstentions
}

# eval loop over checkpoints
for epoch in range(start_epoch, end_epoch + 1, 1): # was 5
    ckpt_path = Path(checkpoints_dir) / f"checkpoint_epoch{epoch}.pth"
    if not ckpt_path.exists():
        print(f"Checkpoint not found, skipping: {ckpt_path}", flush=True)
        continue

    print(f"\nLoading checkpoint: {ckpt_path}", flush=True)
    checkpoint = torch.load(str(ckpt_path), map_location="cuda")
    model_state = adapt_classifier_state_dict(checkpoint["model"], num_classes)
    model.load_state_dict(model_state)
    del checkpoint
    model.eval()

    out_csv = Path(checkpoints_dir) / "test" / f"test_{epoch}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

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
                val_loader,
                total=len(val_loader),
                desc=f"Epoch {epoch}",
                leave=True,
                dynamic_ncols=True,
            )
            for step, (inputs, targets) in enumerate(progress):
                inputs  = inputs.cuda()
                targets = targets.cuda()
                outputs = model(inputs)
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
    # Tail-aware metric: mean accuracy for bins 0,1,2 (1-4, 5-9, 10-19)
    tail_accs = []
    for k in range(3):
        if processed_binned[k] > 0:
            tail_accs.append(100.0 * correct_binned[k] / processed_binned[k])
    tail_metric = float(sum(tail_accs)) / max(len(tail_accs), 1)
    print(
        f"Epoch {epoch} complete | loss={avg_loss:.4f} | "
        f"top1={top_k_accs[0]:.2f}% | top5={top_k_accs[4]:.2f}% | tail_metric={tail_metric:.3f}",
        flush=True,
    )
    bin_labels = ["1-4", "5-9", "10-19", "20-49", "50+"]
    bin_accs = [
        f"{bin_labels[k]}: {100.0 * correct_binned[k] / processed_binned[k]:.1f}%"
        if processed_binned[k] > 0 else f"{bin_labels[k]}: n/a"
        for k in range(5)
    ]
    print(f"Binned accuracy    | {' | '.join(bin_accs)}", flush=True)

    # TensorBoard logging
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