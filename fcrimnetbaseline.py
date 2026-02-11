# ============================================================
# fedavg_ucfcrime_resnet50_wandb_trainfrac.py
# FedAvg on UCF-Crime frames (ResNet-50)
# - IID 5 clients
# - Clips NormalVideos in TRAIN to 80,000 (before client split)
# - Optionally uses only a fraction of the (clipped) TRAIN pool (e.g., 20%)
# - Metrics: Macro-F1, Micro-F1, Macro-AUC, Micro-AUC, Weighted-AUC
# - Logs per-round metrics + train loss to Weights & Biases (wandb)
#
# Run:
# python fedavg_ucfcrime_resnet50_wandb_trainfrac.py \
#   --train_dir ../input/ucf-crime-dataset/Train \
#   --test_dir  ../input/ucf-crime-dataset/Test \
#   --num_clients 5 --rounds 20 --local_epochs 1 \
#   --lr 3e-4 --batch_size 64 \
#   --normal_class_name NormalVideos --normal_cap 80000 \
#   --train_frac 0.2 \
#   --use_wandb --wandb_project ucfcrime-fedavg --wandb_run_name fedavg_r50_iid_20pct
#
# Note: install wandb first: pip install wandb
# and login once: wandb login
# ============================================================

import argparse
import random
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.metrics import f1_score, roc_auc_score

# W&B (optional)
try:
    import wandb
except Exception:
    wandb = None


# -----------------------------
# Args
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("FedAvg on UCF-Crime frames (ResNet-50) + W&B + train_frac")
    p.add_argument("--train_dir", type=str, required=True)
    p.add_argument("--test_dir", type=str, required=True)

    p.add_argument("--img_size", type=int, default=224, help="ResNet default is 224")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4, help="ResNet fine-tune LR (SGD)")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--num_clients", type=int, default=5)

    p.add_argument("--num_classes", type=int, default=14)
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--num_workers", type=int, default=2)

    # Optional: freeze backbone
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze all ResNet layers except final FC head")

    # Clip NormalVideos in TRAIN only
    p.add_argument("--normal_class_name", type=str, default="NormalVideos")
    p.add_argument("--normal_cap", type=int, default=80000)

    # Use only a fraction of the (clipped) training pool
    p.add_argument("--train_frac", type=float, default=0.2,
                   help="fraction of clipped TRAIN pool to use (e.g., 0.2 = 20%)")

    # W&B
    p.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", type=str, default="ucfcrime-fedavg")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)

    return p.parse_args()


# -----------------------------
# Seed
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Dataset utilities
# -----------------------------
def count_by_class(targets, num_classes: int):
    c = Counter(targets)
    return [c.get(k, 0) for k in range(num_classes)]


# -----------------------------
# Data
# -----------------------------
def build_loaders(args):
    # ImageNet normalization for pretrained ResNet
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_tfms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(args.train_dir, transform=train_tfms)
    test_ds  = datasets.ImageFolder(args.test_dir,  transform=test_tfms)

    # -------- Clip NormalVideos in TRAIN only --------
    if args.normal_class_name not in train_ds.class_to_idx:
        raise ValueError(
            f"normal_class_name='{args.normal_class_name}' not found. "
            f"Available: {list(train_ds.class_to_idx.keys())}"
        )
    normal_id = train_ds.class_to_idx[args.normal_class_name]
    targets = np.array(train_ds.targets)

    normal_idx = np.where(targets == normal_id)[0]
    crime_idx  = np.where(targets != normal_id)[0]

    rng = np.random.default_rng(args.seed)
    rng.shuffle(normal_idx)

    original_normal = int((targets == normal_id).sum())
    if len(normal_idx) > args.normal_cap:
        normal_idx = normal_idx[:args.normal_cap]

    final_idx = np.concatenate([crime_idx, normal_idx])
    rng.shuffle(final_idx)

    print(f"\n[Clipping] {args.normal_class_name} reduced from {original_normal} → {len(normal_idx)}")
    print(f"[Clipping] Total TRAIN samples after clipping: {len(final_idx)}")

    # -------- Subsample total train pool to train_frac --------
    if not (0 < args.train_frac <= 1.0):
        raise ValueError("--train_frac must be in (0, 1].")
    if args.train_frac < 1.0:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(final_idx)
        keep = int(np.floor(args.train_frac * len(final_idx)))
        keep = max(1, keep)
        final_idx = final_idx[:keep]
        print(f"[TrainFrac] Using {args.train_frac*100:.1f}% of clipped TRAIN pool: {len(final_idx)} samples")

    # Optional: show class distribution AFTER clipping + train_frac
    clipped_targets = targets[final_idx]
    clipped_counts = count_by_class(clipped_targets.tolist(), args.num_classes)
    class_names = [c for c, _ in sorted(train_ds.class_to_idx.items(), key=lambda x: x[1])]

    print("\nTrain class distribution (after clipping + train_frac):")
    for i, (n, c) in enumerate(zip(class_names, clipped_counts)):
        print(f"  [{i:02d}] {n:15s}: {c}")
    print(f"  Total: {sum(clipped_counts)}")

    # IID split across clients on final_idx pool
    idx = final_idx
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    splits = np.array_split(idx, args.num_clients)

    client_loaders = []
    client_sizes = []
    for s in splits:
        subset = Subset(train_ds, s.tolist())
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        client_loaders.append(loader)
        client_sizes.append(len(subset))

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return train_ds, test_ds, client_loaders, client_sizes, test_loader


# -----------------------------
# Model: ResNet-50 pretrained
# -----------------------------
def build_model(num_classes: int, freeze_backbone: bool) -> nn.Module:
    try:
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    except Exception:
        m = models.resnet50(pretrained=True)

    if freeze_backbone:
        for p in m.parameters():
            p.requires_grad = False

    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    for p in m.fc.parameters():
        p.requires_grad = True

    return m


# -----------------------------
# FedAvg helpers
# -----------------------------
@torch.no_grad()
def get_state(model: nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

@torch.no_grad()
def set_state(model: nn.Module, state):
    model.load_state_dict(state, strict=True)

@torch.no_grad()
def fedavg(states, weights):
    total = float(sum(weights))
    w = [float(x) / total for x in weights]
    out = {}
    for k in states[0].keys():
        out[k] = sum(wi * st[k] for wi, st in zip(w, states))
    return out


# -----------------------------
# Train / Eval
# -----------------------------
def train_one_client(model: nn.Module, loader: DataLoader, args, device: str):
    model.to(device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    losses = []

    for _ in range(args.local_epochs):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, num_classes: int, device: str):
    model.to(device)
    model.eval()

    probs_list = []
    labels_list = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        probs_list.append(probs)
        labels_list.append(y.numpy())

    y_true = np.concatenate(labels_list)   # [N]
    y_prob = np.concatenate(probs_list)    # [N, C]
    y_pred = np.argmax(y_prob, axis=1)

    # F1
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")

    # AUC (multiclass OvR)
    def safe_auc(avg):
        try:
            return roc_auc_score(
                y_true, y_prob,
                multi_class="ovr",
                average=avg,
                labels=np.arange(num_classes)
            )
        except ValueError:
            return float("nan")

    macro_auc    = safe_auc("macro")
    micro_auc    = safe_auc("micro")
    weighted_auc = safe_auc("weighted")

    return macro_f1, micro_f1, macro_auc, micro_auc, weighted_auc


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.use_wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed. Run: pip install wandb")
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args)
        )

    train_ds, test_ds, client_loaders, client_sizes, test_loader = build_loaders(args)

    print(f"\nTrain samples (raw): {len(train_ds)} | Test samples: {len(test_ds)}")
    print(f"Client sizes (IID, final pool): {client_sizes}")
    print(f"Freeze backbone: {args.freeze_backbone}")

    global_model = build_model(args.num_classes, args.freeze_backbone)

    for rnd in range(1, args.rounds + 1):
        global_state = get_state(global_model)

        client_states = []
        client_losses = []

        for cid in range(args.num_clients):
            local_model = build_model(args.num_classes, args.freeze_backbone)
            set_state(local_model, global_state)

            avg_loss = train_one_client(local_model, client_loaders[cid], args, device)
            client_losses.append(avg_loss)
            client_states.append(get_state(local_model))

        # Aggregate
        new_global_state = fedavg(client_states, client_sizes)
        set_state(global_model, new_global_state)

        # Metrics
        macro_f1, micro_f1, macro_auc, micro_auc, weighted_auc = evaluate(
            global_model, test_loader, args.num_classes, device
        )

        # Weighted avg loss across clients (by client data size)
        total = float(sum(client_sizes))
        round_loss = float(sum((sz / total) * l for sz, l in zip(client_sizes, client_losses)))

        print(
            f"[Round {rnd:02d}/{args.rounds}] "
            f"TrainLoss={round_loss:.4f} | "
            f"Macro-F1={macro_f1:.4f} | Micro-F1={micro_f1:.4f} | "
            f"Macro-AUC={macro_auc:.4f} | Micro-AUC={micro_auc:.4f} | W-AUC={weighted_auc:.4f}"
        )

        if args.use_wandb:
            wandb.log({
                "round": rnd,
                "train/loss": round_loss,
                "test/macro_f1": macro_f1,
                "test/micro_f1": micro_f1,
                "test/macro_auc": macro_auc,
                "test/micro_auc": micro_auc,
                "test/weighted_auc": weighted_auc,
            }, step=rnd)

    # Final
    macro_f1, micro_f1, macro_auc, micro_auc, weighted_auc = evaluate(
        global_model, test_loader, args.num_classes, device
    )
    print("\nFINAL:")
    print(f"Macro-F1: {macro_f1:.4f} | Micro-F1: {micro_f1:.4f}")
    print(f"Macro-AUC: {macro_auc:.4f} | Micro-AUC: {micro_auc:.4f} | Weighted-AUC: {weighted_auc:.4f}")

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
