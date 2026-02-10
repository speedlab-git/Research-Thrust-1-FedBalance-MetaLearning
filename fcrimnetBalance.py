# ============================================================
# fixmatch_fedavg_ucfcrime_resnet50_logged.py
# FixMatch-style Semi-Supervised Federated Learning (IID, 5 clients)
# - 20% labeled per client, 80% unlabeled
# - FixMatch unlabeled loss: weak -> pseudo-label, strong -> enforce
# - FedAvg aggregation
# - ResNet-50 pretrained
# - Clips NormalVideos in TRAIN to 80,000 samples (before client split)
# - Logs per-round losses + F1 (macro/micro/weighted) + AUC (macro/micro/weighted) to JSON
#
# Run example:
# python fixmatch_fedavg_ucfcrime_resnet50_logged.py \
#   --train_dir ../input/ucf-crime-dataset/Train \
#   --test_dir  ../input/ucf-crime-dataset/Test \
#   --num_clients 5 --rounds 20 --local_epochs 1 \
#   --labeled_frac 0.2 --batch_size 64 --mu 2 \
#   --lr 3e-4 --lambda_u 1.0 --tau 0.95 \
#   --out_json results_fixmatch_fedavg.json
# ============================================================

import argparse
import json
import time
import random
import numpy as np
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.metrics import f1_score, roc_auc_score


# -----------------------------
# Args
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("FixMatch-FedAvg on UCF-Crime frames (ResNet-50) + JSON logging")
    p.add_argument("--train_dir", type=str, required=True)
    p.add_argument("--test_dir", type=str, required=True)

    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--mu", type=int, default=2, help="unlabeled batch multiplier (FixMatch uses mu*B)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--local_epochs", type=int, default=2)
    p.add_argument("--num_clients", type=int, default=5)

    p.add_argument("--num_classes", type=int, default=14)
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--num_workers", type=int, default=2)

    # Semi-supervision knobs
    p.add_argument("--labeled_frac", type=float, default=0.2, help="fraction of labeled data per client")
    p.add_argument("--lambda_u", type=float, default=1.0, help="weight for unlabeled FixMatch loss")
    p.add_argument("--tau", type=float, default=0.90, help="confidence threshold for pseudo-labeling")

    # Optional: freeze backbone
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze all ResNet layers except final FC head")

    # Train clipping for NormalVideos (high imbalance)
    p.add_argument("--normal_class_name", type=str, default="NormalVideos")
    p.add_argument("--normal_cap", type=int, default=80000, help="cap for NormalVideos samples in TRAIN only")

    # Output JSON
    p.add_argument("--out_json", type=str, default="results_fixmatch_fedavg.json")
    return p.parse_args()


# -----------------------------
# Seed
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.is_available() and torch.cuda.manual_seed_all(seed)


# -----------------------------
# Dataset utilities
# -----------------------------
def count_by_class_from_targets(targets, num_classes):
    c = Counter(targets)
    return [c.get(k, 0) for k in range(num_classes)]

def print_class_counts(title, class_names, counts):
    print(f"\n{title}")
    for i, (name, cnt) in enumerate(zip(class_names, counts)):
        print(f"  [{i:02d}] {name:15s} : {cnt}")
    print(f"  Total: {sum(counts)}")

def stratified_labeled_split(indices, targets, num_classes, labeled_frac, seed):
    """
    Given a list of indices (client subset), select labeled_frac per class (stratified).
    The rest are unlabeled.
    """
    rng = np.random.default_rng(seed)
    per_class = defaultdict(list)
    for idx in indices:
        y = targets[idx]
        per_class[y].append(idx)

    labeled_idx = []
    unlabeled_idx = []

    for k in range(num_classes):
        cls_idx = per_class.get(k, [])
        if len(cls_idx) == 0:
            continue
        rng.shuffle(cls_idx)
        m = int(np.floor(labeled_frac * len(cls_idx)))
        if labeled_frac > 0 and m == 0:
            m = 1  # ensure at least 1 labeled if class exists
        labeled_idx.extend(cls_idx[:m])
        unlabeled_idx.extend(cls_idx[m:])

    rng.shuffle(labeled_idx)
    rng.shuffle(unlabeled_idx)
    return labeled_idx, unlabeled_idx


# -----------------------------
# FixMatch unlabeled view dataset
# -----------------------------
class FixMatchUnlabeledView(datasets.ImageFolder):
    """
    Same files as ImageFolder, but returns (weak_view, strong_view).
    The label is ignored for unlabeled training.
    """
    def __init__(self, root, weak_transform, strong_transform):
        super().__init__(root=root, transform=None)
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __getitem__(self, index):
        path, _y = self.samples[index]
        img = self.loader(path)
        x_w = self.weak_transform(img)
        x_s = self.strong_transform(img)
        return x_w, x_s


def build_loaders(args):
    # ImageNet normalization for pretrained ResNet
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # Weak augmentation (labeled + unlabeled weak branch)
    weak_tfms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Strong augmentation (unlabeled strong branch)
    strong_tfms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25),
    ])

    test_tfms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_labeled_ds = datasets.ImageFolder(args.train_dir, transform=weak_tfms)
    train_unlabeled_ds = FixMatchUnlabeledView(
        root=args.train_dir,
        weak_transform=weak_tfms,
        strong_transform=strong_tfms
    )
    test_ds = datasets.ImageFolder(args.test_dir, transform=test_tfms)

    # Print total samples per class (raw)
    class_names = [c for c, _ in sorted(train_labeled_ds.class_to_idx.items(), key=lambda x: x[1])]
    train_counts = count_by_class_from_targets(train_labeled_ds.targets, args.num_classes)
    test_counts  = count_by_class_from_targets(test_ds.targets, args.num_classes)
    print_class_counts("Train set class distribution (raw):", class_names, train_counts)
    print_class_counts("Test set class distribution:",        class_names, test_counts)

    # -----------------------------
    # Clip NormalVideos in TRAIN only (before client split)
    # -----------------------------
    if args.normal_class_name not in train_labeled_ds.class_to_idx:
        raise ValueError(
            f"normal_class_name='{args.normal_class_name}' not found in class_to_idx keys: "
            f"{list(train_labeled_ds.class_to_idx.keys())}"
        )

    normal_class_id = train_labeled_ds.class_to_idx[args.normal_class_name]
    targets = np.array(train_labeled_ds.targets)

    normal_idx = np.where(targets == normal_class_id)[0]
    crime_idx  = np.where(targets != normal_class_id)[0]

    rng = np.random.default_rng(args.seed)
    rng.shuffle(normal_idx)

    original_normal = int((targets == normal_class_id).sum())
    if len(normal_idx) > args.normal_cap:
        normal_idx = normal_idx[:args.normal_cap]

    final_idx = np.concatenate([crime_idx, normal_idx])
    rng.shuffle(final_idx)

    print(f"\n[Clipping] {args.normal_class_name} reduced from {original_normal} → {len(normal_idx)}")
    print(f"[Clipping] Total training samples after clipping: {len(final_idx)}")

    # IID split across clients using the clipped index pool
    idx_all = final_idx
    rng.shuffle(idx_all)
    client_splits = np.array_split(idx_all, args.num_clients)

    client_labeled_loaders = []
    client_unlabeled_loaders = []
    client_sizes = []  # FedAvg weights (use total client data in clipped pool)

    for cid, split in enumerate(client_splits):
        split = split.tolist()
        lab_idx, unlab_idx = stratified_labeled_split(
            split, train_labeled_ds.targets, args.num_classes, args.labeled_frac, seed=args.seed + cid
        )

        labeled_subset = Subset(train_labeled_ds, lab_idx)
        unlabeled_subset = Subset(train_unlabeled_ds, unlab_idx)

        labeled_loader = DataLoader(
            labeled_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True if len(labeled_subset) >= args.batch_size else False
        )

        unlabeled_loader = DataLoader(
            unlabeled_subset,
            batch_size=args.batch_size * args.mu,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True if len(unlabeled_subset) >= args.batch_size * args.mu else False
        )

        client_labeled_loaders.append(labeled_loader)
        client_unlabeled_loaders.append(unlabeled_loader)
        client_sizes.append(len(split))

        print(f"\nClient {cid}: total={len(split)} | labeled={len(lab_idx)} | unlabeled={len(unlab_idx)}")

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return train_labeled_ds, test_ds, class_names, client_labeled_loaders, client_unlabeled_loaders, client_sizes, test_loader


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
# FixMatch local training + loss logging
# -----------------------------
def compute_class_weights_from_labeled_loader(labeled_loader, num_classes: int, device: str):
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, y in labeled_loader:
        y_np = y.numpy()
        for k in y_np:
            counts[k] += 1

    weights = np.ones(num_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = (counts[nonzero].sum() / counts[nonzero]).astype(np.float32)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_client_fixmatch(model: nn.Module,
                              labeled_loader: DataLoader,
                              unlabeled_loader: DataLoader,
                              args,
                              device: str):
    model.to(device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # class-balanced supervised CE
    class_w = compute_class_weights_from_labeled_loader(labeled_loader, args.num_classes, device)
    sup_criterion = nn.CrossEntropyLoss(weight=class_w)

    unlab_iter = iter(unlabeled_loader) if len(unlabeled_loader) > 0 else None

    sup_losses, unsup_losses, total_losses = [], [], []

    for _ in range(args.local_epochs):
        for x_lab, y_lab in labeled_loader:
            x_lab = x_lab.to(device, non_blocking=True)
            y_lab = y_lab.to(device, non_blocking=True)

            # ----- supervised -----
            logits_lab = model(x_lab)
            loss_sup = sup_criterion(logits_lab, y_lab)

            # ----- unlabeled FixMatch -----
            loss_u = torch.tensor(0.0, device=device)
            if unlab_iter is not None:
                try:
                    x_w, x_s = next(unlab_iter)
                except StopIteration:
                    unlab_iter = iter(unlabeled_loader)
                    x_w, x_s = next(unlab_iter)

                x_w = x_w.to(device, non_blocking=True)
                x_s = x_s.to(device, non_blocking=True)

                with torch.no_grad():
                    q = torch.softmax(model(x_w), dim=1)
                    conf, pseudo = torch.max(q, dim=1)
                    mask = conf.ge(args.tau).float()

                logits_s = model(x_s)
                per_sample = nn.functional.cross_entropy(logits_s, pseudo, reduction="none")

                if mask.sum() > 0:
                    loss_u = (per_sample * mask).sum() / mask.sum()

            loss = loss_sup + args.lambda_u * loss_u

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            sup_losses.append(loss_sup.item())
            unsup_losses.append(loss_u.item())
            total_losses.append(loss.item())

    return {
        "loss_sup": float(np.mean(sup_losses)) if sup_losses else 0.0,
        "loss_unsup": float(np.mean(unsup_losses)) if unsup_losses else 0.0,
        "loss_total": float(np.mean(total_losses)) if total_losses else 0.0,
    }


# -----------------------------
# Evaluation (F1 + AUC: macro/micro/weighted)
# -----------------------------
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

    y_true = np.concatenate(labels_list)     # [N]
    y_prob = np.concatenate(probs_list)      # [N, C]
    y_pred = np.argmax(y_prob, axis=1)

    # F1 variants
    macro_f1    = f1_score(y_true, y_pred, average="macro")
    micro_f1    = f1_score(y_true, y_pred, average="micro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    # AUC variants (multiclass OvR)
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

    return macro_f1, micro_f1, weighted_f1, macro_auc, micro_auc, weighted_auc


# -----------------------------
# JSON saver
# -----------------------------
def save_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"FixMatch setting: labeled_frac={args.labeled_frac}, mu={args.mu}, lambda_u={args.lambda_u}, tau={args.tau}")
    print(f"Train clipping: {args.normal_class_name} cap={args.normal_cap} (TRAIN only)")
    print(f"Logging to: {args.out_json}")

    _, _, class_names, labeled_loaders, unlabeled_loaders, client_sizes, test_loader = build_loaders(args)
    print(f"\nClient weights for FedAvg (total client train data sizes): {client_sizes}")
    print(f"Freeze backbone: {args.freeze_backbone}")

    global_model = build_model(args.num_classes, args.freeze_backbone)

    results = {
        "config": vars(args),
        "rounds": {}
    }
    start_time = time.time()

    for rnd in range(1, args.rounds + 1):
        global_state = get_state(global_model)

        client_states = []
        client_losses = []

        for cid in range(args.num_clients):
            local_model = build_model(args.num_classes, args.freeze_backbone)
            set_state(local_model, global_state)

            losses = train_one_client_fixmatch(
                local_model,
                labeled_loaders[cid],
                unlabeled_loaders[cid],
                args,
                device
            )

            client_losses.append(losses)
            client_states.append(get_state(local_model))

        # FedAvg aggregation
        new_global_state = fedavg(client_states, client_sizes)
        set_state(global_model, new_global_state)

        # Evaluate global model
        macro_f1, micro_f1, weighted_f1, macro_auc, micro_auc, weighted_auc = evaluate(
            global_model, test_loader, args.num_classes, device
        )

        # Average losses across clients
        avg_sup = float(np.mean([c["loss_sup"] for c in client_losses]))
        avg_unsup = float(np.mean([c["loss_unsup"] for c in client_losses]))
        avg_total = float(np.mean([c["loss_total"] for c in client_losses]))

        elapsed = time.time() - start_time

        print(
            f"[Round {rnd:02d}/{args.rounds}] "
            f"SupLoss={avg_sup:.4f} | UnsupLoss={avg_unsup:.4f} | TotalLoss={avg_total:.4f} | "
            f"Macro-F1={macro_f1:.4f} | Micro-F1={micro_f1:.4f} | W-F1={weighted_f1:.4f} | "
            f"Macro-AUC={macro_auc:.4f} | Micro-AUC={micro_auc:.4f} | W-AUC={weighted_auc:.4f}"
        )

        results["rounds"][f"round_{rnd}"] = {
            "loss_supervised": avg_sup,
            "loss_unlabeled": avg_unsup,
            "loss_total": avg_total,

            "macro_f1": float(macro_f1),
            "F1": float(micro_f1),
            "weighted_f1": float(weighted_f1),

            "macro_auc": float(macro_auc),
            "AUC": float(micro_auc),
            "weighted_auc": float(weighted_auc),

            "elapsed_time_sec": float(elapsed)
        }

        save_json(args.out_json, results)

    print("\nTraining complete.")
    print(f"Saved results to: {args.out_json}")


if __name__ == "__main__":
    main()
