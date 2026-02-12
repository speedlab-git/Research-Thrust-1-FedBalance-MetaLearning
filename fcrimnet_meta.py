# ============================================================
# meta_fed_ucfcrime_resnet50_wandb_trainfrac.py
# Federated Meta-Learning (FOMAML-style) on UCF-Crime frames (ResNet-50)
#
# Key idea (clients-as-tasks):
# - Each client has its own TRAIN subset (IID split like your FedAvg script)
# - For each round, for each client:
#     inner-loop: adapt on SUPPORT split (few steps)
#     outer-loop: compute QUERY loss on QUERY split
#     collect meta-gradients (w.r.t. adapted params, first-order)
# - Server aggregates meta-gradients (weighted by client sizes) and updates global model
# - Evaluate global model on TEST each round
# - Save BEST model to: meta_fed/<run_name_or_default>/best.pth
#
# Run example:
# python meta_fed_ucfcrime_resnet50_wandb_trainfrac.py \
#   --train_dir ../input/ucf-crime-dataset/Train \
#   --test_dir  ../input/ucf-crime-dataset/Test \
#   --num_clients 5 --rounds 20 \
#   --outer_lr 1e-3 --inner_lr 1e-2 --inner_steps 5 \
#   --batch_size 64 --support_ratio 0.5 \
#   --normal_class_name NormalVideos --normal_cap 80000 \
#   --train_frac 0.2 \
#   --use_wandb --wandb_project ucfcrime-meta-fed --wandb_run_name meta_fed_r50_iid_20pct
#
# Note:
# - This is "first-order MAML" approximation (no second-order derivatives).
# - Much more stable than using .data updates inside MAML.
# ============================================================

import argparse
import os
import random
import numpy as np
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.metrics import f1_score, roc_auc_score

# W&B (optional)
try:
    import wandb
except Exception:
    wandb = None

from torch.nn.utils.stateless import functional_call


# -----------------------------
# Args
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Meta-Fed (FOMAML) on UCF-Crime frames (ResNet-50) + W&B + train_frac")

    p.add_argument("--train_dir", type=str, required=True)
    p.add_argument("--test_dir", type=str, required=True)

    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--num_clients", type=int, default=5)

    # Meta-learning hyperparams
    p.add_argument("--outer_lr", type=float, default=1e-3, help="Server meta step size")
    p.add_argument("--inner_lr", type=float, default=1e-2, help="Client adaptation step size")
    p.add_argument("--inner_steps", type=int, default=5, help="How many adaptation steps on support split per round")
    p.add_argument("--support_ratio", type=float, default=0.5, help="Support/query split ratio per client")

    p.add_argument("--num_classes", type=int, default=14)
    p.add_argument("--seed", type=int, default=12)

    # Optional: freeze backbone
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze all ResNet layers except final FC head")

    # Clip NormalVideos in TRAIN only
    p.add_argument("--normal_class_name", type=str, default="NormalVideos")
    p.add_argument("--normal_cap", type=int, default=80000)

    # Use only a fraction of the (clipped) training pool
    p.add_argument("--train_frac", type=float, default=0.2,
                   help="fraction of clipped TRAIN pool to use (e.g., 0.2 = 20%)")

    # Save path
    p.add_argument("--save_root", type=str, default="meta_fed", help="Root folder to save best model")

    # W&B
    p.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", type=str, default="ucfcrime-meta-fed")
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

    # ----- Clip NormalVideos in TRAIN only -----
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

    # ----- Subsample pool to train_frac -----
    if not (0 < args.train_frac <= 1.0):
        raise ValueError("--train_frac must be in (0, 1].")
    if args.train_frac < 1.0:
        rng.shuffle(final_idx)
        keep = int(np.floor(args.train_frac * len(final_idx)))
        keep = max(1, keep)
        final_idx = final_idx[:keep]
        print(f"[TrainFrac] Using {args.train_frac*100:.1f}% of clipped TRAIN pool: {len(final_idx)} samples")

    # Show distribution after clipping+frac
    clipped_targets = targets[final_idx]
    clipped_counts = count_by_class(clipped_targets.tolist(), args.num_classes)
    class_names = [c for c, _ in sorted(train_ds.class_to_idx.items(), key=lambda x: x[1])]

    print("\nTrain class distribution (after clipping + train_frac):")
    for i, (n, c) in enumerate(zip(class_names, clipped_counts)):
        print(f"  [{i:02d}] {n:15s}: {c}")
    print(f"  Total: {sum(clipped_counts)}")

    # ----- IID split across clients -----
    idx = final_idx.copy()
    rng.shuffle(idx)
    splits = np.array_split(idx, args.num_clients)

    # For meta-learning: each client gets support/query loaders
    client_support_loaders = []
    client_query_loaders = []
    client_sizes = []

    for s in splits:
        s = s.tolist()
        rng.shuffle(s)

        n = len(s)
        n_sup = int(round(args.support_ratio * n))
        n_sup = max(1, min(n - 1, n_sup))  # ensure both non-empty
        sup_idx = s[:n_sup]
        qry_idx = s[n_sup:]

        sup_subset = Subset(train_ds, sup_idx)
        qry_subset = Subset(train_ds, qry_idx)

        sup_loader = DataLoader(
            sup_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        qry_loader = DataLoader(
            qry_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )

        client_support_loaders.append(sup_loader)
        client_query_loaders.append(qry_loader)
        client_sizes.append(n)

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return train_ds, test_ds, client_support_loaders, client_query_loaders, client_sizes, test_loader


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
# Metrics (same as your FedAvg)
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

    y_true = np.concatenate(labels_list)
    y_prob = np.concatenate(probs_list)
    y_pred = np.argmax(y_prob, axis=1)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")

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
# Meta-learning (FOMAML-style)
# -----------------------------
def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def client_fomaml_meta_gradients(
    global_model: nn.Module,
    support_loader: DataLoader,
    query_loader: DataLoader,
    inner_lr: float,
    inner_steps: int,
    device: str
):
    """
    Returns:
      query_loss (float),
      meta_grads (list[Tensor] on CPU matching global_model.parameters() order)
    """
    criterion = nn.CrossEntropyLoss()

    global_model.to(device)
    global_model.train()

    # Work with named parameters to use functional_call
    params = OrderedDict((name, p) for (name, p) in global_model.named_parameters() if p.requires_grad)

    sup_it = cycle(support_loader)
    qry_it = cycle(query_loader)

    # ---- inner loop (support) ----
    for _ in range(inner_steps):
        x, y = next(sup_it)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = functional_call(global_model, params, (x,))
        loss = criterion(logits, y)

        grads = torch.autograd.grad(loss, params.values(), create_graph=False)
        # First-order update: treat grads as constants (no second-order)
        params = OrderedDict((k, w - inner_lr * g) for (k, w), g in zip(params.items(), grads))

    # ---- outer objective (query) ----
    xq, yq = next(qry_it)
    xq = xq.to(device, non_blocking=True)
    yq = yq.to(device, non_blocking=True)

    qlogits = functional_call(global_model, params, (xq,))
    qloss = criterion(qlogits, yq)

    # meta-grads w.r.t adapted params (FOMAML approximation)
    meta_grads = torch.autograd.grad(qloss, params.values(), create_graph=False)

    # Move grads to CPU (so we can aggregate cleanly)
    meta_grads_cpu = [g.detach().cpu() for g in meta_grads]
    return float(qloss.detach().cpu().item()), meta_grads_cpu


def aggregate_grads(client_grads, client_sizes):
    """
    Weighted average of gradient lists.
    client_grads: list of list[Tensor], each list matches param order
    """
    total = float(sum(client_sizes))
    weights = [float(sz) / total for sz in client_sizes]

    agg = []
    for pi in range(len(client_grads[0])):
        g = sum(w * cg[pi] for w, cg in zip(weights, client_grads))
        agg.append(gī = g  # keep name readable in editor
        )
    return agg


def apply_meta_update(global_model: nn.Module, agg_grads, outer_lr: float):
    """
    SGD step on global params using aggregated meta-gradients.
    """
    # Only update requires_grad params in the same order we collected them
    with torch.no_grad():
        i = 0
        for p in global_model.parameters():
            if not p.requires_grad:
                continue
            p -= outer_lr * agg_grads[i].to(p.device)
            i += 1


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    run_name = args.wandb_run_name or "meta_fed_run"
    save_dir = os.path.join(args.save_root, run_name)
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "best.pth")
    print(f"Best model will be saved to: {best_path}")

    if args.use_wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed. Run: pip install wandb")
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args)
        )

    train_ds, test_ds, support_loaders, query_loaders, client_sizes, test_loader = build_loaders(args)

    print(f"\nTrain samples (raw): {len(train_ds)} | Test samples: {len(test_ds)}")
    print(f"Client sizes (IID, final pool): {client_sizes}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    print(f"Support ratio: {args.support_ratio}")

    global_model = build_model(args.num_classes, args.freeze_backbone)

    best_macro_f1 = -1.0

    for rnd in range(1, args.rounds + 1):
        # ---- collect client meta-grads ----
        client_grads = []
        client_q_losses = []

        for cid in range(args.num_clients):
            qloss, grads = client_fomaml_meta_gradients(
                global_model=global_model,
                support_loader=support_loaders[cid],
                query_loader=query_loaders[cid],
                inner_lr=args.inner_lr,
                inner_steps=args.inner_steps,
                device=device
            )
            client_q_losses.append(qloss)
            client_grads.append(grads)

        # ---- aggregate + update ----
        # (use same client_sizes from IID split)
        total = float(sum(client_sizes))
        weights = [float(sz) / total for sz in client_sizes]

        # weighted avg of query losses (for logging)
        round_q_loss = float(sum(w * l for w, l in zip(weights, client_q_losses)))

        # aggregate grads
        # (manual aggregate to avoid weird python name issues)
        agg_grads = []
        for pi in range(len(client_grads[0])):
            g = sum(w * cg[pi] for w, cg in zip(weights, client_grads))
            agg_grads.append(g)

        apply_meta_update(global_model, agg_grads, args.outer_lr)

        # ---- evaluate ----
        macro_f1, micro_f1, macro_auc, micro_auc, weighted_auc = evaluate(
            global_model, test_loader, args.num_classes, device
        )

        print(
            f"[Round {rnd:02d}/{args.rounds}] "
            f"MetaQueryLoss={round_q_loss:.4f} | "
            f"Macro-F1={macro_f1:.4f} | Micro-F1={micro_f1:.4f} | "
            f"Macro-AUC={macro_auc:.4f} | Micro-AUC={micro_auc:.4f} | W-AUC={weighted_auc:.4f}"
        )

        if args.use_wandb:
            wandb.log({
                "round": rnd,
                "meta/query_loss": round_q_loss,
                "test/macro_f1": macro_f1,
                "test/micro_f1": micro_f1,
                "test/macro_auc": macro_auc,
                "test/micro_auc": micro_auc,
                "test/weighted_auc": weighted_auc,
            }, step=rnd)

        # ---- save best ----
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(global_model.state_dict(), best_path)
            print(f"✅ New BEST saved: Macro-F1={best_macro_f1:.4f} -> {best_path}")

    # Final eval
    macro_f1, micro_f1, macro_auc, micro_auc, weighted_auc = evaluate(
        global_model, test_loader, args.num_classes, device
    )
    print("\nFINAL:")
    print(f"Macro-F1: {macro_f1:.4f} | Micro-F1: {micro_f1:.4f}")
    print(f"Macro-AUC: {macro_auc:.4f} | Micro-AUC: {micro_auc:.4f} | Weighted-AUC: {weighted_auc:.4f}")
    print(f"Best checkpoint: {best_path}")

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
