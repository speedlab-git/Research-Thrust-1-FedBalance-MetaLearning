# ============================================================
# crime_report_prior_mlp.py
# Train a simple crime-report "prior" model:
#   p_prior(y | hour, day_of_week, month, location_description)
#
# Input columns (default):
#   - Hour (int or string)
#   - Day (e.g., Sunday)
#   - Month (e.g., February)
#   - Location Description (e.g., STREET, BAR OR TAVERN)
#
# Label column:
#   - Primary Type (crime category)
#
# Run:
# python crime_report_prior_mlp.py \
#   --csv_path chicago_crime.csv \
#   --label_col "Primary Type" \
#   --loc_col "Location Description" \
#   --month_col Month --day_col Day --hour_col Hour \
#   --epochs 10 --batch_size 256 --lr 1e-3 \
#   --save_dir ./saved_crime_prior
# ============================================================

import argparse
import os
import json
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, roc_auc_score


# -----------------------------
# Args
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Crime-report prior model (MLP+Embeddings)")
    p.add_argument("--csv_path", type=str, required=True)

    p.add_argument("--label_col", type=str, default="Primary Type")
    p.add_argument("--loc_col", type=str, default="Location Description")
    p.add_argument("--month_col", type=str, default="Month")
    p.add_argument("--day_col", type=str, default="Day")
    p.add_argument("--hour_col", type=str, default="Hour")

    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=12)

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--emb_dim", type=int, default=16, help="embedding dim for categorical features")
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)

    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--save_dir", type=str, default="./saved_crime_prior")
    return p.parse_args()


# -----------------------------
# Seed
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Category encoding helpers
# -----------------------------
def fit_category_map(values: List[str]) -> Dict[str, int]:
    uniq = sorted(pd.Series(values).astype(str).fillna("UNK").unique().tolist())
    return {v: i for i, v in enumerate(uniq)}

def encode_with_map(values: List[str], mapping: Dict[str, int]) -> np.ndarray:
    unk_id = mapping.get("UNK", None)
    if unk_id is None:
        # If no UNK token, map unseen to 0
        unk_id = 0
    out = []
    for v in values:
        s = str(v) if pd.notna(v) else "UNK"
        out.append(mapping.get(s, unk_id))
    return np.array(out, dtype=np.int64)

def compute_class_weights(y: np.ndarray, num_classes: int, device: str) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    w = np.ones(num_classes, dtype=np.float32)
    nz = counts > 0
    w[nz] = counts[nz].sum() / counts[nz]
    w = w / (w.mean() + 1e-8)
    return torch.tensor(w, dtype=torch.float32, device=device)


# -----------------------------
# Dataset
# -----------------------------
class CrimePriorDataset(Dataset):
    def __init__(self, x_hour, x_day, x_month, x_loc, y):
        self.x_hour = torch.tensor(x_hour, dtype=torch.long)
        self.x_day = torch.tensor(x_day, dtype=torch.long)
        self.x_month = torch.tensor(x_month, dtype=torch.long)
        self.x_loc = torch.tensor(x_loc, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "hour": self.x_hour[idx],
            "day": self.x_day[idx],
            "month": self.x_month[idx],
            "loc": self.x_loc[idx],
            "y": self.y[idx]
        }


# -----------------------------
# Model: Embeddings + MLP
# -----------------------------
class PriorMLP(nn.Module):
    def __init__(self, n_hour: int, n_day: int, n_month: int, n_loc: int,
                 emb_dim: int, hidden_dim: int, dropout: float, num_classes: int):
        super().__init__()
        self.emb_hour = nn.Embedding(n_hour, emb_dim)
        self.emb_day = nn.Embedding(n_day, emb_dim)
        self.emb_month = nn.Embedding(n_month, emb_dim)
        self.emb_loc = nn.Embedding(n_loc, emb_dim)

        in_dim = emb_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, hour, day, month, loc):
        z = torch.cat([
            self.emb_hour(hour),
            self.emb_day(day),
            self.emb_month(month),
            self.emb_loc(loc)
        ], dim=1)
        return self.mlp(z)


# -----------------------------
# Train / Eval
# -----------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device: str) -> float:
    model.train()
    losses = []
    for b in loader:
        hour = b["hour"].to(device)
        day = b["day"].to(device)
        month = b["month"].to(device)
        loc = b["loc"].to(device)
        y = b["y"].to(device)

        logits = model(hour, day, month, loc)
        loss = loss_fn(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(model, loader, num_classes: int, device: str) -> Dict[str, float]:
    model.eval()
    probs_list, y_list = [], []

    for b in loader:
        hour = b["hour"].to(device)
        day = b["day"].to(device)
        month = b["month"].to(device)
        loc = b["loc"].to(device)
        y = b["y"].cpu().numpy()

        logits = model(hour, day, month, loc)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        probs_list.append(probs)
        y_list.append(y)

    y_true = np.concatenate(y_list)
    y_prob = np.concatenate(probs_list)
    y_pred = np.argmax(y_prob, axis=1)

    acc = float((y_pred == y_true).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))

    # AUC OvR using binarized labels
    Y = label_binarize(y_true, classes=np.arange(num_classes))

    def safe_auc(avg: str) -> float:
        try:
            return float(roc_auc_score(Y, y_prob, average=avg))
        except ValueError:
            return float("nan")

    macro_auc = safe_auc("macro")
    micro_auc = safe_auc("micro")
    weighted_auc = safe_auc("weighted")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "macro_auc": macro_auc,
        "micro_auc": micro_auc,
        "weighted_auc": weighted_auc,
    }


# -----------------------------
# Save
# -----------------------------
def save_bundle(save_dir: str, model: nn.Module, meta: dict):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    df = pd.read_csv(args.csv_path)

    needed = [args.label_col, args.loc_col, args.month_col, args.day_col, args.hour_col]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'. Found: {list(df.columns)}")

    df = df.dropna(subset=needed).copy()

    # Convert to strings for categorical encoding
    df[args.loc_col] = df[args.loc_col].astype(str)
    df[args.month_col] = df[args.month_col].astype(str)
    df[args.day_col] = df[args.day_col].astype(str)
    df[args.hour_col] = df[args.hour_col].astype(int)

    # Label mapping (Primary Type)
    label_map = fit_category_map(df[args.label_col].astype(str).tolist())
    y = encode_with_map(df[args.label_col].astype(str).tolist(), label_map)
    num_classes = len(label_map)

    # Feature maps
    # Hour: treat as categorical 0..23 (still use mapping for safety)
    hour_map = {str(h): h for h in range(24)}
    x_hour = df[args.hour_col].astype(int).clip(0, 23).values.astype(np.int64)

    day_map = fit_category_map(df[args.day_col].tolist())
    month_map = fit_category_map(df[args.month_col].tolist())

    # For location, add UNK token explicitly (helps if later you map tweet locations)
    loc_values = df[args.loc_col].tolist()
    if "UNK" not in set(loc_values):
        loc_values = loc_values + ["UNK"]
    loc_map = fit_category_map(loc_values)

    x_day = encode_with_map(df[args.day_col].tolist(), day_map)
    x_month = encode_with_map(df[args.month_col].tolist(), month_map)
    x_loc = encode_with_map(df[args.loc_col].tolist(), loc_map)

    # Show distributions
    print(f"\nNum classes (Primary Type): {num_classes}")
    inv_label = {v: k for k, v in label_map.items()}
    counts = np.bincount(y, minlength=num_classes)
    top = np.argsort(-counts)[:15]
    print("Top label counts:")
    for i in top:
        print(f"  {inv_label[i]:20s}: {int(counts[i])}")

    # Stratified split
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    train_ds = CrimePriorDataset(x_hour[train_idx], x_day[train_idx], x_month[train_idx], x_loc[train_idx], y[train_idx])
    test_ds  = CrimePriorDataset(x_hour[test_idx],  x_day[test_idx],  x_month[test_idx],  x_loc[test_idx],  y[test_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = PriorMLP(
        n_hour=24,
        n_day=len(day_map),
        n_month=len(month_map),
        n_loc=len(loc_map),
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_classes=num_classes
    ).to(device)

    class_w = compute_class_weights(y[train_idx], num_classes, device)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_macro_f1 = -1.0

    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        metrics = evaluate(model, test_loader, num_classes, device)

        print(
            f"[Epoch {ep:02d}/{args.epochs}] "
            f"TrainLoss={tr_loss:.4f} | "
            f"Acc={metrics['accuracy']:.4f} | "
            f"Macro-F1={metrics['macro_f1']:.4f} Micro-F1={metrics['micro_f1']:.4f} W-F1={metrics['weighted_f1']:.4f} | "
            f"Macro-AUC={metrics['macro_auc']:.4f} Micro-AUC={metrics['micro_auc']:.4f} W-AUC={metrics['weighted_auc']:.4f}"
        )

        # save best
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            meta = {
                "label_col": args.label_col,
                "feature_cols": {
                    "hour_col": args.hour_col,
                    "day_col": args.day_col,
                    "month_col": args.month_col,
                    "loc_col": args.loc_col,
                },
                "label_map": label_map,
                "day_map": day_map,
                "month_map": month_map,
                "loc_map": loc_map,
                "num_classes": num_classes,
                "best_macro_f1": best_macro_f1,
                "model_args": {
                    "emb_dim": args.emb_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout
                }
            }
            save_bundle(os.path.join(args.save_dir, "best"), model, meta)
            print(f"  -> Saved BEST to {os.path.join(args.save_dir, 'best')} (Macro-F1={best_macro_f1:.4f})")

    # final save
    meta_final = {"best_macro_f1": best_macro_f1}
    save_bundle(args.save_dir, model, meta_final)
    print(f"\nSaved FINAL model to {args.save_dir}")


if __name__ == "__main__":
    main()
