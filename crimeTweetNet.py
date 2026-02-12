

import argparse
import os
import json
import copy
import random
import numpy as np
import pandas as pd
from typing import Dict, List
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, roc_auc_score


# -----------------------------
# Args
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("FedAvg FL Tweet Classification (RoBERTa/BERTweet) + Save Global")
    p.add_argument("--csv_path", type=str, required=True)

    p.add_argument("--text_col", type=str, default="tweet")
    p.add_argument("--label_col", type=str, default="category")

    p.add_argument("--model_name", type=str, default="roberta-base",
                   help="e.g., roberta-base, bert-base-uncased, vinai/bertweet-base, cardiffnlp/twitter-roberta-base")
    p.add_argument("--max_len", type=int, default=128)

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)

    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--num_clients", type=int, default=5)

    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--num_workers", type=int, default=2)

    # Saving
    p.add_argument("--save_dir", type=str, default="./saved_tweet_global_final")
    p.add_argument("--save_best", action="store_true",
                   help="Save best global model (by Macro-F1) under save_dir/best")

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
# Dataset
# -----------------------------
class TweetDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long)
        }


def compute_class_weights(labels: np.ndarray, num_classes: int, device: str):
    """
    Inverse-frequency weights, normalized to mean=1
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    w = np.ones(num_classes, dtype=np.float32)
    nz = counts > 0
    w[nz] = counts[nz].sum() / counts[nz]
    w = w / (w.mean() + 1e-8)
    return torch.tensor(w, dtype=torch.float32, device=device), counts


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
def train_one_client(model: nn.Module,
                     loader: DataLoader,
                     args,
                     device: str,
                     class_weights: torch.Tensor) -> float:
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = max(1, len(loader) * args.local_epochs)
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    losses = []
    for _ in range(args.local_epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = loss_fn(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             num_classes: int,
             device: str) -> Dict[str, float]:
    model.to(device)
    model.eval()

    probs_list, labels_list = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        y = batch["labels"].cpu().numpy()

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        probs_list.append(probs)
        labels_list.append(y)

    y_true = np.concatenate(labels_list)     # [N]
    y_prob = np.concatenate(probs_list)      # [N, C]
    y_pred = np.argmax(y_prob, axis=1)       # [N]

    acc = float((y_pred == y_true).mean())

    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))

    # AUC on binarized labels (OvR)
    Y = label_binarize(y_true, classes=np.arange(num_classes))  # [N, C]

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
# Saving helpers
# -----------------------------
def save_global_bundle(save_dir: str, model, tokenizer, label2id: Dict[str, int], id2label: Dict[int, str]):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "label_map.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    df = pd.read_csv(args.csv_path)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{args.text_col}' and '{args.label_col}'. Found: {list(df.columns)}"
        )

    df = df.dropna(subset=[args.text_col, args.label_col]).copy()
    df[args.text_col] = df[args.text_col].astype(str)
    df[args.label_col] = df[args.label_col].astype(str)

    # Label encoding
    label_names = sorted(df[args.label_col].unique().tolist())
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for name, i in label2id.items()}
    df["label_id"] = df[args.label_col].map(label2id).astype(int)

    num_classes = len(label_names)
    print(f"Classes ({num_classes}): {label2id}")

    # Show full distribution
    full_counts = Counter(df["label_id"].tolist())
    print("\nFull dataset label distribution:")
    for i in range(num_classes):
        print(f"  [{i:02d}] {id2label[i]:12s}: {full_counts.get(i, 0)}")
    print(f"  Total: {len(df)}")

    # Stratified split 80/20
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=args.seed,
        stratify=df["label_id"]
    )
    print(f"\nSplit: train={len(train_df)} | test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds = TweetDataset(
        texts=train_df[args.text_col].tolist(),
        labels=train_df["label_id"].tolist(),
        tokenizer=tokenizer,
        max_len=args.max_len
    )
    test_ds = TweetDataset(
        texts=test_df[args.text_col].tolist(),
        labels=test_df["label_id"].tolist(),
        tokenizer=tokenizer,
        max_len=args.max_len
    )

    # IID split train among clients
    idx_all = np.arange(len(train_ds))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx_all)
    splits = np.array_split(idx_all, args.num_clients)

    client_loaders = []
    client_sizes = []
    for cid, s in enumerate(splits):
        subset = Subset(train_ds, s.tolist())
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        client_loaders.append(loader)
        client_sizes.append(len(subset))
        print(f"Client {cid}: {len(subset)} train samples")

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Class weights computed from TRAIN split
    class_w, train_counts = compute_class_weights(train_df["label_id"].values, num_classes, device)
    print("\nTrain label distribution (for weights):")
    for i in range(num_classes):
        print(f"  [{i:02d}] {id2label[i]:12s}: {int(train_counts[i])}")
    print(f"Class weights (normalized): {class_w.detach().cpu().numpy().round(3).tolist()}")

    # Build GLOBAL model once
    global_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id
    )

    # Track best
    best_macro_f1 = -1.0

    # Save config + tokenizer right away (optional)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "label_map.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
    tokenizer.save_pretrained(args.save_dir)

    for rnd in range(1, args.rounds + 1):
        # snapshot global weights
        global_state = get_state(global_model)

        client_states = []
        client_losses = []

        for cid in range(args.num_clients):
            # IMPORTANT: deepcopy avoids repeated "newly initialized head" warnings
            local_model = copy.deepcopy(global_model)
            set_state(local_model, global_state)  # safe; ensures exact same start

            loss = train_one_client(local_model, client_loaders[cid], args, device, class_w)
            client_losses.append(loss)
            client_states.append(get_state(local_model))

        # FedAvg aggregation
        new_global_state = fedavg(client_states, client_sizes)
        set_state(global_model, new_global_state)

        # Weighted avg train loss
        total = float(sum(client_sizes))
        round_loss = float(sum((sz / total) * l for sz, l in zip(client_sizes, client_losses)))

        # Eval
        metrics = evaluate(global_model, test_loader, num_classes, device)

        print(
            f"[Round {rnd:02d}/{args.rounds}] "
            f"TrainLoss={round_loss:.4f} | "
            f"Acc={metrics['accuracy']:.4f} | "
            f"Macro-F1={metrics['macro_f1']:.4f} Micro-F1={metrics['micro_f1']:.4f} W-F1={metrics['weighted_f1']:.4f} | "
            f"Macro-AUC={metrics['macro_auc']:.4f} Micro-AUC={metrics['micro_auc']:.4f} W-AUC={metrics['weighted_auc']:.4f}"
        )

        # Save LAST global model every round (crash-safe overwrite)
        save_global_bundle(args.save_dir, global_model, tokenizer, label2id, id2label)

        # Save BEST by Macro-F1
        if args.save_best and metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_dir = os.path.join(args.save_dir, "best")
            save_global_bundle(best_dir, global_model, tokenizer, label2id, id2label)
            print(f"  -> Saved BEST (Macro-F1={best_macro_f1:.4f}) to: {best_dir}")

    # Final save (redundant but explicit)
    save_global_bundle(args.save_dir, global_model, tokenizer, label2id, id2label)
    print(f"\nSaved FINAL global model to: {args.save_dir}")
    if args.save_best:
        print(f"Best Macro-F1: {best_macro_f1:.4f} (saved under {os.path.join(args.save_dir, 'best')})")


if __name__ == "__main__":
    main()
