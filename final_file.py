
import argparse
import csv
import os
import random
import sys
from typing import Any, Dict, List, Sequence, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
import torchvision.transforms as transforms
import requests
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = Path("/Users/aadrishmahmood/Lead Scoring System/tml26_task1")
MODEL_PATH = BASE_DIR / "model.pt"
PUB_PATH = BASE_DIR / "pub.pt"
PRIV_PATH = BASE_DIR / "priv.pt"
OUTPUT_CSV = Path("submission.csv")

BASE_URL = "http://34.63.153.158"
API_KEY = "4a797613d5886db7d19d1785f2882002"
TASK_ID = "01-mia"

# -----------------------------
# Dataset classes 
# -----------------------------
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform
    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label
    def __len__(self): return len(self.ids)

class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []
    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]

# -----------------------------
# Utilities
# -----------------------------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

# -----------------------------
# Dataset loading
# -----------------------------
def load_and_preprocess_dataset(pt_path: str, transform=None):
    # Load the object directly (since classes are defined)
    ds = torch.load(pt_path, weights_only=False, map_location="cpu")
    ds.transform = transform
    return ds

def collate_fn_custom(batch: Sequence[Tuple]) -> Dict[str, Any]:
    # Custom collate for the MembershipDataset/TaskDataset output
    # MembershipDataset returns (id, img, label, membership) or (id, img, label)
    ids = torch.tensor([b[0] for b in batch], dtype=torch.long)
    images = torch.stack([b[1] for b in batch], dim=0)
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    
    memberships = None
    if len(batch[0]) > 3 and batch[0][3] is not None:
        memberships = torch.tensor([int(b[3]) for b in batch], dtype=torch.float32)
        
    return {"id": ids, "image": images, "label": labels, "membership": memberships}

# -----------------------------
# Model loading
# -----------------------------
def load_resnet_model(model_path: str, device: torch.device) -> nn.Module:
    model = resnet18(weights=None)
    model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(512, 9)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)

# -----------------------------
# Feature extraction (from user's code)
# -----------------------------
@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta: int = 4,
    noise_std: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_ids = []
    all_labels = []
    all_membership = []
    feat_list = []

    model.eval()

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)

        logits_runs = []
        for k in range(max(1, tta)):
            if k == 0:
                x_aug = x
            elif k % 2 == 1:
                # Horizontal Flip
                x_aug = torch.flip(x, [3])
            else:
                # Add slight noise
                noise = noise_std * torch.randn_like(x)
                x_aug = torch.clamp(x + noise, -2.5, 2.5)
            logits_runs.append(model(x_aug))
        logits_stack = torch.stack(logits_runs, dim=0)  # [T, B, C]

        logits_mean = logits_stack.mean(dim=0)
        probs = F.softmax(logits_mean, dim=1)
        log_probs = F.log_softmax(logits_mean, dim=1)

        losses = F.nll_loss(log_probs, y, reduction="none")
        true_prob = probs.gather(1, y[:, None]).squeeze(1)
        true_logit = logits_mean.gather(1, y[:, None]).squeeze(1)

        top2_probs, top2_idx = torch.topk(probs, k=2, dim=1)
        margin = top2_probs[:, 0] - top2_probs[:, 1]
        entropy = -(probs * log_probs).sum(dim=1)
        max_prob = top2_probs[:, 0]
        is_correct = (logits_mean.argmax(dim=1) == y).float()

        # Augmentation stability features
        probs_runs = F.softmax(logits_stack, dim=2)  # [T, B, C]
        true_prob_runs = probs_runs[:, torch.arange(y.shape[0], device=device), y]
        true_prob_std = true_prob_runs.std(dim=0)
        true_prob_mean = true_prob_runs.mean(dim=0)

        pred_runs = probs_runs.argmax(dim=2)
        pred_consistency = (pred_runs == pred_runs[0:1]).float().mean(dim=0)

        # Modified entropy
        one_hot = F.one_hot(y, num_classes=probs.shape[1]).float()
        rev_probs = 1.0 - probs
        rev_log_probs = torch.log(torch.clamp(rev_probs, min=1e-12))
        modified_entropy = -(
            one_hot * rev_log_probs + (1.0 - one_hot) * log_probs
        ).sum(dim=1)

        feats = torch.stack(
            [
                -losses,
                true_prob,
                true_logit,
                margin,
                -entropy,
                max_prob,
                is_correct,
                -true_prob_std,
                true_prob_mean,
                pred_consistency,
                -modified_entropy,
            ],
            dim=1,
        )

        feat_list.append(feats.cpu().numpy())
        all_ids.append(batch["id"].cpu().numpy())
        all_labels.append(batch["label"].cpu().numpy())
        if batch["membership"] is None:
            all_membership.append(np.full((x.shape[0],), -1.0, dtype=np.float32))
        else:
            all_membership.append(batch["membership"].cpu().numpy())

    return (
        np.concatenate(all_ids, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_membership, axis=0),
        np.concatenate(feat_list, axis=0),
    )

# -----------------------------
# Attack model
# -----------------------------
class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None
    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True) + 1e-8
        return self
    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

class LogisticAttack(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)

def fit_logistic_torch(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    epochs: int = 300,
) -> Tuple[LogisticAttack, float]:
    model = LogisticAttack(x_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    xtr = torch.tensor(x_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.float32, device=device)
    xva = torch.tensor(x_val, dtype=torch.float32, device=device)
    yva = torch.tensor(y_val, dtype=torch.float32, device=device)

    pos = float(max(1.0, y_train.sum()))
    neg = float(max(1.0, len(y_train) - y_train.sum()))
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_val = float("inf")
    patience = 40
    bad = 0

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(xtr)
        loss = criterion(logits, ytr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(xva), yva).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, best_val

def stratified_kfold_indices(y: np.ndarray, n_splits: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    pos_folds = np.array_split(pos_idx, n_splits)
    neg_folds = np.array_split(neg_idx, n_splits)
    for i in range(n_splits):
        val_idx = np.concatenate([pos_folds[i], neg_folds[i]])
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        yield train_idx, val_idx

def oof_and_private_scores(
    x_pub: np.ndarray,
    y_pub: np.ndarray,
    x_priv: np.ndarray,
    device: torch.device,
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(y_pub), dtype=np.float32)
    priv_probs_accum = np.zeros(len(x_priv), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(stratified_kfold_indices(y_pub, n_splits=n_splits, seed=seed), start=1):
        scaler = Standardizer()
        xtr = scaler.fit_transform(x_pub[tr_idx])
        xva = scaler.transform(x_pub[va_idx])
        xpr = scaler.transform(x_priv)

        model, _ = fit_logistic_torch(
            x_train=xtr,
            y_train=y_pub[tr_idx],
            x_val=xva,
            y_val=y_pub[va_idx],
            device=device,
        )

        model.eval()
        with torch.no_grad():
            va_logits = model(torch.tensor(xva, dtype=torch.float32, device=device))
            pr_logits = model(torch.tensor(xpr, dtype=torch.float32, device=device))
            oof[va_idx] = torch.sigmoid(va_logits).cpu().numpy()
            priv_probs_accum += torch.sigmoid(pr_logits).cpu().numpy() / n_splits

        print(f"Finished fold {fold}/{n_splits}")

    return oof, priv_probs_accum

# -----------------------------
# Blending and Evaluation
# -----------------------------
def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.05) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    return np.interp(target_fpr, fpr, tpr)

def minmax01(x: np.ndarray) -> np.ndarray:
    lo = x.min()
    hi = x.max()
    if hi - lo < 1e-12:
        return np.full_like(x, 0.5, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)

def choose_blend(y_pub: np.ndarray, base_oof: np.ndarray, raw_pub_feats: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    candidates = {
        "base": minmax01(base_oof),
        "neg_loss": minmax01(raw_pub_feats[:, 0]),
        "true_prob": minmax01(raw_pub_feats[:, 1]),
        "margin": minmax01(raw_pub_feats[:, 3]),
        "neg_entropy": minmax01(raw_pub_feats[:, 4]),
        "consistency": minmax01(raw_pub_feats[:, 9]),
    }

    best_name = "base"
    best_score = tpr_at_fpr(y_pub, candidates[best_name])
    best = candidates[best_name]

    for name, arr in candidates.items():
        sc = tpr_at_fpr(y_pub, arr)
        if sc > best_score:
            best_name = name
            best_score = sc
            best = arr

    for name, arr in candidates.items():
        if name == "base": continue
        for alpha in np.linspace(0.1, 0.9, 9):
            blend = minmax01(alpha * candidates["base"] + (1.0 - alpha) * arr)
            sc = tpr_at_fpr(y_pub, blend)
            if sc > best_score:
                best_score = sc
                best_name = f"blend_base_{name}_{alpha:.1f}"
                best = blend
    return best, {"selected": best_name, "public_tpr_at_5fpr": best_score}

def apply_same_blend(base_priv: np.ndarray, raw_priv_feats: np.ndarray, info: Dict[str, float]) -> np.ndarray:
    name = info["selected"]
    base = minmax01(base_priv)
    if name == "base": return base
    mapping = {
        "neg_loss": minmax01(raw_priv_feats[:, 0]),
        "true_prob": minmax01(raw_priv_feats[:, 1]),
        "margin": minmax01(raw_priv_feats[:, 3]),
        "neg_entropy": minmax01(raw_priv_feats[:, 4]),
        "consistency": minmax01(raw_priv_feats[:, 9]),
    }
    if name in mapping: return mapping[name]
    if name.startswith("blend_base_"):
        rest = name[len("blend_base_"):]
        feature_name, alpha_str = rest.rsplit("_", 1)
        alpha = float(alpha_str)
        return minmax01(alpha * base + (1.0 - alpha) * mapping[feature_name])
    return base

# -----------------------------
# Submission
# -----------------------------
def submit(csv_path: Path):
    print(f"Submitting {csv_path} to server...")
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    try:
        with open(csv_path, "rb") as f:
            resp = requests.post(
                f"{BASE_URL}/submit/{TASK_ID}",
                headers={"X-API-Key": API_KEY},
                files={"file": (csv_path.name, f, "application/csv")},
                timeout=(10, 600),
            )
        try:
            body = resp.json()
        except Exception:
            body = {"raw_text": resp.text}

        if resp.status_code == 413:
            print("Upload rejected: file too large (HTTP 413).")
            return

        resp.raise_for_status()

        print("Successfully submitted.")
        print("Server response:", body)
        submission_id = body.get("submission_id")
        if submission_id:
            print(f"Submission ID: {submission_id}")

    except requests.exceptions.RequestException as e:
        detail = getattr(e, "response", None)
        print(f"Submission error: {e}")
        if detail is not None:
            try:
                print("Server response:", detail.json())
            except Exception:
                print("Server response (text):", detail.text)

# -----------------------------
# Main
# -----------------------------
def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Normalization
    MEAN = [0.7406, 0.5331, 0.7059]
    STD = [0.1491, 0.1864, 0.1301]
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    print("Loading datasets...")
    pub_ds = load_and_preprocess_dataset(PUB_PATH, transform=transform)
    priv_ds = load_and_preprocess_dataset(PRIV_PATH, transform=transform)

    pub_loader = DataLoader(pub_ds, batch_size=256, shuffle=False, collate_fn=collate_fn_custom)
    priv_loader = DataLoader(priv_ds, batch_size=256, shuffle=False, collate_fn=collate_fn_custom)

    print("Loading model...")
    model = load_resnet_model(MODEL_PATH, device)

    print("Extracting features from public dataset (TTA=8)...")
    pub_ids, pub_labels, pub_membership, pub_feats = extract_features(model, pub_loader, device, tta=8)
    
    print("Extracting features from private dataset (TTA=8)...")
    priv_ids, priv_labels, _, priv_feats = extract_features(model, priv_loader, device, tta=8)

    # RMIA-like calibration: calibrate by the average probability on non-members
    # pub_feats[:, 1] is true_prob
    print("Applying RMIA-like Calibration...")
    y_pub = pub_membership.astype(np.float32)
    
    def apply_rmia_calibration(feats, labels, ref_feats, ref_labels, ref_membership):
        calibrated = np.zeros(len(feats))
        for c in range(9):
            # Non-members of class c in reference (public) data
            ref_idx = (ref_labels == c) & (ref_membership == 0)
            if ref_idx.sum() > 0:
                # Use correct class prob (index 1 in feats)
                ref_avg_prob = np.mean(ref_feats[ref_idx, 1])
                # Divide target prob by ref avg prob (likelihood ratio)
                calibrated[labels == c] = feats[labels == c, 1] / (ref_avg_prob + 1e-10)
            else:
                calibrated[labels == c] = feats[labels == c, 1]
        return calibrated

    pub_calibrated = apply_rmia_calibration(pub_feats, pub_labels, pub_feats, pub_labels, y_pub)
    priv_calibrated = apply_rmia_calibration(priv_feats, priv_labels, pub_feats, pub_labels, y_pub)
    
    # Add calibrated score as a new feature
    pub_feats = np.column_stack([pub_feats, pub_calibrated])
    priv_feats = np.column_stack([priv_feats, priv_calibrated])

    print("Running cross-validation on public features...")
    pub_oof, priv_base = oof_and_private_scores(pub_feats, y_pub, priv_feats, device, n_splits=5)

    print("Choosing best blend...")
    pub_final, blend_info = choose_blend(y_pub, pub_oof, pub_feats)
    priv_final = apply_same_blend(priv_base, priv_feats, blend_info)

    print(f"Selected scorer: {blend_info['selected']}")
    print(f"Public TPR@5%FPR: {blend_info['public_tpr_at_5fpr']:.6f}")

    # Save submission
    df = pd.DataFrame({
        "id": priv_ids,
        "score": priv_final
    })
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved submission to: {OUTPUT_CSV}")

    # Submit 
    submit(OUTPUT_CSV)

if __name__ == "__main__":
    main()
