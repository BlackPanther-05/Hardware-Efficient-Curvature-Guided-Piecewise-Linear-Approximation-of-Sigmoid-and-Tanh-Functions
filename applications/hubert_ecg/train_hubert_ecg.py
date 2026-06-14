"""
Fine-tuning HuBERT-ECG-base on PTB-XL
=======================================
Model  : Edoardo-BS/hubert-ecg-base  (HuggingFace)
Dataset: PTB-XL v1.0.3               (PhysioNet)
Task   : 5-class multi-label ECG classification
         NORM | MI | STTC | CD | HYP
Output : hubert_ecg_ptbxl.pth

What's printed every epoch
---------------------------
  Train Loss | Val Loss
  Per-class AUROC  (NORM, MI, STTC, CD, HYP)  + Macro AUROC
  Per-class Accuracy %                          + Macro Accuracy
  Per-class F1                                  + Macro F1

Full per-class table is printed every 5 epochs and at the end.

Usage
-----
  pip install torch transformers wfdb pandas numpy scikit-learn tqdm
  python train_hubert_ecg_ptbxl.py
"""

import os
import ast
import math
import time
import warnings
import urllib.request
import urllib.error

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm
import wfdb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION  –  edit these if needed
# ─────────────────────────────────────────────────────────────

PTBXL_PATH       = "./ptb-xl-1.0.3"
OUTPUT_PATH      = "./hubert_ecg_ptbxl.pth"

SAMPLING_RATE    = 100          # 100 Hz  (matches model pre-training at 5 s clips)
SAMPLES_PER_LEAD = 500          # 5 s × 100 Hz = 500 samples per lead
NUM_LEADS        = 12
# Backbone input = 12 leads concatenated: 12 × 500 = 6000
BACKBONE_INPUT_LEN = NUM_LEADS * SAMPLES_PER_LEAD

NUM_CLASSES   = 5
SUPERCLASSES  = ["NORM", "MI", "STTC", "CD", "HYP"]

BATCH_SIZE    = 16
NUM_EPOCHS    = 40              # ← increased from 20 to 40
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
DROPOUT       = 0.1
THRESHOLD     = 0.5             # decision threshold for Accuracy and F1

FREEZE_BASE   = False           # False = fine-tune entire backbone
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS   = 4 if DEVICE == "cuda" else 0

print(f"Using device        : {DEVICE}")
print(f"Epochs              : {NUM_EPOCHS}")
print(f"Backbone input len  : {BACKBONE_INPUT_LEN}  "
      f"({NUM_LEADS} leads × {SAMPLES_PER_LEAD} samples @ {SAMPLING_RATE} Hz)")


# ─────────────────────────────────────────────────────────────
#  STEP 1  –  Download / repair dataset
# ─────────────────────────────────────────────────────────────

_BASE_URL   = "https://physionet.org/files/ptb-xl/1.0.3"
_ROOT_FILES = ["ptbxl_database.csv", "scp_statements.csv", "RECORDS", "LICENSE.txt"]


def _http_get(url: str, dest: str, max_retries: int = 5) -> bool:
    """Download url → dest with resume + retries. Returns True on success."""
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    existing = os.path.getsize(dest) if os.path.exists(dest) else 0
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url)
            if existing:
                req.add_header("Range", f"bytes={existing}-")
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.status not in (200, 206):
                    return False
                mode = "ab" if (existing and resp.status == 206) else "wb"
                with open(dest, mode) as fh:
                    while True:
                        chunk = resp.read(1 << 17)
                        if not chunk:
                            break
                        fh.write(chunk)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 416:
                return True     # already complete
            if e.code == 404:
                return False    # permanent failure
            if attempt == max_retries:
                return False
            time.sleep(2 ** attempt)
        except Exception:
            if attempt == max_retries:
                return False
            time.sleep(2 ** attempt)
    return False


def _get_stems(target_dir: str) -> list:
    prefix = "records500/" if SAMPLING_RATE == 500 else "records100/"
    with open(os.path.join(target_dir, "RECORDS")) as fh:
        return [l.strip() for l in fh if l.strip().startswith(prefix)]


def download_ptbxl(target_dir: str) -> None:
    """Download PTB-XL or fill in any missing files."""
    os.makedirs(target_dir, exist_ok=True)

    # Root files
    if any(not os.path.isfile(os.path.join(target_dir, f)) for f in _ROOT_FILES):
        print("[DATA] Downloading root files ...")
        for fname in _ROOT_FILES:
            dest = os.path.join(target_dir, fname)
            if os.path.isfile(dest) and os.path.getsize(dest) > 0:
                continue
            print(f"       {fname} ... ", end="", flush=True)
            print("OK" if _http_get(f"{_BASE_URL}/{fname}", dest) else "FAILED")

    rec_path = os.path.join(target_dir, "RECORDS")
    if not os.path.isfile(rec_path):
        raise RuntimeError(
            f"[DATA] RECORDS not found.\n"
            f"  wget -r -N -c -np {_BASE_URL}/ -P {target_dir}"
        )

    stems = _get_stems(target_dir)
    missing = [
        (stem + ext, f"{_BASE_URL}/{stem}{ext}")
        for stem in stems for ext in (".hea", ".dat")
        if not os.path.isfile(os.path.join(target_dir, stem + ext))
        or os.path.getsize(os.path.join(target_dir, stem + ext)) == 0
    ]
    if not missing:
        print(f"[DATA] All {len(stems) * 2} waveform files present. OK")
        return
    print(f"[DATA] {len(missing)} files missing – downloading ...")
    failed = []
    for rel, url in tqdm(missing, desc="[DATA]", unit="file"):
        if not _http_get(url, os.path.join(target_dir, rel)):
            failed.append(url)
    if failed:
        print(f"[DATA] {len(failed)} files still failed – re-run to retry.")
    else:
        print(f"[DATA] Download complete. ({len(missing)} files fetched)")


# ─────────────────────────────────────────────────────────────
#  STEP 2  –  Metadata & labels
# ─────────────────────────────────────────────────────────────

def load_metadata(ptbxl_dir: str) -> pd.DataFrame:
    df  = pd.read_csv(os.path.join(ptbxl_dir, "ptbxl_database.csv"), index_col="ecg_id")
    scp = pd.read_csv(os.path.join(ptbxl_dir, "scp_statements.csv"), index_col=0)
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    diag_map = scp[scp["diagnostic"] == 1]["diagnostic_class"].dropna().to_dict()

    def make_label(d):
        lbl = np.zeros(NUM_CLASSES, dtype=np.float32)
        for code, lik in d.items():
            if code in diag_map and diag_map[code] in SUPERCLASSES and lik > 0:
                lbl[SUPERCLASSES.index(diag_map[code])] = 1.0
        return lbl

    df["label"] = df["scp_codes"].apply(make_label)
    return df[df["label"].apply(lambda x: x.sum() > 0)].copy()


# ─────────────────────────────────────────────────────────────
#  STEP 3  –  Pre-flight file verification
# ─────────────────────────────────────────────────────────────

def verify_and_repair(df: pd.DataFrame, ptbxl_dir: str, tag: str = "") -> pd.DataFrame:
    col = "filename_hr" if SAMPLING_RATE == 500 else "filename_lr"
    needed = [
        (row[col] + ext, f"{_BASE_URL}/{row[col]}{ext}")
        for _, row in df.iterrows()
        for ext in (".hea", ".dat")
        if not os.path.isfile(os.path.join(ptbxl_dir, row[col] + ext))
        or os.path.getsize(os.path.join(ptbxl_dir, row[col] + ext)) == 0
    ]
    if not needed:
        print(f"[VERIFY] {tag} All {len(df)} records OK")
        return df
    print(f"[VERIFY] {tag} {len(needed)} files missing – fetching ...")
    failed_stems = set()
    for rel, url in tqdm(needed, desc=f"[VERIFY]{tag}", unit="file"):
        if not _http_get(url, os.path.join(ptbxl_dir, rel)):
            failed_stems.add(os.path.splitext(rel)[0])
    if failed_stems:
        before = len(df)
        df = df[~df[col].isin(failed_stems)].copy()
        print(f"[VERIFY] Dropped {before - len(df)} rows; {len(df)} remain.")
    else:
        print(f"[VERIFY] {tag} All files recovered OK")
    return df


# ─────────────────────────────────────────────────────────────
#  STEP 4  –  Preprocessing
#
#  HuBERT-ECG (HubertModel subclass) expects input_values: [B, T]
#  where T = 12 leads concatenated = 12 × SAMPLES_PER_LEAD = 6000
# ─────────────────────────────────────────────────────────────

def preprocess_ecg(signal: np.ndarray) -> np.ndarray:
    """
    signal : [T, 12] or [12, T]  raw wfdb output
    Returns: [12 × SAMPLES_PER_LEAD]  float32  (1D, leads concatenated)
    """
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
    if signal.shape[0] != NUM_LEADS:   # wfdb returns [T, 12]
        signal = signal.T              # → [12, T]

    # Per-lead z-score normalisation
    mean   = signal.mean(axis=1, keepdims=True)
    std    = signal.std(axis=1,  keepdims=True) + 1e-8
    signal = np.clip((signal - mean) / std, -6.0, 6.0)

    # Pad / truncate each lead to SAMPLES_PER_LEAD
    T = signal.shape[1]
    if T >= SAMPLES_PER_LEAD:
        signal = signal[:, :SAMPLES_PER_LEAD]
    else:
        pad    = np.zeros((NUM_LEADS, SAMPLES_PER_LEAD - T), dtype=np.float32)
        signal = np.concatenate([signal, pad], axis=1)

    # Concatenate all 12 leads into one 1D vector
    return signal.reshape(-1).astype(np.float32)   # [6000]


# ─────────────────────────────────────────────────────────────
#  STEP 5  –  Dataset
# ─────────────────────────────────────────────────────────────

class PTBXLDataset(Dataset):
    def __init__(self, df: pd.DataFrame, ptbxl_dir: str):
        col = "filename_hr" if SAMPLING_RATE == 500 else "filename_lr"
        self.records   = df.reset_index()[[col, "label"]].values.tolist()
        self.ptbxl_dir = ptbxl_dir

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rel_path, label = self.records[idx]
        record  = wfdb.rdrecord(os.path.join(self.ptbxl_dir, rel_path))
        signal  = preprocess_ecg(record.p_signal.astype(np.float32))
        return torch.tensor(signal), torch.tensor(label, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────
#  STEP 6  –  Model
# ─────────────────────────────────────────────────────────────

class HuBERTECGClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=DROPOUT, freeze_base=FREEZE_BASE):
        super().__init__()
        print("[MODEL] Loading Edoardo-BS/hubert-ecg-base ...")
        self.backbone = AutoModel.from_pretrained(
            "Edoardo-BS/hubert-ecg-base", trust_remote_code=True
        )
        if freeze_base:
            for p in self.backbone.parameters():
                p.requires_grad = False

        h = self.backbone.config.hidden_size
        print(f"[MODEL] Backbone hidden size: {h}")

        self.head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Dropout(dropout),
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h // 2, num_classes),
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, 12 × SAMPLES_PER_LEAD]  (1D, leads concatenated)"""
        hidden = self.backbone(input_values=x).last_hidden_state  # [B, T', H]
        return self.head(hidden.mean(dim=1))                       # [B, C]


# ─────────────────────────────────────────────────────────────
#  STEP 7  –  Metrics
# ─────────────────────────────────────────────────────────────

def compute_all_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    """
    labels : [N, 5]  ground truth (0/1)
    probs  : [N, 5]  sigmoid probabilities

    Returns a dict with per-class and macro metrics:
      auroc_per, macro_auroc
      acc_per,   macro_acc
      f1_per,    macro_f1
    """
    preds = (probs >= THRESHOLD).astype(int)

    auroc_per, acc_per, f1_per = [], [], []
    for c in range(NUM_CLASSES):
        # AUROC
        if len(np.unique(labels[:, c])) > 1:
            auroc_per.append(roc_auc_score(labels[:, c], probs[:, c]))
        else:
            auroc_per.append(float("nan"))
        # Accuracy
        acc_per.append(accuracy_score(labels[:, c], preds[:, c]) * 100.0)
        # F1
        f1_per.append(f1_score(labels[:, c], preds[:, c], zero_division=0))

    return {
        "auroc_per":   auroc_per,
        "macro_auroc": float(np.nanmean(auroc_per)),
        "acc_per":     acc_per,
        "macro_acc":   float(np.mean(acc_per)),
        "f1_per":      f1_per,
        "macro_f1":    float(np.mean(f1_per)),
    }


def print_metrics_table(metrics: dict, split: str, loss: float = None) -> None:
    """Print a nicely formatted per-class metrics table."""
    bar = "─" * 60
    hdr = f"  {split}"
    if loss is not None:
        hdr += f"   |   Loss: {loss:.4f}"
    print(f"\n  {bar}")
    print(hdr)
    print(f"  {bar}")
    print(f"  {'Class':<8}  {'AUROC':>8}  {'Accuracy':>9}  {'F1':>8}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*8}")
    for i, cls in enumerate(SUPERCLASSES):
        a = metrics["auroc_per"][i]
        a_str = f"{a:.4f}" if not math.isnan(a) else "   N/A"
        print(f"  {cls:<8}  {a_str:>8}  "
              f"{metrics['acc_per'][i]:>8.2f}%  "
              f"{metrics['f1_per'][i]:>8.4f}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*8}")
    print(f"  {'MACRO':<8}  {metrics['macro_auroc']:>8.4f}  "
          f"{metrics['macro_acc']:>8.2f}%  "
          f"{metrics['macro_f1']:>8.4f}")
    print(f"  {bar}\n")


# ─────────────────────────────────────────────────────────────
#  STEP 8  –  Train / Eval loops
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total = 0.0
    for signals, labels in tqdm(loader, desc="  Train", leave=False):
        signals, labels = signals.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            enabled=(device.type == "cuda"),
        ):
            loss = criterion(model(signals), labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, probs_l, labels_l = 0.0, [], []
    for signals, labels in tqdm(loader, desc="  Eval ", leave=False):
        signals, labels = signals.to(device), labels.to(device)
        logits = model(signals)
        total += criterion(logits, labels).item()
        probs_l.append(torch.sigmoid(logits).cpu().numpy())
        labels_l.append(labels.cpu().numpy())
    all_p    = np.concatenate(probs_l)
    all_l    = np.concatenate(labels_l)
    metrics  = compute_all_metrics(all_l, all_p)
    avg_loss = total / len(loader)
    return avg_loss, metrics


# ─────────────────────────────────────────────────────────────
#  STEP 9  –  Main
# ─────────────────────────────────────────────────────────────

def main():
    # 9.1  Download + verify ──────────────────────────────────
    download_ptbxl(PTBXL_PATH)

    print("[DATA] Loading metadata ...")
    df = load_metadata(PTBXL_PATH)
    print(f"[DATA] Total labelled records: {len(df)}")

    train_df = df[df["strat_fold"] <= 8].copy()
    val_df   = df[df["strat_fold"] == 9].copy()
    test_df  = df[df["strat_fold"] == 10].copy()

    print("[VERIFY] Checking files before training ...")
    train_df = verify_and_repair(train_df, PTBXL_PATH, "[TRAIN]")
    val_df   = verify_and_repair(val_df,   PTBXL_PATH, "[VAL]  ")
    test_df  = verify_and_repair(test_df,  PTBXL_PATH, "[TEST] ")
    print(f"[DATA] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # 9.2  DataLoaders ────────────────────────────────────────
    pin = (DEVICE == "cuda")
    train_loader = DataLoader(PTBXLDataset(train_df, PTBXL_PATH),
                              BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(PTBXLDataset(val_df,   PTBXL_PATH),
                              BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader  = DataLoader(PTBXLDataset(test_df,  PTBXL_PATH),
                              BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    # 9.3  Model ──────────────────────────────────────────────
    device = torch.device(DEVICE)
    model  = HuBERTECGClassifier().to(device)
    total_p   = sum(p.numel() for p in model.parameters())
    train_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Total params: {total_p:,}  |  Trainable: {train_p:,}")

    # 9.4  Loss / optimiser / scheduler ───────────────────────
    lm        = np.stack(train_df["label"].values)
    pos_w     = torch.tensor(
        (lm.shape[0] - lm.sum(0)) / (lm.sum(0) + 1e-8), dtype=torch.float32
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    # Cosine annealing: smoothly decreases LR from LR to LR*0.01 over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=LR * 0.01
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    # 9.5  Training loop ──────────────────────────────────────
    best_auroc, best_state, history = 0.0, None, []
    print(f"\n[TRAIN] Starting {NUM_EPOCHS}-epoch training ...\n")
    print(f"  {'Ep':>3}  {'Tr Loss':>8}  {'Vl Loss':>8}  "
          f"{'AUROC':>7}  {'Acc%':>7}  {'F1':>7}  {'LR':>9}  {'Time':>6}")
    print(f"  {'─'*3}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*9}  {'─'*6}")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        tr_loss             = train_one_epoch(model, train_loader, optimizer,
                                              criterion, scaler, device)
        vl_loss, vl_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        cur_lr     = optimizer.param_groups[0]["lr"]
        vl_auroc   = vl_metrics["macro_auroc"]
        vl_acc     = vl_metrics["macro_acc"]
        vl_f1      = vl_metrics["macro_f1"]

        # Compact one-line summary every epoch
        print(f"  {epoch:>3}  {tr_loss:>8.4f}  {vl_loss:>8.4f}  "
              f"{vl_auroc:>7.4f}  {vl_acc:>6.2f}%  {vl_f1:>7.4f}  "
              f"{cur_lr:>9.2e}  {elapsed:>5.0f}s")

        # Full per-class table every 5 epochs
        if epoch % 5 == 0:
            print_metrics_table(vl_metrics, split=f"Validation – Epoch {epoch}", loss=vl_loss)

        history.append({
            "epoch":      epoch,
            "lr":         cur_lr,
            "train_loss": round(tr_loss,  5),
            "val_loss":   round(vl_loss,  5),
            "val_auroc":  round(vl_auroc, 5),
            "val_acc":    round(vl_acc,   3),
            "val_f1":     round(vl_f1,    5),
        })

        # Save best checkpoint (tracked by Val AUROC)
        if vl_auroc > best_auroc:
            best_auroc = vl_auroc
            best_state = {
                "epoch":           epoch,
                "model_state":     {k: v.cpu() for k, v in model.state_dict().items()},
                "optimizer_state": optimizer.state_dict(),
                "val_auroc":       vl_auroc,
                "val_loss":        vl_loss,
                "val_metrics":     vl_metrics,
                "config": {
                    "num_classes":        NUM_CLASSES,
                    "superclasses":       SUPERCLASSES,
                    "sampling_rate":      SAMPLING_RATE,
                    "samples_per_lead":   SAMPLES_PER_LEAD,
                    "backbone_input_len": BACKBONE_INPUT_LEN,
                    "backbone":           "Edoardo-BS/hubert-ecg-base",
                    "dropout":            DROPOUT,
                    "threshold":          THRESHOLD,
                },
                "history": history,
            }
            print(f"  *** Best checkpoint: epoch {epoch}  "
                  f"AUROC {vl_auroc:.4f}  Acc {vl_acc:.2f}%  F1 {vl_f1:.4f} ***")

    # 9.6  Save weights ───────────────────────────────────────
    torch.save(best_state, OUTPUT_PATH)
    print(f"\n[SAVE] Weights saved → {OUTPUT_PATH}")
    print(f"       Best epoch : {best_state['epoch']}")
    print_metrics_table(best_state["val_metrics"],
                        split="Best Validation Checkpoint", loss=best_state["val_loss"])

    # 9.7  Final test evaluation ──────────────────────────────
    print("[TEST] Loading best weights for test evaluation ...")
    model.load_state_dict({k: v.to(device) for k, v in best_state["model_state"].items()})
    te_loss, te_metrics = evaluate(model, test_loader, criterion, device)
    print_metrics_table(te_metrics, split="TEST SET – Final Results", loss=te_loss)


# ─────────────────────────────────────────────────────────────
#  STEP 10  –  Inference helper
# ─────────────────────────────────────────────────────────────

def load_and_infer(weights_path: str, ecg_signal: np.ndarray) -> dict:
    """
    Run inference on a single ECG.

    Parameters
    ----------
    weights_path : str
        Path to the saved .pth file.
    ecg_signal : np.ndarray
        Shape [12, T] or [T, 12] at SAMPLING_RATE Hz.
        For 100 Hz: T should be at least 500 (5 seconds).

    Returns
    -------
    dict  e.g.:
        {
          "NORM": {"probability": 0.91, "prediction": "POSITIVE"},
          "MI":   {"probability": 0.04, "prediction": "negative"},
          ...
        }
    """
    ckpt   = torch.load(weights_path, map_location="cpu")
    cfg    = ckpt["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HuBERTECGClassifier(num_classes=cfg["num_classes"], dropout=0.0).to(device)
    model.load_state_dict({k: v.to(device) for k, v in ckpt["model_state"].items()})
    model.eval()

    signal = preprocess_ecg(ecg_signal)
    x      = torch.tensor(signal).unsqueeze(0).to(device)  # [1, 6000]

    with torch.no_grad():
        probs = torch.sigmoid(model(x)).squeeze(0).cpu().numpy()

    thr = cfg.get("threshold", THRESHOLD)
    return {
        cls: {
            "probability": round(float(p), 4),
            "prediction":  "POSITIVE" if p >= thr else "negative",
        }
        for cls, p in zip(cfg["superclasses"], probs)
    }


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
