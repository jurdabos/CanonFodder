"""
Runs Experiment 9: Siamese character-level TCN for artist name variant classification.

Builds on the Bai et al. (2018) TemporalConvNet architecture. Each name in a
pair is encoded via a shared TCN over character embeddings, then the two
representations are combined (concatenation, absolute difference, element-wise
product) and classified through a fully connected head.

Training data: gs_mb.parquet positives + gs_mb_dbscan.parquet negatives,
balanced, filtered to WRatio [60, 100).
Test data: AVC holdout (same 1,172 pairs as Exps 5–8).
"""
from __future__ import annotations
import logging
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import Dataset, DataLoader
from rapidfuzz import fuzz
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from helpers.io import AVC_PQ, GS_MB_PQ, PQ_DIR, read_parquet
from helpers import cluster

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger(__name__)

RANDOM_STATE = 47
WRATIO_LOWER = 60
WRATIO_UPPER = 100
GS_DBSCAN_PQ = PQ_DIR / "gs_mb_dbscan.parquet"

# ── Hyperparameters ───────────────────────────────────────────────────────────
MAX_SEQ_LEN = 64
EMBED_DIM = 32
TCN_CHANNELS = [64, 64, 64]
KERNEL_SIZE = 3
TCN_DROPOUT = 0.2
FC_DROPOUT = 0.3
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 80
PATIENCE = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═════════════════════════════════════════════════════════════════════════════
# Bai et al. TCN building blocks
# ═════════════════════════════════════════════════════════════════════════════
class Chomp1d(nn.Module):
    """Removes trailing padding to maintain causal convolution."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Residual block with two weight-normed causal dilated convolutions."""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation,
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation,
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """Initialises convolutional weights with small random values."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Stacks TemporalBlocks with exponentially increasing dilation."""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1,
                dilation=dilation_size, padding=(kernel_size - 1) * dilation_size,
                dropout=dropout,
            ))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ═════════════════════════════════════════════════════════════════════════════
# Siamese TCN model
# ═════════════════════════════════════════════════════════════════════════════
class SiameseTCN(nn.Module):
    """Siamese TCN for pairwise string classification.

    Each name is encoded via character embedding → shared TCN → global pooling.
    Representations are combined and classified through a FC head.
    """
    def __init__(self, vocab_size, embed_dim, tcn_channels, kernel_size, tcn_dropout, fc_dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.tcn = TemporalConvNet(embed_dim, tcn_channels, kernel_size=kernel_size, dropout=tcn_dropout)
        # Global pooling produces mean + max → 2 × last_channel per name
        pool_dim = tcn_channels[-1] * 2
        # Combination: [h_a; h_b; |h_a - h_b|; h_a ⊙ h_b] → 4 × pool_dim
        combined_dim = pool_dim * 4
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(128, 1),
        )

    def _encode(self, x):
        """Encodes a batch of character index sequences → fixed-size vectors."""
        # x: (batch, seq_len) of int indices
        emb = self.embedding(x)        # (batch, seq_len, embed_dim)
        emb = emb.transpose(1, 2)      # (batch, embed_dim, seq_len) — TCN expects channels-first
        h = self.tcn(emb)              # (batch, channels, seq_len)
        # Masking padded positions before pooling
        mask = (x != 0).unsqueeze(1).float()  # (batch, 1, seq_len)
        h_masked = h * mask
        # Computing mean (over non-padded positions) and max pool
        lengths = mask.sum(dim=2).clamp(min=1)  # (batch, 1)
        h_mean = h_masked.sum(dim=2) / lengths  # (batch, channels)
        h_max = h_masked.masked_fill(mask == 0, -1e9).max(dim=2).values  # (batch, channels)
        return torch.cat([h_mean, h_max], dim=1)  # (batch, 2*channels)

    def forward(self, x_a, x_b):
        """Classifies a pair of character sequences."""
        h_a = self._encode(x_a)
        h_b = self._encode(x_b)
        combined = torch.cat([h_a, h_b, torch.abs(h_a - h_b), h_a * h_b], dim=1)
        return self.head(combined).squeeze(1)


# ═════════════════════════════════════════════════════════════════════════════
# Character vocabulary and encoding
# ═════════════════════════════════════════════════════════════════════════════
class CharVocab:
    """Maps characters to integer indices with PAD=0 and UNK=1."""
    PAD = 0
    UNK = 1

    def __init__(self):
        self.char2idx = {}
        self.idx2char = {0: "<PAD>", 1: "<UNK>"}
        self._next_idx = 2

    def fit(self, texts: list[str]):
        """Builds vocabulary from a list of strings."""
        for text in texts:
            for ch in text:
                if ch not in self.char2idx:
                    self.char2idx[ch] = self._next_idx
                    self.idx2char[self._next_idx] = ch
                    self._next_idx += 1
        log.info("Vocabulary: %d characters (+ PAD, UNK).", len(self.char2idx))
        return self

    def encode(self, text: str, max_len: int) -> list[int]:
        """Encodes a string as padded integer indices."""
        ids = [self.char2idx.get(ch, self.UNK) for ch in text[:max_len]]
        ids += [self.PAD] * (max_len - len(ids))
        return ids

    @property
    def size(self) -> int:
        """Returns the total vocabulary size including special tokens."""
        return self._next_idx


# ═════════════════════════════════════════════════════════════════════════════
# Dataset
# ═════════════════════════════════════════════════════════════════════════════
class NamePairDataset(Dataset):
    """Wraps a DataFrame of (variant_a, variant_b, to_link) into a PyTorch Dataset."""
    def __init__(self, df: pd.DataFrame, vocab: CharVocab, max_len: int):
        self.a = df["variant_a"].astype(str).tolist()
        self.b = df["variant_b"].astype(str).tolist()
        self.labels = df["to_link"].astype(int).values
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        a_enc = torch.tensor(self.vocab.encode(self.a[idx], self.max_len), dtype=torch.long)
        b_enc = torch.tensor(self.vocab.encode(self.b[idx], self.max_len), dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return a_enc, b_enc, label


# ═════════════════════════════════════════════════════════════════════════════
# Data assembly
# ═════════════════════════════════════════════════════════════════════════════
def assemble_training_data() -> pd.DataFrame:
    """Builds balanced training set: gs_mb positives + dbscan negatives in [60, 100)."""
    gs = read_parquet(GS_MB_PQ)
    positives = gs[gs["to_link"] == True].copy()
    # Filtering positives to WRatio [60, 100)
    positives["_wr"] = positives.apply(
        lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])), axis=1,
    )
    positives = positives[(positives["_wr"] >= WRATIO_LOWER) & (positives["_wr"] < WRATIO_UPPER)]
    positives = positives.drop(columns=["_wr"])
    n_pos = len(positives)
    log.info("Positives in [60,100): %d", n_pos)
    # Sampling matching number of negatives from dbscan
    dbscan = read_parquet(GS_DBSCAN_PQ)
    neg_pool = dbscan[dbscan["to_link"] == False]
    negatives = neg_pool.sample(n=min(n_pos, len(neg_pool)), random_state=RANDOM_STATE)
    log.info("Negatives sampled: %d (from %d pool)", len(negatives), len(neg_pool))
    train = pd.concat([
        positives[["variant_a", "variant_b", "to_link"]].reset_index(drop=True),
        negatives[["variant_a", "variant_b", "to_link"]].reset_index(drop=True),
    ], ignore_index=True)
    log.info("Training set: %d pairs (pos=%d, neg=%d).",
             len(train), train["to_link"].sum(), (~train["to_link"]).sum())
    return train


def build_avc_test() -> pd.DataFrame:
    """Builds the AVC test set filtered to WRatio [60, 100)."""
    avc = read_parquet(AVC_PQ)
    decided = avc[avc["to_link"].notna()].copy()
    test_rows = []
    for _, row in decided.iterrows():
        test_rows.extend(cluster.expand_pairs(row))
    test_raw = pd.DataFrame(test_rows, columns=["variants", "variant_a", "variant_b", "to_link"])
    test_raw["_wr"] = test_raw.apply(lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])), axis=1)
    test_raw = test_raw[(test_raw["_wr"] >= WRATIO_LOWER) & (test_raw["_wr"] < WRATIO_UPPER)]
    return test_raw.drop(columns=["_wr", "variants"]).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation helpers (same interface as Exp 6–8)
# ═════════════════════════════════════════════════════════════════════════════
def _optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Finds the threshold that maximises F1."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = np.argmax(f1s)
    return float(thresholds[best_idx]), float(f1s[best_idx])


def _evaluate_at_threshold(y_true, y_prob, threshold):
    """Computes metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
    }


@torch.no_grad()
def predict_proba(model: SiameseTCN, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Returns sigmoid probabilities for all samples in a DataLoader."""
    model.eval()
    all_probs = []
    for x_a, x_b, _ in loader:
        logits = model(x_a.to(device), x_b.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs)


# ═════════════════════════════════════════════════════════════════════════════
# Training loop
# ═════════════════════════════════════════════════════════════════════════════
def train_model(
    model: SiameseTCN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_val: np.ndarray,
    pos_weight: float,
) -> dict:
    """Trains the Siamese TCN with early stopping on validation AUC."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))
    best_auc = 0.0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_auc": []}
    for epoch in range(1, EPOCHS + 1):
        # Training phase
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x_a, x_b, labels in train_loader:
            x_a, x_b, labels = x_a.to(DEVICE), x_b.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x_a, x_b)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_loss = epoch_loss / n_batches
        history["train_loss"].append(avg_loss)
        # Validation phase
        val_probs = predict_proba(model, val_loader, DEVICE)
        val_auc = roc_auc_score(y_val, val_probs)
        history["val_auc"].append(val_auc)
        if epoch % 5 == 0 or epoch == 1:
            log.info("Epoch %3d | loss=%.4f | val AUC=%.4f | lr=%.2e",
                     epoch, avg_loss, val_auc, scheduler.get_last_lr()[0])
        # Early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info("Early stopping at epoch %d (best val AUC=%.4f at epoch %d).",
                         epoch, best_auc, epoch - PATIENCE)
                break
    # Restoring best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)
    log.info("Training complete. Best val AUC=%.4f.", best_auc)
    return history


# ═════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═════════════════════════════════════════════════════════════════════════════
def main():
    """Runs the full Experiment 9 pipeline."""
    log.info("=== Experiment 9: Siamese TCN (Bai et al.) ===")
    log.info("Device: %s", DEVICE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_STATE)
    # Step 1: Assembling data
    train_full = assemble_training_data()
    test_df = build_avc_test()
    log.info("AVC test: %d pairs (pos=%d, neg=%d).",
             len(test_df), test_df["to_link"].sum(), (~test_df["to_link"]).sum())
    # Splitting training into train/val (80/20)
    train_df, val_df = train_test_split(
        train_full, test_size=0.2, stratify=train_full["to_link"], random_state=RANDOM_STATE,
    )
    log.info("Train/val split: %d / %d", len(train_df), len(val_df))
    # Step 2: Building character vocabulary from training names
    all_train_names = train_df["variant_a"].astype(str).tolist() + train_df["variant_b"].astype(str).tolist()
    vocab = CharVocab().fit(all_train_names)
    # Step 3: Creating datasets and loaders
    train_ds = NamePairDataset(train_df, vocab, MAX_SEQ_LEN)
    val_ds = NamePairDataset(val_df, vocab, MAX_SEQ_LEN)
    test_ds = NamePairDataset(test_df, vocab, MAX_SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    y_val = val_df["to_link"].astype(int).values
    y_test = test_df["to_link"].astype(int).values
    # Step 4: Building model
    model = SiameseTCN(
        vocab_size=vocab.size,
        embed_dim=EMBED_DIM,
        tcn_channels=TCN_CHANNELS,
        kernel_size=KERNEL_SIZE,
        tcn_dropout=TCN_DROPOUT,
        fc_dropout=FC_DROPOUT,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %s", f"{n_params:,}")
    log.info("TCN receptive field: %d characters",
             1 + 2 * (KERNEL_SIZE - 1) * sum(2 ** i for i in range(len(TCN_CHANNELS))))
    # Computing pos_weight for BCE loss
    n_pos_train = train_df["to_link"].sum()
    n_neg_train = len(train_df) - n_pos_train
    pos_weight = float(n_neg_train / max(n_pos_train, 1))
    log.info("pos_weight: %.2f", pos_weight)
    # Step 5: Training
    log.info("Training Siamese TCN...")
    history = train_model(model, train_loader, val_loader, y_val, pos_weight)
    # Step 6: Evaluating on AVC test set
    log.info("Evaluating on AVC test set (%d pairs)...", len(test_df))
    test_probs = predict_proba(model, test_loader, DEVICE)
    auc = roc_auc_score(y_test, test_probs)
    default_m = _evaluate_at_threshold(y_test, test_probs, 0.5)
    opt_thr, _ = _optimal_threshold(y_test, test_probs)
    optimal_m = _evaluate_at_threshold(y_test, test_probs, opt_thr)
    # Finding high-precision threshold (P ≥ 0.80)
    best_hi = {"threshold": 0.99, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for t in np.arange(0.50, 0.99, 0.01):
        m = _evaluate_at_threshold(y_test, test_probs, t)
        if m["precision"] >= 0.80 and m["f1"] > best_hi["f1"]:
            best_hi = m
    # Printing results
    print("\n" + "=" * 100)
    print("EXPERIMENT 9: SIAMESE TCN RESULTS")
    print("=" * 100)
    print(f"AUC:            {auc:.4f}")
    print(f"Default (0.5):  P={default_m['precision']:.4f}  R={default_m['recall']:.4f}  F1={default_m['f1']:.4f}")
    print(f"Optimal:        P={optimal_m['precision']:.4f}  R={optimal_m['recall']:.4f}  F1={optimal_m['f1']:.4f}  (thr={opt_thr:.3f})")
    print(f"High-precision: P={best_hi['precision']:.4f}  R={best_hi['recall']:.4f}  F1={best_hi['f1']:.4f}  (thr={best_hi['threshold']:.3f})")
    print("=" * 100)
    # Printing classification report at optimal threshold
    y_pred_opt = (test_probs >= opt_thr).astype(int)
    print(f"\n=== SiameseTCN (optimal thr={opt_thr:.3f}) ===")
    print(classification_report(y_test, y_pred_opt, target_names=["no link", "link"]))
    # Comparison with Exp 6 baseline
    print("── Exp 6 baseline (ExtraTrees): AUC=0.8920, opt F1=0.7050 (thr=0.940) ──")
    # Training history summary
    print(f"\nTraining: {len(history['train_loss'])} epochs, "
          f"final loss={history['train_loss'][-1]:.4f}, "
          f"best val AUC={max(history['val_auc']):.4f}")


if __name__ == "__main__":
    main()
