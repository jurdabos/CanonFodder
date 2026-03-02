"""
Unified TCN training pipeline for ``c9r train tcn``.

Consolidates the Siamese TCN (Exp 9) and Hybrid TCN + feature floor
(Exp 10) architectures into a single entry point with MLflow tracking.

Architecture reference: Bai et al. (2018) "An Empirical Evaluation of
Generic Convolutional and Recurrent Networks for Sequence Modeling."
"""

from __future__ import annotations
import logging
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
from sklearn.preprocessing import RobustScaler
from helpers.io import AVC_PQ, GS_MB_PQ, PQ_DIR, read_parquet
from helpers.features import compute_pair_features
from helpers.stats import length_stats
from helpers import cluster, experiment

log = logging.getLogger(__name__)

RANDOM_STATE = 47
WRATIO_LOWER = 60
WRATIO_UPPER = 100
GS_DBSCAN_PQ = PQ_DIR / "gs_mb_dbscan.parquet"

# ── Default hyperparameters ───────────────────────────────────────────────────
MAX_SEQ_LEN = 64
EMBED_DIM = 32
TCN_CHANNELS = [64, 64, 64]
KERNEL_SIZE = 3
TCN_DROPOUT = 0.2
FC_DROPOUT = 0.3
N_BASE_FEATURES = 28  # 23 pair features + 5 length stats


# ═════════════════════════════════════════════════════════════════════════════
# Bai et al. TCN building blocks
# ═════════════════════════════════════════════════════════════════════════════
class Chomp1d(nn.Module):
    """Removes trailing padding to maintain causal convolution."""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Residual block with two causal dilated convolutions."""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, use_weight_norm=True):
        super().__init__()
        conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        if use_weight_norm:
            conv1 = weight_norm(conv1)
            conv2 = weight_norm(conv2)
        self.conv1 = conv1
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = conv2
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
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

    def __init__(
        self, num_inputs, num_channels, kernel_size=2, dropout=0.2, use_weight_norm=True, use_layer_norm=False
    ):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2**i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            layers.append(
                TemporalBlock(
                    in_ch,
                    num_channels[i],
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                    use_weight_norm=use_weight_norm,
                )
            )
        self.network = nn.Sequential(*layers)
        self.layer_norm = nn.LayerNorm(num_channels[-1]) if use_layer_norm else None

    def forward(self, x):
        out = self.network(x)
        if self.layer_norm is not None:
            out = self.layer_norm(out.transpose(1, 2)).transpose(1, 2)
        return out


# ═════════════════════════════════════════════════════════════════════════════
# Character vocabulary
# ═════════════════════════════════════════════════════════════════════════════
class CharVocab:
    """Maps characters to integer indices with PAD=0 and UNK=1."""

    PAD = 0
    UNK = 1

    def __init__(self):
        self.char2idx: dict[str, int] = {}
        self._next_idx = 2

    def fit(self, texts: list[str]):
        """Builds vocabulary from a list of strings."""
        for text in texts:
            for ch in text:
                if ch not in self.char2idx:
                    self.char2idx[ch] = self._next_idx
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
# Datasets
# ═════════════════════════════════════════════════════════════════════════════
class NamePairDataset(Dataset):
    """Wraps a DataFrame of (variant_a, variant_b, to_link) for the Siamese TCN."""

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


class HybridDataset(Dataset):
    """Wraps character sequences and precomputed features for the hybrid model."""

    def __init__(self, df: pd.DataFrame, vocab: CharVocab, max_len: int, features: np.ndarray):
        self.a = df["variant_a"].astype(str).tolist()
        self.b = df["variant_b"].astype(str).tolist()
        self.labels = df["to_link"].astype(int).values
        self.features = features
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        a_enc = torch.tensor(self.vocab.encode(self.a[idx], self.max_len), dtype=torch.long)
        b_enc = torch.tensor(self.vocab.encode(self.b[idx], self.max_len), dtype=torch.long)
        feats = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return a_enc, b_enc, feats, label


# ═════════════════════════════════════════════════════════════════════════════
# Model architectures
# ═════════════════════════════════════════════════════════════════════════════
class SiameseTCN(nn.Module):
    """Siamese TCN for pairwise string classification (Exp 9).

    Each name is encoded via character embedding → shared TCN → global pooling.
    Representations are combined and classified through a FC head.
    """

    def __init__(self, vocab_size, embed_dim, tcn_channels, kernel_size, tcn_dropout, fc_dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.tcn = TemporalConvNet(embed_dim, tcn_channels, kernel_size=kernel_size, dropout=tcn_dropout)
        pool_dim = tcn_channels[-1] * 2
        combined_dim = pool_dim * 4  # [h_a; h_b; |h_a - h_b|; h_a ⊙ h_b]
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(128, 1),
        )

    def _encode(self, x):
        """Encodes a batch of character index sequences → fixed-size vectors."""
        emb = self.embedding(x).transpose(1, 2)
        h = self.tcn(emb)
        mask = (x != 0).unsqueeze(1).float()
        h_masked = h * mask
        lengths = mask.sum(dim=2).clamp(min=1)
        h_mean = h_masked.sum(dim=2) / lengths
        h_max = h_masked.masked_fill(mask == 0, -1e9).max(dim=2).values
        return torch.cat([h_mean, h_max], dim=1)

    def forward(self, x_a, x_b):
        """Classifies a pair of character sequences."""
        h_a = self._encode(x_a)
        h_b = self._encode(x_b)
        combined = torch.cat([h_a, h_b, torch.abs(h_a - h_b), h_a * h_b], dim=1)
        return self.head(combined).squeeze(1)


class HybridTCN(nn.Module):
    """Siamese TCN fused with a hand-crafted feature branch (Exp 10).

    The feature branch has a skip connection to the output, ensuring
    features always contribute directly — a stable signal floor.
    """

    def __init__(self, vocab_size, embed_dim, tcn_channels, kernel_size, tcn_dropout, fc_dropout, n_features):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.tcn = TemporalConvNet(
            embed_dim,
            tcn_channels,
            kernel_size=kernel_size,
            dropout=tcn_dropout,
            use_weight_norm=False,
            use_layer_norm=True,
        )
        pool_dim = tcn_channels[-1] * 2
        tcn_combined_dim = pool_dim * 4
        self.feat_branch = nn.Sequential(
            nn.BatchNorm1d(n_features, eps=1e-3),
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.feat_skip = nn.Linear(32, 1)
        self.head = nn.Sequential(
            nn.Linear(tcn_combined_dim + 32, 128),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(128, 1),
        )

    def _encode_seq(self, x):
        """Encodes a character index sequence via TCN → masked global pool."""
        emb = self.embedding(x).transpose(1, 2)
        h = self.tcn(emb)
        mask = (x != 0).unsqueeze(1).float()
        h_masked = h * mask
        lengths = mask.sum(dim=2).clamp(min=1)
        h_mean = h_masked.sum(dim=2) / lengths
        h_max = h_masked.masked_fill(mask == 0, -1e9).max(dim=2).values
        return torch.cat([h_mean, h_max], dim=1)

    def forward(self, x_a, x_b, features):
        """Classifies a name pair using both character sequences and features."""
        h_a = self._encode_seq(x_a)
        h_b = self._encode_seq(x_b)
        h_tcn = torch.cat([h_a, h_b, torch.abs(h_a - h_b), h_a * h_b], dim=1)
        h_feat = self.feat_branch(features)
        main_logit = self.head(torch.cat([h_tcn, h_feat], dim=1)).squeeze(1)
        skip_logit = self.feat_skip(h_feat).squeeze(1)
        return main_logit + skip_logit


# ═════════════════════════════════════════════════════════════════════════════
# Data assembly
# ═════════════════════════════════════════════════════════════════════════════
def _assemble_training_data() -> pd.DataFrame:
    """Builds balanced training set: gs_mb positives + dbscan negatives in [60, 100)."""
    gs = read_parquet(GS_MB_PQ)
    if gs is None or gs.empty:
        raise RuntimeError("gs_mb.parquet not found — run 'c9r canon avc augment' first.")
    positives = gs[gs["to_link"].eq(True)].copy()
    positives["_wr"] = positives.apply(
        lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])),
        axis=1,
    )
    positives = positives[(positives["_wr"] >= WRATIO_LOWER) & (positives["_wr"] < WRATIO_UPPER)]
    positives = positives.drop(columns=["_wr"])
    n_pos = len(positives)
    log.info("Positives in [60,100): %d", n_pos)
    dbscan = read_parquet(GS_DBSCAN_PQ)
    if dbscan is None or dbscan.empty:
        raise RuntimeError("gs_mb_dbscan.parquet not found.")
    neg_pool = dbscan[dbscan["to_link"].eq(False)]
    negatives = neg_pool.sample(n=min(n_pos, len(neg_pool)), random_state=RANDOM_STATE)
    log.info("Negatives sampled: %d (from %d pool)", len(negatives), len(neg_pool))
    train = pd.concat(
        [
            positives[["variant_a", "variant_b", "to_link"]].reset_index(drop=True),
            negatives[["variant_a", "variant_b", "to_link"]].reset_index(drop=True),
        ],
        ignore_index=True,
    )
    log.info("Training set: %d pairs (pos=%d, neg=%d).", len(train), train["to_link"].sum(), (~train["to_link"]).sum())
    return train


def _build_avc_test() -> pd.DataFrame:
    """Builds the AVC test set filtered to WRatio [60, 100)."""
    avc = read_parquet(AVC_PQ)
    if avc is None or avc.empty:
        raise RuntimeError("avc.parquet is empty or missing.")
    decided = avc[avc["to_link"].notna()].copy()
    test_rows: list[tuple] = []
    for _, row in decided.iterrows():
        test_rows.extend(cluster.expand_pairs(row))
    test_raw = pd.DataFrame(test_rows, columns=["variants", "variant_a", "variant_b", "to_link"])
    test_raw["_wr"] = test_raw.apply(lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])), axis=1)
    test_raw = test_raw[(test_raw["_wr"] >= WRATIO_LOWER) & (test_raw["_wr"] < WRATIO_UPPER)]
    return test_raw.drop(columns=["_wr", "variants"]).reset_index(drop=True)


def _precompute_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Computes 28 features (23 base + 5 length stats) for every pair."""
    n = len(df)
    log.info("Precomputing %d features for %d pairs...", N_BASE_FEATURES, n)
    feat_rows: list[dict] = []
    for i, (_, row) in enumerate(df.iterrows()):
        a, b = str(row["variant_a"]), str(row["variant_b"])
        feats = compute_pair_features(a, b)
        ls = length_stats(f"{a}{{{b}")
        feats.update(ls.to_dict())
        feat_rows.append(feats)
        if (i + 1) % 50000 == 0:
            log.info("  Features: %d/%d (%.0f%%)", i + 1, n, 100 * (i + 1) / n)
    feat_df = pd.DataFrame(feat_rows)
    arr = feat_df.values.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, list(feat_df.columns)


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═════════════════════════════════════════════════════════════════════════════
def _optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Finds the threshold that maximises F1."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = np.argmax(f1s)
    return float(thresholds[best_idx]), float(f1s[best_idx])


def _eval_at(y_true, y_prob, threshold):
    """Computes metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
    }


@torch.no_grad()
def _predict_siamese(model: SiameseTCN, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Returns sigmoid probabilities for the Siamese model."""
    model.eval()
    all_probs: list[np.ndarray] = []
    for x_a, x_b, _ in loader:
        logits = model(x_a.to(device), x_b.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs)


@torch.no_grad()
def _predict_hybrid(model: HybridTCN, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Returns sigmoid probabilities for the Hybrid model."""
    model.eval()
    all_probs: list[np.ndarray] = []
    for x_a, x_b, feats, _ in loader:
        logits = model(x_a.to(device), x_b.to(device), feats.to(device))
        logits = torch.clamp(logits, -20.0, 20.0)
        probs = torch.sigmoid(logits).cpu().numpy()
        probs = np.nan_to_num(probs, nan=0.5)
        all_probs.append(probs)
    return np.concatenate(all_probs)


# ═════════════════════════════════════════════════════════════════════════════
# Training loops
# ═════════════════════════════════════════════════════════════════════════════
def _train_siamese(model, train_loader, val_loader, y_val, pos_weight, *, epochs, lr, patience, device):
    """Trains the Siamese TCN with early stopping on validation AUC."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    best_auc = 0.0
    best_state = None
    patience_ctr = 0
    history = {"train_loss": [], "val_auc": []}
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x_a, x_b, labels in train_loader:
            x_a, x_b, labels = x_a.to(device), x_b.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(x_a, x_b)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)
        val_probs = _predict_siamese(model, val_loader, device)
        val_auc = roc_auc_score(y_val, val_probs)
        history["val_auc"].append(val_auc)
        if epoch % 5 == 0 or epoch == 1:
            log.info(
                "Epoch %3d | loss=%.4f | val AUC=%.4f | lr=%.2e", epoch, avg_loss, val_auc, scheduler.get_last_lr()[0]
            )
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                log.info("Early stopping at epoch %d (best val AUC=%.4f).", epoch, best_auc)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    log.info("Training complete. Best val AUC=%.4f.", best_auc)
    return history


def _train_hybrid(model, train_loader, val_loader, y_val, pos_weight, *, epochs, lr, patience, device):
    """Trains the hybrid TCN with early stopping on validation AUC."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    best_auc = 0.0
    best_state = None
    last_good_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience_ctr = 0
    history = {"train_loss": [], "val_auc": []}
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        nan_batches = 0
        for x_a, x_b, feats, labels in train_loader:
            x_a = x_a.to(device)
            x_b = x_b.to(device)
            feats = feats.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(x_a, x_b, feats)
            logits = torch.clamp(logits, -15.0, 15.0)
            loss = criterion(logits, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                model.load_state_dict({k: v.to(device) for k, v in last_good_state.items()})
                continue
            loss.backward()
            has_nan_grad = any(
                p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                for p in model.parameters()
            )
            if has_nan_grad:
                nan_batches += 1
                optimizer.zero_grad()
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if nan_batches > 0:
            log.warning("Epoch %d: %d NaN batches (of %d total).", epoch, nan_batches, n_batches + nan_batches)
        scheduler.step()
        if n_batches == 0:
            log.error("Epoch %d: ALL batches produced NaN. Stopping.", epoch)
            break
        avg_loss = epoch_loss / n_batches
        history["train_loss"].append(avg_loss)
        last_good_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        val_probs = _predict_hybrid(model, val_loader, device)
        val_auc = roc_auc_score(y_val, val_probs)
        history["val_auc"].append(val_auc)
        if epoch % 5 == 0 or epoch == 1:
            log.info(
                "Epoch %3d | loss=%.4f | val AUC=%.4f | lr=%.2e", epoch, avg_loss, val_auc, scheduler.get_last_lr()[0]
            )
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                log.info("Early stopping at epoch %d (best val AUC=%.4f).", epoch, best_auc)
                break
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    log.info("Training complete. Best val AUC=%.4f.", best_auc)
    return history


# ═════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═════════════════════════════════════════════════════════════════════════════
def run_tcn_training(
    *,
    model_type: str = "siamese",
    epochs: int = 80,
    batch_size: int | None = None,
    lr: float | None = None,
    patience: int = 12,
    experiment_num: int | None = None,
    run_name: str | None = None,
) -> dict[str, float]:
    """Runs the TCN training pipeline with MLflow tracking.

    Parameters
    ----------
    model_type : 'siamese' (Exp 9) or 'hybrid' (Exp 10).
    epochs : max training epochs.
    batch_size : mini-batch size (default: 256 for siamese, 512 for hybrid).
    lr : learning rate (default: 1e-3 for siamese, 3e-4 for hybrid).
    patience : early stopping patience.
    experiment_num : explicit experiment number for MLflow labelling.
    run_name : optional MLflow run name.
    """
    # Resolving defaults based on model type
    if batch_size is None:
        batch_size = 256 if model_type == "siamese" else 512
    if lr is None:
        lr = 1e-3 if model_type == "siamese" else 3e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("=== TCN Training: %s (device=%s) ===", model_type, device)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_STATE)
    # Assembling data
    train_full = _assemble_training_data()
    test_df = _build_avc_test()
    log.info(
        "AVC test: %d pairs (pos=%d, neg=%d).", len(test_df), test_df["to_link"].sum(), (~test_df["to_link"]).sum()
    )
    val_size = 0.2 if model_type == "siamese" else 0.15
    train_df, val_df = train_test_split(
        train_full,
        test_size=val_size,
        stratify=train_full["to_link"],
        random_state=RANDOM_STATE,
    )
    log.info("Train/val split: %d / %d", len(train_df), len(val_df))
    # Building character vocabulary from training names
    all_train_names = train_df["variant_a"].astype(str).tolist() + train_df["variant_b"].astype(str).tolist()
    vocab = CharVocab().fit(all_train_names)
    # Computing pos_weight
    n_pos_train = train_df["to_link"].sum()
    n_neg_train = len(train_df) - n_pos_train
    pos_weight = float(n_neg_train / max(n_pos_train, 1))
    log.info("pos_weight: %.2f", pos_weight)
    y_val = val_df["to_link"].astype(int).values
    y_test = test_df["to_link"].astype(int).values
    if model_type == "siamese":
        # Creating datasets and loaders
        train_ds = NamePairDataset(train_df, vocab, MAX_SEQ_LEN)
        val_ds = NamePairDataset(val_df, vocab, MAX_SEQ_LEN)
        test_ds = NamePairDataset(test_df, vocab, MAX_SEQ_LEN)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        # Building model
        model = SiameseTCN(
            vocab_size=vocab.size,
            embed_dim=EMBED_DIM,
            tcn_channels=TCN_CHANNELS,
            kernel_size=KERNEL_SIZE,
            tcn_dropout=TCN_DROPOUT,
            fc_dropout=FC_DROPOUT,
        ).to(device)
        log.info("Model parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")
        # Training
        history = _train_siamese(
            model, train_loader, val_loader, y_val, pos_weight, epochs=epochs, lr=lr, patience=patience, device=device
        )
        # Evaluating
        test_probs = _predict_siamese(model, test_loader, device)
    else:
        # Precomputing features for hybrid model
        log.info("Precomputing features for train/val/test...")
        train_feats, feat_cols = _precompute_features(train_df)
        val_feats, _ = _precompute_features(val_df)
        test_feats, _ = _precompute_features(test_df)
        scaler = RobustScaler()
        train_feats = scaler.fit_transform(train_feats)
        val_feats = scaler.transform(val_feats)
        test_feats = scaler.transform(test_feats)
        # Creating datasets and loaders
        train_ds = HybridDataset(train_df, vocab, MAX_SEQ_LEN, train_feats)
        val_ds = HybridDataset(val_df, vocab, MAX_SEQ_LEN, val_feats)
        test_ds = HybridDataset(test_df, vocab, MAX_SEQ_LEN, test_feats)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        # Building model
        model = HybridTCN(
            vocab_size=vocab.size,
            embed_dim=EMBED_DIM,
            tcn_channels=TCN_CHANNELS,
            kernel_size=KERNEL_SIZE,
            tcn_dropout=TCN_DROPOUT,
            fc_dropout=FC_DROPOUT,
            n_features=N_BASE_FEATURES,
        ).to(device)
        log.info("Model parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")
        # Training
        history = _train_hybrid(
            model, train_loader, val_loader, y_val, pos_weight, epochs=epochs, lr=lr, patience=patience, device=device
        )
        # Evaluating
        test_probs = _predict_hybrid(model, test_loader, device)
    # Evaluating at 3 operating points
    auc = roc_auc_score(y_test, test_probs)
    default_m = _eval_at(y_test, test_probs, 0.5)
    opt_thr, _ = _optimal_threshold(y_test, test_probs)
    optimal_m = _eval_at(y_test, test_probs, opt_thr)
    best_hi = {"threshold": 0.99, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for t in np.arange(0.50, 0.99, 0.01):
        m = _eval_at(y_test, test_probs, t)
        if m["precision"] >= 0.80 and m["f1"] > best_hi["f1"]:
            best_hi = m
    # Logging to MLflow
    experiment.init_experiment()
    parent_name = run_name or f"tcn_{model_type}"
    with experiment.start_run(run_name=parent_name):
        experiment.log_params(
            {
                "experiment": experiment_num or 0,
                "experiment_type": f"tcn_{model_type}",
                "model_architecture": model_type,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "patience": patience,
                "max_seq_len": MAX_SEQ_LEN,
                "embed_dim": EMBED_DIM,
                "tcn_channels": str(TCN_CHANNELS),
                "kernel_size": KERNEL_SIZE,
                "tcn_dropout": TCN_DROPOUT,
                "fc_dropout": FC_DROPOUT,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "device": str(device),
                "random_state": RANDOM_STATE,
                "vocab_size": vocab.size,
                "n_params": sum(p.numel() for p in model.parameters()),
            }
        )
        experiment.log_metrics(
            {
                "auc": auc,
                "default_f1": default_m["f1"],
                "default_precision": default_m["precision"],
                "default_recall": default_m["recall"],
                "opt_threshold": optimal_m["threshold"],
                "opt_f1": optimal_m["f1"],
                "opt_precision": optimal_m["precision"],
                "opt_recall": optimal_m["recall"],
                "hiprec_threshold": best_hi["threshold"],
                "hiprec_f1": best_hi["f1"],
                "hiprec_precision": best_hi["precision"],
                "hiprec_recall": best_hi["recall"],
                "best_val_auc": max(history["val_auc"]) if history["val_auc"] else 0.0,
                "final_train_loss": history["train_loss"][-1] if history["train_loss"] else 0.0,
                "total_epochs": len(history["train_loss"]),
            }
        )
    # Printing results
    results = {
        "auc": auc,
        "default_prec": default_m["precision"],
        "default_rec": default_m["recall"],
        "default_f1": default_m["f1"],
        "opt_thr": optimal_m["threshold"],
        "opt_prec": optimal_m["precision"],
        "opt_rec": optimal_m["recall"],
        "opt_f1": optimal_m["f1"],
        "hiprec_thr": best_hi["threshold"],
        "hiprec_prec": best_hi["precision"],
        "hiprec_rec": best_hi["recall"],
        "hiprec_f1": best_hi["f1"],
    }
    model_label = "SiameseTCN" if model_type == "siamese" else "HybridTCN"
    print(f"\n{'=' * 100}")
    print(f"{model_label} RESULTS")
    print(f"{'=' * 100}")
    print(f"AUC:            {auc:.4f}")
    print(f"Default (0.5):  P={default_m['precision']:.4f}  R={default_m['recall']:.4f}  F1={default_m['f1']:.4f}")
    print(
        f"Optimal:        P={optimal_m['precision']:.4f}  R={optimal_m['recall']:.4f}  F1={optimal_m['f1']:.4f}  (thr={opt_thr:.3f})"
    )
    print(
        f"High-precision: P={best_hi['precision']:.4f}  R={best_hi['recall']:.4f}  F1={best_hi['f1']:.4f}  (thr={best_hi['threshold']:.3f})"
    )
    print(f"{'=' * 100}")
    y_pred_opt = (test_probs >= opt_thr).astype(int)
    print(f"\n=== {model_label} (optimal thr={opt_thr:.3f}) ===")
    print(classification_report(y_test, y_pred_opt, target_names=["no link", "link"]))
    c9r_score = 0.4 * best_hi["precision"] + 0.3 * best_hi["f1"] + 0.3 * auc
    print(f"c9r score: {c9r_score:.4f}")
    print(
        f"\nTraining: {len(history['train_loss'])} epochs, "
        f"final loss={history['train_loss'][-1]:.4f}, "
        f"best val AUC={max(history['val_auc']):.4f}"
    )
    return results
