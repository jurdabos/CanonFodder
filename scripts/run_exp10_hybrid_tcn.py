"""
Runs Experiment 10: Hybrid Siamese TCN with feature-engineering signal floor.

Combines the Bai et al. TCN character encoder with a hand-crafted feature
branch that provides a stable signal the TCN cannot overfit away from.

Steps:
1. Extract ALL alias pairs from the local MBDB mirror → gs_mb_max.parquet
2. Compute WRatio for positives, distribution-match negatives from DBSCAN
3. Precompute 28 hand-crafted features for every pair
4. Train a hybrid model: Siamese TCN + feature branch (with skip connection)
5. Evaluate on AVC holdout with threshold sweep
"""

from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rapidfuzz import fuzz
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from corefunc.mb_local import _psql_csv, check_local_mb  # noqa: E402
from helpers import cluster  # noqa: E402
from helpers.features import compute_pair_features  # noqa: E402
from helpers.io import AVC_PQ, PQ_DIR, dump_parquet, read_parquet  # noqa: E402
from helpers.stats import length_stats  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger(__name__)

RANDOM_STATE = 47
WRATIO_LOWER = 60
WRATIO_UPPER = 100
GS_DBSCAN_PQ = PQ_DIR / "gs_mb_dbscan.parquet"
GS_MB_MAX_PQ = PQ_DIR / "gs_mb_max.parquet"
MAX_NAMES_PER_ARTIST = 30

# ── Hyperparameters ───────────────────────────────────────────────────────────
MAX_SEQ_LEN = 64
EMBED_DIM = 32
TCN_CHANNELS = [64, 64, 64]
KERNEL_SIZE = 3
TCN_DROPOUT = 0.2
FC_DROPOUT = 0.3
BATCH_SIZE = 512
LR = 3e-4
EPOCHS = 80
PATIENCE = 12
N_BASE_FEATURES = 28  # 23 pair features + 5 length stats
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═════════════════════════════════════════════════════════════════════════════
# Step 1: Extract all MBDB alias pairs → gs_mb_max.parquet
# ═════════════════════════════════════════════════════════════════════════════
def step1_extract_mbdb_max() -> pd.DataFrame:
    """Extracts all artist alias pairs from MBDB and saves gs_mb_max.parquet.

    For each artist with aliases, collects {primary name} ∪ {aliases},
    caps at MAX_NAMES_PER_ARTIST, and generates all C(n,2) positive pairs.
    """
    if GS_MB_MAX_PQ.exists():
        existing = read_parquet(GS_MB_MAX_PQ)
        log.info("gs_mb_max.parquet already exists: %d pairs. Skipping extraction.", len(existing))
        return existing
    if not check_local_mb():
        raise RuntimeError("Local MBDB mirror not reachable.")
    log.info("Extracting all artist names + aliases from MBDB...")
    sql = """\
SELECT a.gid::text AS mbid, a.name AS name
FROM musicbrainz.artist a
WHERE EXISTS (SELECT 1 FROM musicbrainz.artist_alias aa WHERE aa.artist = a.id)
UNION ALL
SELECT a.gid::text AS mbid, aa.name AS name
FROM musicbrainz.artist a
JOIN musicbrainz.artist_alias aa ON aa.artist = a.id
ORDER BY mbid, name"""
    raw = _psql_csv(sql)
    log.info("MBDB returned %d (mbid, name) rows.", len(raw))
    # Dropping rows with missing names
    raw = raw.dropna(subset=["name", "mbid"])
    raw["name"] = raw["name"].astype(str)
    log.info("After NaN filter: %d rows.", len(raw))
    # Grouping by mbid, capping, generating pairs
    groups = raw.groupby("mbid")["name"].apply(lambda names: sorted(set(names))).to_dict()
    log.info("Artists with aliases: %d", len(groups))
    rows = []
    n_capped = 0
    for mbid, names in groups.items():
        if len(names) < 2:
            continue
        if len(names) > MAX_NAMES_PER_ARTIST:
            names = names[:MAX_NAMES_PER_ARTIST]
            n_capped += 1
        for a, b in itertools.combinations(names, 2):
            rows.append({"variant_a": a, "variant_b": b, "to_link": True, "source": "mb_alias_max"})
    if n_capped:
        log.info("Capped %d artists at %d names.", n_capped, MAX_NAMES_PER_ARTIST)
    df = pd.DataFrame(rows)
    log.info("Generated %d positive alias pairs.", len(df))
    dump_parquet(df, GS_MB_MAX_PQ)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Step 2: WRatio filtering + distribution-matched negatives
# ═════════════════════════════════════════════════════════════════════════════
def _compute_wratio_bulk(df: pd.DataFrame, label: str = "") -> pd.Series:
    """Computes WRatio for every row with progress logging."""
    n = len(df)
    log.info("Computing WRatio for %d %s pairs...", n, label)
    wr = df.apply(lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])), axis=1)
    log.info("  Done. WRatio range: [%.0f, %.0f]", wr.min(), wr.max())
    return wr


def step2_assemble_training(positives_all: pd.DataFrame) -> pd.DataFrame:
    """Filters positives to [60,100), distribution-matches negatives from DBSCAN."""
    # Filtering positives
    pos_wr = _compute_wratio_bulk(positives_all, "positive")
    mask = (pos_wr >= WRATIO_LOWER) & (pos_wr < WRATIO_UPPER)
    positives = positives_all[mask].reset_index(drop=True)
    pos_wr_filtered = pos_wr[mask].reset_index(drop=True)
    log.info("Positives in [%d,%d): %d (of %d total).", WRATIO_LOWER, WRATIO_UPPER, len(positives), len(positives_all))
    # Loading DBSCAN negatives
    dbscan = read_parquet(GS_DBSCAN_PQ)
    neg_pool = dbscan[dbscan["to_link"].eq(False)].reset_index(drop=True)
    neg_wr = _compute_wratio_bulk(neg_pool, "negative")
    neg_pool = neg_pool.copy()
    neg_pool["_wr"] = neg_wr
    # Distribution matching with 8 bins in [60, 100)
    n_bins = 8
    bin_edges = np.linspace(60, 100, n_bins + 1)
    pos_hist, _ = np.histogram(pos_wr_filtered, bins=bin_edges)
    pos_fracs = pos_hist / pos_hist.sum()
    n_target = min(len(positives), len(neg_pool))
    log.info("Target negatives: %d (min of %d pos, %d neg pool).", n_target, len(positives), len(neg_pool))
    log.info(
        "Positive WRatio distribution: %s",
        dict(
            zip(
                [f"[{bin_edges[i]:.0f},{bin_edges[i + 1]:.0f})" for i in range(n_bins)],
                pos_hist,
            )
        ),
    )
    neg_pool["_bin"] = pd.cut(neg_pool["_wr"], bins=bin_edges, right=False, labels=False)
    neg_pool = neg_pool.dropna(subset=["_bin"])
    neg_pool["_bin"] = neg_pool["_bin"].astype(int)
    targets = (pos_fracs * n_target).astype(int)
    targets[np.argmax(pos_fracs)] += n_target - targets.sum()
    sampled_parts = []
    shortfall = 0
    available_bins = []
    for i in range(n_bins):
        bin_df = neg_pool[neg_pool["_bin"] == i]
        if len(bin_df) < targets[i]:
            shortfall += targets[i] - len(bin_df)
            sampled_parts.append(bin_df)
        else:
            available_bins.append((i, bin_df, targets[i]))
    if shortfall > 0 and available_bins:
        total_surplus = sum(len(bdf) - t for _, bdf, t in available_bins)
        for i, bin_df, base_target in available_bins:
            surplus = len(bin_df) - base_target
            extra = int(round(shortfall * surplus / max(total_surplus, 1)))
            final_n = min(base_target + extra, len(bin_df))
            sampled_parts.append(bin_df.sample(n=final_n, random_state=RANDOM_STATE))
    else:
        for i, bin_df, target in available_bins:
            sampled_parts.append(bin_df.sample(n=target, random_state=RANDOM_STATE))
    neg_sampled = pd.concat(sampled_parts, ignore_index=True).drop(columns=["_wr", "_bin"])
    log.info("Distribution-matched negatives: %d.", len(neg_sampled))
    # Combining
    train = pd.concat(
        [
            positives[["variant_a", "variant_b", "to_link", "source"]].reset_index(drop=True),
            neg_sampled[["variant_a", "variant_b", "to_link", "source"]].reset_index(drop=True),
        ],
        ignore_index=True,
    )
    log.info("Training set: %d pairs (pos=%d, neg=%d).", len(train), train["to_link"].sum(), (~train["to_link"]).sum())
    return train


# ═════════════════════════════════════════════════════════════════════════════
# Step 3: Precompute hand-crafted features
# ═════════════════════════════════════════════════════════════════════════════
def precompute_features(df: pd.DataFrame) -> np.ndarray:
    """Computes 28 features (23 base + 5 length stats) for every pair."""
    n = len(df)
    log.info("Precomputing %d features for %d pairs...", N_BASE_FEATURES, n)
    feat_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        a, b = str(row["variant_a"]), str(row["variant_b"])
        feats = compute_pair_features(a, b)
        ls = length_stats(f"{a}{{{b}")
        feats.update(ls.to_dict())
        feat_rows.append(feats)
        if (i + 1) % 50000 == 0:
            log.info("  Features: %d/%d (%.0f%%)", i + 1, n, 100 * (i + 1) / n)
    feat_df = pd.DataFrame(feat_rows)
    # Replacing NaN/inf with 0
    arr = feat_df.values.astype(np.float32)
    n_nan = np.isnan(arr).sum() + np.isinf(arr).sum()
    if n_nan > 0:
        log.warning("Replacing %d NaN/inf values in features.", n_nan)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    log.info("Feature matrix: %s, columns: %d", arr.shape, len(feat_df.columns))
    return arr, list(feat_df.columns)


# ═════════════════════════════════════════════════════════════════════════════
# Bai et al. TCN building blocks (same as Exp 9)
# ═════════════════════════════════════════════════════════════════════════════
class Chomp1d(nn.Module):
    """Removes trailing padding to maintain causal convolution."""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Residual block with two causal dilated convolutions + LayerNorm."""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
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
        self.init_weights()

    def init_weights(self):
        """Initialises convolution weights with Kaiming normal."""
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity="relu")

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Stacks TemporalBlocks with exponentially increasing dilation + LayerNorm."""

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
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
                )
            )
        self.network = nn.Sequential(*layers)
        # Adding LayerNorm over channel dim for numerical stability
        self.layer_norm = nn.LayerNorm(num_channels[-1])

    def forward(self, x):
        # x: (B, C, L)
        out = self.network(x)
        # Applying LayerNorm over channel dim: transpose to (B, L, C) → norm → back
        out = self.layer_norm(out.transpose(1, 2)).transpose(1, 2)
        return out


# ═════════════════════════════════════════════════════════════════════════════
# Hybrid model: Siamese TCN + feature branch with skip connection
# ═════════════════════════════════════════════════════════════════════════════
class HybridTCN(nn.Module):
    """Siamese TCN fused with a hand-crafted feature branch.

    The feature branch has a skip connection to the output, ensuring
    features always contribute directly — a stable signal floor.
    """

    def __init__(self, vocab_size, embed_dim, tcn_channels, kernel_size, tcn_dropout, fc_dropout, n_features):
        super().__init__()
        # TCN branch (Siamese, shared weights)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.tcn = TemporalConvNet(embed_dim, tcn_channels, kernel_size=kernel_size, dropout=tcn_dropout)
        pool_dim = tcn_channels[-1] * 2
        tcn_combined_dim = pool_dim * 4  # [h_a; h_b; |h_a-h_b|; h_a*h_b]
        # Feature branch with BatchNorm for stable normalisation
        self.feat_branch = nn.Sequential(
            nn.BatchNorm1d(n_features, eps=1e-3),
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        # Skip connection: features → logit directly (signal floor)
        self.feat_skip = nn.Linear(32, 1)
        # Fusion head
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
        # TCN branch
        h_a = self._encode_seq(x_a)
        h_b = self._encode_seq(x_b)
        h_tcn = torch.cat([h_a, h_b, torch.abs(h_a - h_b), h_a * h_b], dim=1)
        # Feature branch
        h_feat = self.feat_branch(features)
        # Fusion + skip
        main_logit = self.head(torch.cat([h_tcn, h_feat], dim=1)).squeeze(1)
        skip_logit = self.feat_skip(h_feat).squeeze(1)
        return main_logit + skip_logit


# ═════════════════════════════════════════════════════════════════════════════
# Character vocabulary (same as Exp 9)
# ═════════════════════════════════════════════════════════════════════════════
class CharVocab:
    """Maps characters to integer indices with PAD=0 and UNK=1."""

    PAD = 0
    UNK = 1

    def __init__(self):
        self.char2idx = {}
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
        """Returns total vocabulary size including special tokens."""
        return self._next_idx


# ═════════════════════════════════════════════════════════════════════════════
# Dataset for hybrid model
# ═════════════════════════════════════════════════════════════════════════════
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
# Evaluation helpers
# ═════════════════════════════════════════════════════════════════════════════
def _optimal_threshold(y_true, y_prob):
    """Finds the threshold that maximises F1."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    idx = np.argmax(f1s)
    return float(thr[idx]), float(f1s[idx])


def _eval_at(y_true, y_prob, thr):
    """Computes metrics at a given threshold."""
    y_pred = (y_prob >= thr).astype(int)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "threshold": thr,
    }


@torch.no_grad()
def predict_proba(model, loader, device):
    """Returns sigmoid probabilities for all samples."""
    model.eval()
    all_probs = []
    for x_a, x_b, feats, _ in loader:
        logits = model(x_a.to(device), x_b.to(device), feats.to(device))
        # Clamping logits to prevent NaN from extreme values
        logits = torch.clamp(logits, -20.0, 20.0)
        probs = torch.sigmoid(logits).cpu().numpy()
        probs = np.nan_to_num(probs, nan=0.5)
        all_probs.append(probs)
    return np.concatenate(all_probs)


# ═════════════════════════════════════════════════════════════════════════════
# Training loop
# ═════════════════════════════════════════════════════════════════════════════
def train_model(model, train_loader, val_loader, y_val, pos_weight):
    """Trains the hybrid TCN with early stopping on validation AUC."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))
    best_auc = 0.0
    best_state = None
    last_good_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience_ctr = 0
    history = {"train_loss": [], "val_auc": []}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        nan_batches = 0
        for x_a, x_b, feats, labels in train_loader:
            x_a = x_a.to(DEVICE)
            x_b = x_b.to(DEVICE)
            feats = feats.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x_a, x_b, feats)
            logits = torch.clamp(logits, -15.0, 15.0)
            loss = criterion(logits, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                if nan_batches <= 3:
                    log.warning(
                        "NaN/Inf loss at epoch %d, batch %d. Restoring weights.", epoch, n_batches + nan_batches
                    )
                # Restoring last known good weights to recover
                model.load_state_dict({k: v.to(DEVICE) for k, v in last_good_state.items()})
                continue
            loss.backward()
            # Checking for NaN in gradients before stepping
            has_nan_grad = False
            for p in model.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan_grad = True
                    break
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
        # Saving last good state before validation
        last_good_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        val_probs = predict_proba(model, val_loader, DEVICE)
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
            if patience_ctr >= PATIENCE:
                log.info("Early stopping at epoch %d (best val AUC=%.4f).", epoch, best_auc)
                break
    if best_state:
        model.load_state_dict(best_state)
        model.to(DEVICE)
    log.info("Training complete. Best val AUC=%.4f.", best_auc)
    return history


# ═════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═════════════════════════════════════════════════════════════════════════════
def main():
    """Runs the full Experiment 10 pipeline."""
    log.info("=== Experiment 10: Hybrid Siamese TCN + feature floor ===")
    log.info("Device: %s", DEVICE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_STATE)
    # ── Step 1: MBDB extraction ───────────────────────────────────────────
    positives_all = step1_extract_mbdb_max()
    # ── Step 2: Filter + distribution-match negatives ─────────────────────
    train_full = step2_assemble_training(positives_all)
    # ── AVC test set ──────────────────────────────────────────────────────
    avc = read_parquet(AVC_PQ)
    decided = avc[avc["to_link"].notna()].copy()
    test_rows = []
    for _, row in decided.iterrows():
        test_rows.extend(cluster.expand_pairs(row))
    test_raw = pd.DataFrame(test_rows, columns=["variants", "variant_a", "variant_b", "to_link"])
    test_raw["_wr"] = test_raw.apply(lambda r: fuzz.WRatio(str(r["variant_a"]), str(r["variant_b"])), axis=1)
    test_raw = test_raw[(test_raw["_wr"] >= WRATIO_LOWER) & (test_raw["_wr"] < WRATIO_UPPER)]
    test_df = test_raw.drop(columns=["_wr", "variants"]).reset_index(drop=True)
    log.info(
        "AVC test: %d pairs (pos=%d, neg=%d).", len(test_df), test_df["to_link"].sum(), (~test_df["to_link"]).sum()
    )
    # ── Train/val split ───────────────────────────────────────────────────
    train_df, val_df = train_test_split(
        train_full,
        test_size=0.15,
        stratify=train_full["to_link"],
        random_state=RANDOM_STATE,
    )
    log.info("Train/val split: %d / %d", len(train_df), len(val_df))
    # ── Step 3: Precompute features ───────────────────────────────────────
    log.info("Precomputing features for train split...")
    train_feats, feat_cols = precompute_features(train_df)
    log.info("Precomputing features for val split...")
    val_feats, _ = precompute_features(val_df)
    log.info("Precomputing features for test set...")
    test_feats, _ = precompute_features(test_df)
    # Fitting scaler on training features
    scaler = RobustScaler()
    train_feats = scaler.fit_transform(train_feats)
    val_feats = scaler.transform(val_feats)
    test_feats = scaler.transform(test_feats)
    # ── Character vocabulary ──────────────────────────────────────────────
    all_train_names = train_df["variant_a"].astype(str).tolist() + train_df["variant_b"].astype(str).tolist()
    vocab = CharVocab().fit(all_train_names)
    # ── Datasets & loaders ────────────────────────────────────────────────
    train_ds = HybridDataset(train_df, vocab, MAX_SEQ_LEN, train_feats)
    val_ds = HybridDataset(val_df, vocab, MAX_SEQ_LEN, val_feats)
    test_ds = HybridDataset(test_df, vocab, MAX_SEQ_LEN, test_feats)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    y_val = val_df["to_link"].astype(int).values
    y_test = test_df["to_link"].astype(int).values
    # ── Step 4: Build model ───────────────────────────────────────────────
    model = HybridTCN(
        vocab_size=vocab.size,
        embed_dim=EMBED_DIM,
        tcn_channels=TCN_CHANNELS,
        kernel_size=KERNEL_SIZE,
        tcn_dropout=TCN_DROPOUT,
        fc_dropout=FC_DROPOUT,
        n_features=N_BASE_FEATURES,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %s", f"{n_params:,}")
    n_pos = train_df["to_link"].sum()
    n_neg = len(train_df) - n_pos
    pos_weight = float(n_neg / max(n_pos, 1))
    log.info("pos_weight: %.2f (pos=%d, neg=%d)", pos_weight, n_pos, n_neg)
    # ── Step 5: Train ─────────────────────────────────────────────────────
    log.info("Training hybrid model...")
    history = train_model(model, train_loader, val_loader, y_val, pos_weight)
    # ── Step 6: Evaluate ──────────────────────────────────────────────────
    log.info("Evaluating on AVC test set (%d pairs)...", len(test_df))
    test_probs = predict_proba(model, test_loader, DEVICE)
    auc = roc_auc_score(y_test, test_probs)
    default_m = _eval_at(y_test, test_probs, 0.5)
    opt_thr, _ = _optimal_threshold(y_test, test_probs)
    optimal_m = _eval_at(y_test, test_probs, opt_thr)
    best_hi = {"threshold": 0.99, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for t in np.arange(0.50, 0.99, 0.01):
        m = _eval_at(y_test, test_probs, t)
        if m["precision"] >= 0.80 and m["f1"] > best_hi["f1"]:
            best_hi = m
    # ── Results ───────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("EXPERIMENT 10: HYBRID SIAMESE TCN + FEATURE FLOOR")
    print("=" * 100)
    print(f"AUC:            {auc:.4f}")
    print(f"Default (0.5):  P={default_m['precision']:.4f}  R={default_m['recall']:.4f}  F1={default_m['f1']:.4f}")
    print(
        f"Optimal:        P={optimal_m['precision']:.4f}  R={optimal_m['recall']:.4f}  F1={optimal_m['f1']:.4f}  (thr={opt_thr:.3f})"
    )
    print(
        f"High-precision: P={best_hi['precision']:.4f}  R={best_hi['recall']:.4f}  F1={best_hi['f1']:.4f}  (thr={best_hi['threshold']:.3f})"
    )
    print("=" * 100)
    y_pred_opt = (test_probs >= opt_thr).astype(int)
    print(f"\n=== HybridTCN (optimal thr={opt_thr:.3f}) ===")
    print(classification_report(y_test, y_pred_opt, target_names=["no link", "link"]))
    print("── Baselines ──")
    print("Exp 6  (ExtraTrees):  AUC=0.8920, opt F1=0.7050 (thr=0.940)")
    print("Exp 9  (TCN-only):    AUC=0.6505, opt F1=0.3900 (thr=1.000)")
    print(
        f"\nTraining: {len(history['train_loss'])} epochs, "
        f"final loss={history['train_loss'][-1]:.4f}, "
        f"best val AUC={max(history['val_auc']):.4f}"
    )


if __name__ == "__main__":
    main()
