# ============================================================
#   DCMFANet: Deep Cross-Modal Feature Attention Network
#   Multi-Omics Cancer Classification - Complete Pipeline
#
#   REVISED VERSION with fixes for:
#   - Class imbalance (Focal Loss, class weights)
#   - Overfitting (dropout, weight decay, early stopping)
#   - Hyperparameter tuning (expanded search space)
#   - Feature reduction (balanced)
# ============================================================

from imblearn.over_sampling import BorderlineSMOTE

import os
import time
import warnings
import numpy as np
import pandas as pd
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler, label_binarize
from imblearn.over_sampling import SMOTE
from sklearn.metrics.pairwise import cosine_similarity

# Optional imports
try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import dense_to_sparse
except ImportError:
    print(" torch_geometric not installed. GCN layers will not work.")
    GCNConv = None
    dense_to_sparse = None

try:
    import optuna
except ImportError:
    print(" optuna not installed. Hyperparameter tuning will be skipped.")
    optuna = None

try:
    from pycombat import pycombat
except ImportError:
    pycombat = None
    print(" pycombat not installed. ComBat batch correction will be skipped.")

try:
    import shap
    SHAP_AVAILABLE = True
    print(" SHAP imported successfully.")
except ImportError:
    SHAP_AVAILABLE = False
    print(" SHAP not installed. Install with: pip install shap")

warnings.filterwarnings('ignore')


# ============================================================
# ------------------- SETUP & CONFIGURATION -------------------
# ============================================================

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# ------------------- DATA PROCESSING ------------------------
# ============================================================
class DataProcessor:
    def __init__(self, data, feature_order=None):
        self.data = data
        self.fitted = False
        self.selected_indices = None
        self.min_vals = None
        self.max_vals = None
        self.feature_order = feature_order

    def variance_filter(self, n_features=2000):
        variances = self.data.var()
        self.selected_indices = variances.nlargest(n_features).index
        return self.data[self.selected_indices]

    def batch_correct(self, batch_labels=None):
        if batch_labels is not None and pycombat is not None:
            data_corrected = pycombat(self.data.T, batch_labels).T
            self.data = pd.DataFrame(data_corrected,
                                     index=self.data.index,
                                     columns=self.data.columns)
            print("  ✓ Batch correction applied using ComBat.")
        elif batch_labels is not None:
            print("  pycombat not installed; skipping batch correction.")
        return self.data

    def minmax_scale(self):
        self.min_vals = self.data.min()
        self.max_vals = self.data.max()
        data_scaled = (self.data - self.min_vals) / (self.max_vals - self.min_vals + 1e-8)
        return data_scaled

    def fit_transform(self, n_features=2000, batch_labels=None):
        if batch_labels is not None:
            self.batch_correct(batch_labels)
        self.data = self.variance_filter(n_features)
        data_scaled = self.minmax_scale()
        self.fitted = True
        if self.feature_order is not None:
            common = [f for f in self.feature_order if f in data_scaled.columns]
            data_scaled = data_scaled[common]
        return data_scaled.values

    def transform(self, data):
        if not self.fitted:
            raise ValueError("Processor must be fitted first.")
        data_subset = data[self.selected_indices]
        data_scaled = (data_subset - self.min_vals) / (self.max_vals - self.min_vals + 1e-8)
        if self.feature_order is not None:
            common = [f for f in self.feature_order if f in data_scaled.columns]
            data_scaled = data_scaled[common]
        return data_scaled.values


# NEW: supervised feature selection using mutual information
def select_features_by_mi(X, y, n_features):
    """
    Select top n_features using mutual information with the target.
    Returns selected data array.
    """
    selector = SelectKBest(mutual_info_classif, k=n_features)
    X_selected = selector.fit_transform(X, y)
    return X_selected, selector


def load_omics_data(mrna_path, meth_path, mirna_path, label_path,
                    n_features_mrna=2000, n_features_meth=2000, n_features_mirna=500,
                    batch_correct=False, batch_labels=None,
                    feature_orders=None, use_mi=False):   # NEW: use_mi flag
    """Load and preprocess three omics datasets with auto‑transposition."""
    omics1_raw = pd.read_csv(mrna_path, index_col=0)
    omics2_raw = pd.read_csv(meth_path, index_col=0)
    omics3_raw = pd.read_csv(mirna_path, index_col=0)
    labels_df = pd.read_csv(label_path, index_col=0)

    print(f"Initial mRNA shape: {omics1_raw.shape}")
    print(f"Initial Meth shape: {omics2_raw.shape}")
    print(f"Initial miRNA shape: {omics3_raw.shape}")

    # ---- Auto‑detect and transpose if features are rows ----
    feature_prefixes = ('ENSG', 'cg', 'hsa', 'ENSMUSG', 'chr')
    def needs_transpose(df):
        if df.index.size == 0:
            return False
        first_idx = str(df.index[0])
        return any(first_idx.startswith(prefix) for prefix in feature_prefixes)

    if needs_transpose(omics1_raw):
        print("  mRNA appears transposed (features as rows) – transposing...")
        omics1_raw = omics1_raw.T
    if needs_transpose(omics2_raw):
        print("  Methylation appears transposed – transposing...")
        omics2_raw = omics2_raw.T
    if needs_transpose(omics3_raw):
        print("  miRNA appears transposed – transposing...")
        omics3_raw = omics3_raw.T

    print(f"After transpose: mRNA {omics1_raw.shape}, Meth {omics2_raw.shape}, miRNA {omics3_raw.shape}")

    # ---- Align samples ----
    common_samples = set(omics1_raw.index) & set(omics2_raw.index) & set(omics3_raw.index) & set(labels_df.index)
    common_samples = sorted(common_samples)
    print(f"Common samples: {len(common_samples)}")

    omics1_raw = omics1_raw.loc[common_samples]
    omics2_raw = omics2_raw.loc[common_samples]
    omics3_raw = omics3_raw.loc[common_samples]
    y_raw = labels_df.loc[common_samples].values.ravel()

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"Loaded {len(common_samples)} samples.")
    print(f"  mRNA: {omics1_raw.shape[1]} features")
    print(f"  Methylation: {omics2_raw.shape[1]} features")
    print(f"  miRNA: {omics3_raw.shape[1]} features")
    print(f"  Original labels: {np.unique(y_raw)} → Encoded: {np.unique(y)}")

    # ---- Apply variance filtering (or MI selection) ----
    processors = {}
    order1 = feature_orders.get('mRNA', None) if feature_orders else None
    order2 = feature_orders.get('Methylation', None) if feature_orders else None
    order3 = feature_orders.get('miRNA', None) if feature_orders else None

    if use_mi:
        print("Applying supervised feature selection (mutual information)...")
        X1, _ = select_features_by_mi(omics1_raw.values, y, n_features_mrna)
        X2, _ = select_features_by_mi(omics2_raw.values, y, n_features_meth)
        X3, _ = select_features_by_mi(omics3_raw.values, y, n_features_mirna)
        # Keep processors as dummy (not used for transform)
        processors['mRNA'] = None
        processors['Methylation'] = None
        processors['miRNA'] = None
        print(f"After supervised selection: mRNA {X1.shape}, Meth {X2.shape}, miRNA {X3.shape}")
    else:
        processor1 = DataProcessor(omics1_raw, feature_order=order1)
        X1 = processor1.fit_transform(n_features=n_features_mrna,
                                      batch_labels=batch_labels if batch_correct else None)
        processors['mRNA'] = processor1

        processor2 = DataProcessor(omics2_raw, feature_order=order2)
        X2 = processor2.fit_transform(n_features=n_features_meth,
                                      batch_labels=batch_labels if batch_correct else None)
        processors['Methylation'] = processor2

        processor3 = DataProcessor(omics3_raw, feature_order=order3)
        X3 = processor3.fit_transform(n_features=n_features_mirna,
                                      batch_labels=batch_labels if batch_correct else None)
        processors['miRNA'] = processor3

    print(f"Final shapes: mRNA {X1.shape}, Meth {X2.shape}, miRNA {X3.shape}")
    return X1, X2, X3, y, processors


def build_similarity_graph(X, threshold=0.7, metric='cosine'):
    """Build sample similarity graph for GCN with different metrics."""
    if metric == 'cosine':
        sim = cosine_similarity(X)
    elif metric == 'pearson':
        sim = np.corrcoef(X)
        sim = np.maximum(0, sim)
    elif metric == 'euclidean':
        from sklearn.metrics.pairwise import euclidean_distances
        dist = euclidean_distances(X)
        sigma = np.median(dist) if np.median(dist) > 0 else 1.0
        sim = np.exp(-dist ** 2 / (2 * sigma ** 2))
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    np.fill_diagonal(sim, 0)
    sim[sim < threshold] = 0
    return torch.tensor(sim, dtype=torch.float32)


# ============================================================
# ------------------- LOSS FUNCTIONS -------------------------
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Focal Loss for multi-class classification.
        gamma: focusing parameter; alpha: class weights (tensor)
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================
# ------------------- DCMFANet COMPONENTS --------------------
# ============================================================

class OmicsLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            1, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_seq = x.squeeze(1).unsqueeze(-1)
        out, _ = self.lstm(x_seq)
        aggregated = out.mean(dim=1)
        return self.dropout(aggregated)


class OmicsMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_flat = x.squeeze(1)
        return self.mlp(x_flat)


class OmicsTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(1, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_seq = x.squeeze(1).unsqueeze(-1)
        x_proj = self.input_proj(x_seq)
        out = self.transformer(x_proj)
        aggregated = out.mean(dim=1)
        return self.dropout(aggregated)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        if GCNConv is None:
            raise ImportError("torch_geometric required for GCN layers.")
        self.conv = GCNConv(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, adj):
        edge_index, edge_weight = dense_to_sparse(adj)
        out = self.conv(x, edge_index, edge_weight)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out


class CCAM(nn.Module):
    def __init__(self, dim, alpha=0.5):
        super().__init__()
        self.alpha = alpha

        self.proj_cons = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim)
        )

        self.proj_comp = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim)
        )

    def forward(self, P1, P2, P3):
        P_stack = torch.stack([P1, P2, P3], dim=1)

        # Shared representation
        P_avg = P_stack.mean(dim=1)

        # Complementary information
        P_max, _ = P_stack.max(dim=1)

        cons_att = torch.sigmoid(
            self.proj_cons(P_avg)
        )

        comp_att = torch.sigmoid(
            self.proj_comp(P_max)
        )

        att = (
            self.alpha * cons_att
            + (1 - self.alpha) * comp_att
        )

        # Attention-weighted fusion
        P_weighted = P_stack * att.unsqueeze(1)

        fused = P_weighted.mean(dim=1)

        # Consistency
        L_cons = F.mse_loss(
            P_avg,
            fused
        )

        # Normalize representations
        P_norm = F.normalize(
            P_stack,
            p=2,
            dim=-1
        )

        # Complementarity regularization
        pairwise_sim = 0.0
        M = P_stack.size(1)

        for i in range(M):
            for j in range(i + 1, M):
                sim = (
                    P_norm[:, i, :]
                    * P_norm[:, j, :]
                ).sum(dim=-1)
                pairwise_sim += sim.abs().mean()

        L_comp = (
            pairwise_sim
            / (M * (M - 1) / 2)
        )

        return fused, L_cons, L_comp


class LabelConfidenceModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits):
        probs = F.softmax(logits, dim=1)
        G = probs.T @ probs
        G_norm = F.softmax(G, dim=1)
        diag = torch.diag(G_norm)
        loss = -torch.log(diag + 1e-8).mean()
        return loss


class DCMFANet(nn.Module):
    def __init__(self, input_dims, hidden_dim=128, num_classes=2,
                 num_lstm_layers=2, dropout=0.3, alpha=0.5,
                 encoder_type='lstm'):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == 'lstm':
            self.enc1 = OmicsLSTM(input_dims[0], hidden_dim, num_lstm_layers, dropout)
            self.enc2 = OmicsLSTM(input_dims[1], hidden_dim, num_lstm_layers, dropout)
            self.enc3 = OmicsLSTM(input_dims[2], hidden_dim, num_lstm_layers, dropout)
        elif encoder_type == 'mlp':
            self.enc1 = OmicsMLP(input_dims[0], hidden_dim, dropout)
            self.enc2 = OmicsMLP(input_dims[1], hidden_dim, dropout)
            self.enc3 = OmicsMLP(input_dims[2], hidden_dim, dropout)
        elif encoder_type == 'transformer':
            self.enc1 = OmicsTransformer(input_dims[0], hidden_dim, num_layers=num_lstm_layers, dropout=dropout)
            self.enc2 = OmicsTransformer(input_dims[1], hidden_dim, num_layers=num_lstm_layers, dropout=dropout)
            self.enc3 = OmicsTransformer(input_dims[2], hidden_dim, num_layers=num_lstm_layers, dropout=dropout)
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        self.gcn1 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn3 = GCNLayer(hidden_dim, hidden_dim)

        self.ccam = CCAM(hidden_dim, alpha)
        self.lclm = LabelConfidenceModule()

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x1, x2, x3, adj1, adj2, adj3):
        h1 = self.enc1(x1)
        h2 = self.enc2(x2)
        h3 = self.enc3(x3)

        z1 = self.gcn1(h1, adj1)
        z2 = self.gcn2(h2, adj2)
        z3 = self.gcn3(h3, adj3)

        fused, L_cons, L_comp = self.ccam(z1, z2, z3)
        logits = self.classifier(fused)
        L_lcl = self.lclm(logits)

        return logits, L_cons, L_comp, L_lcl, fused


# ============================================================
# ------------------- TRAINING & EVALUATION ------------------
# ============================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, data, adj, labels, optimizer, loss_weights=(1.0, 0.01, 0.01, 0.001),
                class_weights=None, use_focal=False, focal_gamma=2.0):
    model.train()
    x1, x2, x3 = data
    a1, a2, a3 = adj

    optimizer.zero_grad()
    logits, L_cons, L_comp, L_lcl, _ = model(x1, x2, x3, a1, a2, a3)

    if use_focal:
        focal_loss_fn = FocalLoss(gamma=focal_gamma, alpha=class_weights, reduction='mean')
        L_cls = focal_loss_fn(logits, labels)
    else:
        L_cls = F.cross_entropy(logits, labels, weight=class_weights)

    w_cls, w_cons, w_comp, w_lcl = loss_weights
    loss = w_cls * L_cls + w_cons * L_cons + w_comp * L_comp + w_lcl * L_lcl

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item(), L_cls.item(), L_cons.item(), L_comp.item(), L_lcl.item()


def evaluate_model(model, data, adj, labels, return_probs=False):
    model.eval()
    x1, x2, x3 = data
    a1, a2, a3 = adj

    with torch.no_grad():
        logits, L_cons, L_comp, L_lcl, fused = model(x1, x2, x3, a1, a2, a3)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()
    probs_np = probs.cpu().numpy()

    total_classes = logits.shape[1]
    avg_type = 'binary' if total_classes == 2 else 'macro'

    metrics = {
        'accuracy': accuracy_score(labels_np, preds_np),
        'precision': precision_score(labels_np, preds_np, average=avg_type, zero_division=0),
        'recall': recall_score(labels_np, preds_np, average=avg_type, zero_division=0),
        'f1': f1_score(labels_np, preds_np, average=avg_type, zero_division=0),
    }

    present_classes = np.unique(labels_np)
    if len(present_classes) >= 2:
        aucs = []
        for cls in present_classes:
            y_true_bin = (labels_np == cls).astype(int)
            y_score_cls = probs_np[:, cls]
            try:
                auc_cls = roc_auc_score(y_true_bin, y_score_cls)
            except ValueError:
                auc_cls = 0.0
            aucs.append(auc_cls)
        metrics['auc'] = np.mean(aucs)
    else:
        metrics['auc'] = 0.0

    report = classification_report(labels_np, preds_np, output_dict=True, zero_division=0)
    for cls in present_classes:
        metrics[f'precision_class{cls}'] = report[str(cls)]['precision']
        metrics[f'recall_class{cls}'] = report[str(cls)]['recall']
        metrics[f'f1_class{cls}'] = report[str(cls)]['f1-score']

    metrics['confusion_matrix'] = confusion_matrix(labels_np, preds_np)

    if return_probs:
        return metrics, fused, probs_np
    return metrics, fused


def plot_confusion_matrix(cm, class_labels, title, save_path, figsize=(8, 6)):
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels,
                cbar=True, square=True)
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved to {save_path}")


# ============================================================
# ------------------- CORE CROSS-VALIDATION ------------------
# MODIFIED: expanded hidden_dim range, adjusted default patience, etc.
def run_dcmf_cv(X1, X2, X3, y, n_folds=5, n_trials=25, epochs=200,
                loss_weights=(1.0, 0.1, 0.1, 0.01), encoder_type='mlp',
                graph_metric='pearson', graph_threshold=0.7,
                use_smote=True, record_train_val=True,
                save_dir=None, graph_builder=None,
                use_focal=True, focal_gamma=0.5,
                early_stopping_patience=100):
    """
    Run cross-validation for DCMFANet with Optuna hyperparameter tuning.
    """
    if optuna is None:
        print(" Optuna not installed. Using default hyperparameters.")
        n_trials = 1

    if graph_builder is None:
        graph_builder = build_similarity_graph

    num_classes = len(np.unique(y))
    print(f"  Number of classes: {num_classes}")

    set_seed(42)
    input_dims = [X1.shape[1], X2.shape[1], X3.shape[1]]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_fold_results = []
    all_fused_reps = []
    all_train_histories = []
    all_confusion_matrices = []
    all_test_probs = []
    all_test_labels = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X1, y)):
        print(f"\n{'=' * 50}")
        print(f"  DCMFANet - Fold {fold + 1}/{n_folds}")
        print(f"  Encoder: {encoder_type}, Graph: {graph_metric}, SMOTE: {use_smote}")
        print(f"  Focal Loss: {use_focal} (gamma={focal_gamma})")
        print(f"{'=' * 50}")

        X1_train, X1_test = X1[train_idx], X1[test_idx]
        X2_train, X2_test = X2[train_idx], X2[test_idx]
        X3_train, X3_test = X3[train_idx], X3[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if use_smote:
            X_train_combined = np.hstack([X1_train, X2_train, X3_train])
            smote = BorderlineSMOTE(random_state=42, k_neighbors=5)
            X_resampled, y_resampled = smote.fit_resample(X_train_combined, y_train)
            dim1 = X1_train.shape[1]
            dim2 = X2_train.shape[1]
            X1_res = X_resampled[:, :dim1]
            X2_res = X_resampled[:, dim1:dim1 + dim2]
            X3_res = X_resampled[:, dim1 + dim2:]
            print(f"  SMOTE: {len(y_train)} -> {len(y_resampled)} samples")
        else:
            X1_res, X2_res, X3_res = X1_train, X2_train, X3_train
            y_resampled = y_train
            print(f"  No SMOTE: {len(y_train)} samples")

        # ---- Compute class weights ----
        class_counts = np.bincount(y_resampled)
        class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float).to(device)
        class_weights = class_weights / class_weights.mean()

        # Convert to tensors
        x1_train = torch.tensor(X1_res, dtype=torch.float32).unsqueeze(1).to(device)
        x2_train = torch.tensor(X2_res, dtype=torch.float32).unsqueeze(1).to(device)
        x3_train = torch.tensor(X3_res, dtype=torch.float32).unsqueeze(1).to(device)
        y_train_t = torch.tensor(y_resampled, dtype=torch.long).to(device)

        x1_test = torch.tensor(X1_test, dtype=torch.float32).unsqueeze(1).to(device)
        x2_test = torch.tensor(X2_test, dtype=torch.float32).unsqueeze(1).to(device)
        x3_test = torch.tensor(X3_test, dtype=torch.float32).unsqueeze(1).to(device)
        y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

        # Build adjacency matrices
        adj1_train = graph_builder(X1_res, threshold=graph_threshold, metric=graph_metric).to(device)
        adj2_train = graph_builder(X2_res, threshold=graph_threshold, metric=graph_metric).to(device)
        adj3_train = graph_builder(X3_res, threshold=graph_threshold, metric=graph_metric).to(device)
        adj1_test = graph_builder(X1_test, threshold=graph_threshold, metric=graph_metric).to(device)
        adj2_test = graph_builder(X2_test, threshold=graph_threshold, metric=graph_metric).to(device)
        adj3_test = graph_builder(X3_test, threshold=graph_threshold, metric=graph_metric).to(device)

        train_data = (x1_train, x2_train, x3_train)
        train_adj = (adj1_train, adj2_train, adj3_train)
        test_data = (x1_test, x2_test, x3_test)
        test_adj = (adj1_test, adj2_test, adj3_test)

        # Hyperparameter tuning (Optuna)
        if optuna is not None and n_trials > 1:
            def objective(trial):
                # MODIFIED: allow hidden_dim up to 256 for MLP as well
                if encoder_type == 'transformer':
                    hidden_dim = trial.suggest_int('hidden_dim', 32, 256, step=4)
                else:
                    hidden_dim = trial.suggest_int('hidden_dim', 32, 256)   # increased from 128
                lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
                dropout = trial.suggest_float('dropout', 0.3, 0.8)          # keep high dropout,0.3-0.8
                num_layers = trial.suggest_int('num_layers', 1, 3)
                weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)

                val_size = int(0.2 * len(y_resampled))
                indices = np.random.permutation(len(y_resampled))
                val_idx = indices[:val_size]

                x1_val = x1_train[val_idx]
                x2_val = x2_train[val_idx]
                x3_val = x3_train[val_idx]
                y_val = y_train_t[val_idx]

                def subset_adj(adj, idx):
                    return adj[idx][:, idx]

                a1_val = subset_adj(adj1_train, val_idx)
                a2_val = subset_adj(adj2_train, val_idx)
                a3_val = subset_adj(adj3_train, val_idx)

                model = DCMFANet(input_dims, hidden_dim, num_classes=num_classes,
                                 num_lstm_layers=num_layers, dropout=dropout,
                                 encoder_type=encoder_type).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                val_data = (x1_val, x2_val, x3_val)
                val_adj = (a1_val, a2_val, a3_val)
                for _ in range(20):
                    train_epoch(model, train_data, train_adj, y_train_t, optimizer, loss_weights,
                                class_weights=class_weights, use_focal=use_focal, focal_gamma=focal_gamma)
                metrics, _ = evaluate_model(model, val_data, val_adj, y_val)
                return metrics['accuracy']

            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            best_params = study.best_params
            print(f"  Best params: {best_params}")
        else:
            best_params = {'hidden_dim': 128, 'lr': 0.001, 'dropout': 0.5, 'num_layers': 2, 'weight_decay': 1e-4}
            print(f"  Using default params: {best_params}")

        # Train final model with early stopping
        model = DCMFANet(
            input_dims,
            hidden_dim=best_params['hidden_dim'],
            num_classes=num_classes,
            num_lstm_layers=best_params.get('num_layers', 2),
            dropout=best_params.get('dropout', 0.3),
            alpha=0.5,
            encoder_type=encoder_type
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=best_params['lr'],
                                     weight_decay=best_params.get('weight_decay', 1e-4))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        start_time = time.time()
        train_losses, train_accs, val_accs = [], [], []
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            loss, cls_loss, cons_loss, comp_loss, lcl_loss = train_epoch(
                model, train_data, train_adj, y_train_t, optimizer, loss_weights,
                class_weights=class_weights, use_focal=use_focal, focal_gamma=focal_gamma
            )
            scheduler.step(loss)

            with torch.no_grad():
                logits, _, _, _, _ = model(x1_train, x2_train, x3_train, adj1_train, adj2_train, adj3_train)
                train_preds = logits.argmax(dim=1)
                train_acc = (train_preds == y_train_t).float().mean().item()

            val_metrics, _ = evaluate_model(model, test_data, test_adj, y_test_t)
            val_acc = val_metrics['accuracy']

            train_losses.append(loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            # ---- Early stopping ----
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch + 1}/{epochs}: Loss={loss:.4f}, TrainAcc={train_acc:.4f}, ValAcc={val_acc:.4f}")

            if patience_counter >= early_stopping_patience:
                print(f"    Early stopping triggered at epoch {epoch+1}. Best val acc: {best_val_acc:.4f}")
                model.load_state_dict(best_model_state)
                break

        if best_model_state is not None and patience_counter < early_stopping_patience:
            model.load_state_dict(best_model_state)
            print(f"    Using best model from epoch {best_epoch+1} with val acc {best_val_acc:.4f}")

        training_time = time.time() - start_time

        # Final evaluation on test set using best model
        start_inf = time.time()
        metrics, fused, probs_np = evaluate_model(model, test_data, test_adj, y_test_t, return_probs=True)
        inference_time = time.time() - start_inf
        throughput = len(y_test) / inference_time if inference_time > 0 else 0
        n_params = count_parameters(model)

        all_test_probs.append(probs_np)
        all_test_labels.append(y_test)

        cm = metrics['confusion_matrix']
        all_confusion_matrices.append(cm)

        result = {
            'fold': fold + 1,
            'model': 'DCMFANet',
            'encoder': encoder_type,
            'graph_metric': graph_metric,
            'use_smote': use_smote,
            'use_focal': use_focal,
            'focal_gamma': focal_gamma,
            **metrics,
            'training_time': training_time,
            'inference_time': inference_time,
            'throughput': throughput,
            'n_params': n_params,
            'hidden_dim': best_params['hidden_dim'],
            'lr': best_params['lr'],
            'dropout': best_params.get('dropout', 0.3),
            'num_layers': best_params.get('num_layers', 2),
            'weight_decay': best_params.get('weight_decay', 1e-4),
            'early_stop_epoch': best_epoch + 1,
        }
        if 'confusion_matrix' in result:
            result.pop('confusion_matrix')
        all_fold_results.append(result)
        all_fused_reps.append(fused.cpu().numpy())

        all_train_histories.append({
            'fold': fold + 1,
            'train_loss': train_losses,
            'train_acc': train_accs,
            'val_acc': val_accs,
        })

        print(f"\n  Fold {fold + 1} Results:")
        for k, v in metrics.items():
            if k != 'confusion_matrix':
                print(f"    {k}: {v:.4f}")
        print(f"    Confusion Matrix:\n{cm}")

    # Summary statistics
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'confusion_matrix'} for r in all_fold_results])
    summary = {}
    for col in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            n = len(df)
            ci = stats.t.interval(0.95, n - 1, loc=mean_val, scale=std_val / np.sqrt(n))
            summary[col] = {'mean': mean_val, 'std': std_val, 'ci_lower': ci[0], 'ci_upper': ci[1]}

    print("\n" + "=" * 50)
    print("  Overall Performance (Mean ± Std, 95% CI)")
    for k, v in summary.items():
        print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}  [{v['ci_lower']:.4f}, {v['ci_upper']:.4f}]")
    print("=" * 50)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        class_labels = [str(i) for i in range(num_classes)]
        for i, cm in enumerate(all_confusion_matrices):
            plot_confusion_matrix(
                cm, class_labels,
                title=f'Confusion Matrix - Fold {i + 1}',
                save_path=os.path.join(save_dir, f'confusion_matrix_fold_{i + 1}.jpg')
            )
        if len(all_confusion_matrices) > 0:
            overall_cm = np.sum(all_confusion_matrices, axis=0)
            plot_confusion_matrix(
                overall_cm, class_labels,
                title='Overall Confusion Matrix (Summed Across Folds)',
                save_path=os.path.join(save_dir, 'confusion_matrix_overall.jpg')
            )
        print(f" Confusion matrices saved to {save_dir}")

    return all_fold_results, all_fused_reps, all_train_histories, summary, all_test_probs, all_test_labels


# ============================================================
# ------------------- PLOT ROC CURVES ------------------------
# ============================================================
def plot_roc_curves_cv(all_test_probs, all_test_labels, output_dir, class_names=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    probs = np.vstack(all_test_probs)
    y_true = np.concatenate(all_test_labels)
    num_classes = probs.shape[1]
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))

    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        plt.legend(loc='lower right')
    else:
        y_onehot = label_binarize(y_true, classes=range(num_classes))
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            label = f'Class {i}' if class_names is None else class_names[i]
            plt.plot(fpr[i], tpr[i], lw=2, label=f'{label} (AUC = {roc_auc[i]:.3f})')
        fpr_micro, tpr_micro, _ = roc_curve(y_onehot.ravel(), probs.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        plt.plot(fpr_micro, tpr_micro, 'k--', lw=2, label=f'Micro-average (AUC = {roc_auc_micro:.3f})')
        plt.plot([0, 1], [0, 1], 'k:', lw=1, label='Random')
        plt.legend(loc='lower right')

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - DCMFANet (5‑Fold CV)', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves_cv.jpg'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f" ROC curves saved to {output_dir}")


# ============================================================

# ============================================================
# ------------------- ADDITIONAL ANALYSES --------------------
# ============================================================
def run_graph_comparison(X1, X2, X3, y, output_dir, n_folds=3, epochs=20):
    print("\n" + "=" * 60)
    print("  GRAPH CONSTRUCTION COMPARISON")
    print("=" * 60)

    all_results = []
    for metric in ['cosine', 'pearson', 'euclidean']:
        print(f"\n--- Graph Metric: {metric.upper()} ---")
        results, _, _, _, _, _ = run_dcmf_cv(   # unpack 6 values
            X1, X2, X3, y,
            n_folds=n_folds, n_trials=5, epochs=epochs,
            graph_metric=metric, use_smote=True
        )
        for r in results:
            r['graph_metric'] = metric
            all_results.append(r)

    df = pd.DataFrame(all_results)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'graph_comparison.csv'), index=False)
    # Plot
    metrics_to_plot = ['accuracy', 'f1', 'auc']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        means = df.groupby('graph_metric')[metric].mean()
        stds = df.groupby('graph_metric')[metric].std()
        ax.bar(means.index, means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_title(f'{metric.upper()}')
        ax.set_ylabel(metric)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_comparison.jpg'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f" Graph comparison saved to {output_dir}")
    return df


def run_encoder_comparison(X1, X2, X3, y, output_dir, n_folds=3, epochs=20):
    print("\n" + "=" * 60)
    print("  ENCODER COMPARISON (LSTM vs MLP vs Transformer)")
    print("=" * 60)

    all_results = []
    for enc in ['lstm', 'mlp', 'transformer']:
        print(f"\n--- Encoder: {enc.upper()} ---")
        results, _, _, _, _, _ = run_dcmf_cv(   # unpack 6 values
            X1, X2, X3, y,
            n_folds=n_folds, n_trials=5, epochs=epochs,
            encoder_type=enc, use_smote=True
        )
        for r in results:
            r['encoder'] = enc
            all_results.append(r)

    df = pd.DataFrame(all_results)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'encoder_comparison.csv'), index=False)

    metrics_to_plot = ['accuracy', 'f1', 'auc']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        means = df.groupby('encoder')[metric].mean()
        stds = df.groupby('encoder')[metric].std()
        ax.bar(means.index, means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_title(f'{metric.upper()}')
        ax.set_ylabel(metric)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'encoder_comparison.jpg'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f" Encoder comparison saved to {output_dir}")
    return df




def run_smote_comparison(X1, X2, X3, y, output_dir, n_folds=3, epochs=20):
    """Compare with and without SMOTE."""
    print("\n" + "=" * 60)
    print("  SMOTE COMPARISON")
    print("=" * 60)

    all_results = []
    for smote_flag in [True, False]:
        print(f"\n--- SMOTE: {smote_flag} ---")
        results, _, _, _, _, _ = run_dcmf_cv(   # <-- now unpack all 6 values
            X1, X2, X3, y,
            n_folds=n_folds, n_trials=5, epochs=epochs,
            use_smote=smote_flag
        )
        for r in results:
            r['use_smote'] = smote_flag
            all_results.append(r)

    df = pd.DataFrame(all_results)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'smote_comparison.csv'), index=False)

    metrics_to_plot = ['accuracy', 'f1', 'auc']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        means = df.groupby('use_smote')[metric].mean()
        stds = df.groupby('use_smote')[metric].std()
        ax.bar(['No SMOTE', 'With SMOTE'], means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_title(f'{metric.upper()}')
        ax.set_ylabel(metric)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'smote_comparison.jpg'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f" SMOTE comparison saved to {output_dir}")
    return df




def run_graph_perturbation(X1, X2, X3, y, output_dir, n_folds=3, epochs=20):
    print("\n" + "=" * 60)
    print("  GRAPH PERTURBATION (Edge Removal)")
    print("=" * 60)

    all_results = []
    for remove_ratio in [0.0, 0.1, 0.2, 0.3]:
        print(f"\n--- Edge Removal: {remove_ratio * 100:.0f}% ---")

        # Define a custom graph builder that applies edge removal
        def build_graph_perturbed(X, threshold=0.7, metric='cosine'):
            if metric == 'cosine':
                sim = cosine_similarity(X)
            elif metric == 'pearson':
                sim = np.corrcoef(X)
                sim = np.maximum(0, sim)
            elif metric == 'euclidean':
                from sklearn.metrics.pairwise import euclidean_distances
                dist = euclidean_distances(X)
                sigma = np.median(dist) if np.median(dist) > 0 else 1.0
                sim = np.exp(-dist ** 2 / (2 * sigma ** 2))
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            np.fill_diagonal(sim, 0)
            sim[sim < threshold] = 0
            # Remove edges proportionally
            if remove_ratio > 0:
                edges = np.argwhere(sim > 0)
                if len(edges) > 0:
                    n_remove = int(len(edges) * remove_ratio)
                    remove_idx = np.random.choice(len(edges), n_remove, replace=False)
                    for idx in remove_idx:
                        i, j = edges[idx]
                        sim[i, j] = 0
                        sim[j, i] = 0
            return torch.tensor(sim, dtype=torch.float32)

        # Pass the custom builder directly
        results, _, _, _, _, _ = run_dcmf_cv(
            X1, X2, X3, y,
            n_folds=n_folds, n_trials=10, epochs=epochs,
            use_smote=True,
            graph_builder=build_graph_perturbed   # <-- key change
        )

        for r in results:
            r['remove_ratio'] = remove_ratio
            all_results.append(r)

    # Create DataFrame and save
    df = pd.DataFrame(all_results)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'graph_perturbation.csv'), index=False)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_to_plot = ['accuracy', 'f1', 'auc']
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        means = df.groupby('remove_ratio')[metric].mean()
        stds = df.groupby('remove_ratio')[metric].std()
        ax.errorbar(means.index, means, yerr=stds, marker='o', capsize=5)
        ax.set_xlabel('Edge Removal Ratio')
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_perturbation.jpg'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f" Graph perturbation saved to {output_dir}")
    return df





def run_hyperparameter_sensitivity(X1, X2, X3, y, output_dir):
    """Run hyperparameter sensitivity analysis on a single split."""
    print("\n" + "=" * 60)
    print("  HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("=" * 60)

    set_seed(42)
    input_dims = [X1.shape[1], X2.shape[1], X3.shape[1]]
    num_classes = len(np.unique(y))

    # Single train/val split
    X1_tr, X1_val, X2_tr, X2_val, X3_tr, X3_val, y_tr, y_val = train_test_split(
        X1, X2, X3, y, test_size=0.2, stratify=y, random_state=42
    )

    # Apply SMOTE
    X_train_combined = np.hstack([X1_tr, X2_tr, X3_tr])
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_combined, y_tr)
    dim1 = X1_tr.shape[1]
    dim2 = X2_tr.shape[1]
    X1_res = X_res[:, :dim1]
    X2_res = X_res[:, dim1:dim1 + dim2]
    X3_res = X_res[:, dim1 + dim2:]

    # Convert to tensors
    x1_tr = torch.tensor(X1_res, dtype=torch.float32).unsqueeze(1).to(device)
    x2_tr = torch.tensor(X2_res, dtype=torch.float32).unsqueeze(1).to(device)
    x3_tr = torch.tensor(X3_res, dtype=torch.float32).unsqueeze(1).to(device)
    y_tr_t = torch.tensor(y_res, dtype=torch.long).to(device)

    x1_v = torch.tensor(X1_val, dtype=torch.float32).unsqueeze(1).to(device)
    x2_v = torch.tensor(X2_val, dtype=torch.float32).unsqueeze(1).to(device)
    x3_v = torch.tensor(X3_val, dtype=torch.float32).unsqueeze(1).to(device)
    y_v_t = torch.tensor(y_val, dtype=torch.long).to(device)

    adj1_tr = build_similarity_graph(X1_res, threshold=0.7).to(device)
    adj2_tr = build_similarity_graph(X2_res, threshold=0.7).to(device)
    adj3_tr = build_similarity_graph(X3_res, threshold=0.7).to(device)
    adj1_v = build_similarity_graph(X1_val, threshold=0.7).to(device)
    adj2_v = build_similarity_graph(X2_val, threshold=0.7).to(device)
    adj3_v = build_similarity_graph(X3_val, threshold=0.7).to(device)

    train_data = (x1_tr, x2_tr, x3_tr)
    train_adj = (adj1_tr, adj2_tr, adj3_tr)
    val_data = (x1_v, x2_v, x3_v)
    val_adj = (adj1_v, adj2_v, adj3_v)

    default_params = {
        'hidden_dim': 128,
        'num_mlp_layers': 2,
        'dropout': 0.3,
        'alpha': 0.5,
        'lr': 1e-3,
        'loss_weights': (1.0, 0.1, 0.1, 0.01)
    }

    param_grids = {
        'hidden_dim': [32, 64, 128, 256, 512],
        'num_mlp_layers': [1, 2, 3],
        'dropout': [0.0, 0.2, 0.4, 0.6],
        'alpha': [0.0, 0.25, 0.5, 0.75, 1.0],
        'lr': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        'loss_weights_cons': [0.0, 0.01, 0.1, 1.0],
        'loss_weights_comp': [0.0, 0.01, 0.1, 1.0],
        'loss_weights_lcl': [0.0, 0.01, 0.1, 1.0]
    }

    results = {}
    for param_name, values in param_grids.items():
        print(f"\n  Sensitivity: {param_name}")
        param_results = []
        for val in values:
            params = default_params.copy()
            if param_name == 'loss_weights_cons':
                w_cls, _, w_comp, w_lcl = default_params['loss_weights']
                params['loss_weights'] = (w_cls, val, w_comp, w_lcl)
            elif param_name == 'loss_weights_comp':
                w_cls, w_cons, _, w_lcl = default_params['loss_weights']
                params['loss_weights'] = (w_cls, w_cons, val, w_lcl)
            elif param_name == 'loss_weights_lcl':
                w_cls, w_cons, w_comp, _ = default_params['loss_weights']
                params['loss_weights'] = (w_cls, w_cons, w_comp, val)
            else:
                params[param_name] = val

            model = DCMFANet(
                input_dims,
                hidden_dim=params['hidden_dim'],
                num_classes=num_classes,
                num_lstm_layers=params['num_mlp_layers'],
                dropout=params['dropout'],
                alpha=params['alpha']
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

            for epoch in range(30):
                loss, _, _, _, _ = train_epoch(model, train_data, train_adj, y_tr_t,
                                               optimizer, params['loss_weights'])
                scheduler.step(loss)

            metrics, _ = evaluate_model(model, val_data, val_adj, y_v_t)
            param_results.append({
                'param_value': val,
                'accuracy': metrics['accuracy'],
                'f1': metrics['f1'],
                'auc': metrics['auc']
            })
            print(f"    {param_name} = {val}: Acc={metrics['accuracy']:.4f}")

        results[param_name] = param_results

    # Save and plot
    os.makedirs(output_dir, exist_ok=True)

    # Plot
    n_params = len(param_grids)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, (param_name, param_results) in enumerate(results.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        values = [r['param_value'] for r in param_results]
        acc = [r['accuracy'] for r in param_results]
        f1 = [r['f1'] for r in param_results]

        x_pos = np.arange(len(values))
        ax.plot(x_pos, acc, 'o-', label='Accuracy')
        ax.plot(x_pos, f1, 's-', label='F1')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(v) for v in values], rotation=45, ha='right')
        ax.set_xlabel(param_name)
        ax.set_ylabel('Metric')
        ax.set_title(f'Sensitivity: {param_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    for idx in range(len(results), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hyperparameter_sensitivity.jpg'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save CSV
    rows = []
    for param_name, param_results in results.items():
        for r in param_results:
            rows.append({'parameter': param_name, **r})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, 'hyperparameter_sensitivity.csv'), index=False)

    print(f" Hyperparameter sensitivity saved to {output_dir}")
    return results


def run_feature_order_sensitivity(X1, X2, X3, y, output_dir):
    """Evaluate DCMFANet under different feature orderings."""
    print("\n" + "=" * 60)
    print("  FEATURE ORDER SENSITIVITY")
    print("=" * 60)

    set_seed(42)
    input_dims = [X1.shape[1], X2.shape[1], X3.shape[1]]
    num_classes = len(np.unique(y))

    # Single train/val split
    X1_tr, X1_val, X2_tr, X2_val, X3_tr, X3_val, y_tr, y_val = train_test_split(
        X1, X2, X3, y, test_size=0.2, stratify=y, random_state=42
    )

    # Apply SMOTE
    X_train_combined = np.hstack([X1_tr, X2_tr, X3_tr])
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_combined, y_tr)
    dim1 = X1_tr.shape[1]
    dim2 = X2_tr.shape[1]
    X1_res = X_res[:, :dim1]
    X2_res = X_res[:, dim1:dim1 + dim2]
    X3_res = X_res[:, dim1 + dim2:]

    # Build graphs (same for all orderings)
    adj1_tr = build_similarity_graph(X1_res, threshold=0.7).to(device)
    adj2_tr = build_similarity_graph(X2_res, threshold=0.7).to(device)
    adj3_tr = build_similarity_graph(X3_res, threshold=0.7).to(device)
    adj1_v = build_similarity_graph(X1_val, threshold=0.7).to(device)
    adj2_v = build_similarity_graph(X2_val, threshold=0.7).to(device)
    adj3_v = build_similarity_graph(X3_val, threshold=0.7).to(device)

    # Generate different orderings
    n1, n2, n3 = X1.shape[1], X2.shape[1], X3.shape[1]

    np.random.seed(42)
    orderings = [
        ('Genomic', (np.arange(n1), np.arange(n2), np.arange(n3))),
        ('Random_1', (np.random.permutation(n1), np.random.permutation(n2), np.random.permutation(n3))),
        ('Random_2', (np.random.permutation(n1), np.random.permutation(n2), np.random.permutation(n3))),
        ('Random_3', (np.random.permutation(n1), np.random.permutation(n2), np.random.permutation(n3))),
        ('Reverse', (np.arange(n1)[::-1], np.arange(n2)[::-1], np.arange(n3)[::-1])),
    ]

    results = []
    for order_name, (o1, o2, o3) in orderings:
        print(f"\n  Order: {order_name}")

        # Reorder features
        X1_res_o = X1_res[:, o1]
        X2_res_o = X2_res[:, o2]
        X3_res_o = X3_res[:, o3]
        X1_val_o = X1_val[:, o1]
        X2_val_o = X2_val[:, o2]
        X3_val_o = X3_val[:, o3]

        x1_tr = torch.tensor(X1_res_o, dtype=torch.float32).unsqueeze(1).to(device)
        x2_tr = torch.tensor(X2_res_o, dtype=torch.float32).unsqueeze(1).to(device)
        x3_tr = torch.tensor(X3_res_o, dtype=torch.float32).unsqueeze(1).to(device)
        y_tr_t = torch.tensor(y_res, dtype=torch.long).to(device)
        x1_v = torch.tensor(X1_val_o, dtype=torch.float32).unsqueeze(1).to(device)
        x2_v = torch.tensor(X2_val_o, dtype=torch.float32).unsqueeze(1).to(device)
        x3_v = torch.tensor(X3_val_o, dtype=torch.float32).unsqueeze(1).to(device)
        y_v_t = torch.tensor(y_val, dtype=torch.long).to(device)

        train_data = (x1_tr, x2_tr, x3_tr)
        train_adj = (adj1_tr, adj2_tr, adj3_tr)
        val_data = (x1_v, x2_v, x3_v)
        val_adj = (adj1_v, adj2_v, adj3_v)

        model = DCMFANet(input_dims, hidden_dim=128, num_classes=num_classes,
                         num_lstm_layers=2, dropout=0.3, alpha=0.5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        for epoch in range(30):
            loss, _, _, _, _ = train_epoch(model, train_data, train_adj, y_tr_t,
                                           optimizer, (1.0, 0.1, 0.1, 0.01))
            scheduler.step(loss)

        metrics, _ = evaluate_model(model, val_data, val_adj, y_v_t)
        results.append({
            'order': order_name,
            'accuracy': metrics['accuracy'],
            'f1': metrics['f1'],
            'auc': metrics['auc']
        })
        print(f"    Acc={metrics['accuracy']:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'feature_order_sensitivity.csv'), index=False)

    # Plot
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(results))
    width = 0.25
    acc = [r['accuracy'] for r in results]
    f1 = [r['f1'] for r in results]
    auc = [r['auc'] for r in results]

    plt.bar(x_pos - width, acc, width, label='Accuracy')
    plt.bar(x_pos, f1, width, label='F1')
    plt.bar(x_pos + width, auc, width, label='AUC')
    plt.xticks(x_pos, [r['order'] for r in results], rotation=45, ha='right')
    plt.ylabel('Metric')
    plt.title('Feature Order Sensitivity')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_order_sensitivity.jpg'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f" Feature order sensitivity saved to {output_dir}")
    return df


class SHAPModelWrapper:
    def __init__(self, model, dims, device, threshold=0.7, metric='cosine', batch_size=8):
        self.model = model
        self.d1, self.d2, self.d3 = dims
        self.device = device
        self.threshold = threshold
        self.metric = metric
        self.batch_size = batch_size
        self.model.eval()

    def predict_proba(self, x_concat):
        n = x_concat.shape[0]
        all_probs = []
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch = x_concat[start:end]

            x1_np = batch[:, :self.d1]
            x2_np = batch[:, self.d1:self.d1+self.d2]
            x3_np = batch[:, self.d1+self.d2:]

            x1 = torch.tensor(x1_np, dtype=torch.float32).unsqueeze(1).to(self.device)
            x2 = torch.tensor(x2_np, dtype=torch.float32).unsqueeze(1).to(self.device)
            x3 = torch.tensor(x3_np, dtype=torch.float32).unsqueeze(1).to(self.device)

            adj1 = build_similarity_graph(x1_np, threshold=self.threshold, metric=self.metric).to(self.device)
            adj2 = build_similarity_graph(x2_np, threshold=self.threshold, metric=self.metric).to(self.device)
            adj3 = build_similarity_graph(x3_np, threshold=self.threshold, metric=self.metric).to(self.device)

            with torch.no_grad():
                logits, _, _, _, _ = self.model(x1, x2, x3, adj1, adj2, adj3)
                probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
        return np.vstack(all_probs)

def run_shap_analysis(X1, X2, X3, y, output_dir, feature_names=None):
    """
    Trains a DCMFANet model and runs SHAP analysis.
    Saves SHAP summary plots and raw values.
    """
    if not SHAP_AVAILABLE:
        print(" SHAP not available. Skipping SHAP analysis.")
        return None

    print("\n" + "=" * 60)
    print("  SHAP ANALYSIS FOR MODEL INTERPRETABILITY")
    print("=" * 60)

    print(f"Input shapes: mRNA {X1.shape}, Meth {X2.shape}, miRNA {X3.shape}")
    total_features = X1.shape[1] + X2.shape[1] + X3.shape[1]
    print(f"Total features: {total_features}")

    num_classes = len(np.unique(y))
    print(f"  Number of classes: {num_classes}")

    # 1. Prepare data
    X1_tr, X1_te, X2_tr, X2_te, X3_tr, X3_te, y_tr, y_te = train_test_split(
        X1, X2, X3, y, test_size=0.2, stratify=y, random_state=42
    )

    # 2. Apply SMOTE
    X_train_combined = np.hstack([X1_tr, X2_tr, X3_tr])
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_combined, y_tr)
    dim1 = X1_tr.shape[1]
    dim2 = X2_tr.shape[1]
    X1_res = X_res[:, :dim1]
    X2_res = X_res[:, dim1:dim1 + dim2]
    X3_res = X_res[:, dim1 + dim2:]

    # 3. Convert to tensors
    x1_train = torch.tensor(X1_res, dtype=torch.float32).unsqueeze(1).to(device)
    x2_train = torch.tensor(X2_res, dtype=torch.float32).unsqueeze(1).to(device)
    x3_train = torch.tensor(X3_res, dtype=torch.float32).unsqueeze(1).to(device)
    y_train_t = torch.tensor(y_res, dtype=torch.long).to(device)

    # 4. Build adjacency matrices
    adj1_train = build_similarity_graph(X1_res, threshold=0.7, metric='cosine').to(device)
    adj2_train = build_similarity_graph(X2_res, threshold=0.7, metric='cosine').to(device)
    adj3_train = build_similarity_graph(X3_res, threshold=0.7, metric='cosine').to(device)

    # 5. Train model
    input_dims = [X1.shape[1], X2.shape[1], X3.shape[1]]
    model = DCMFANet(
        input_dims, hidden_dim=128, num_classes=num_classes,
        num_lstm_layers=2, dropout=0.3, alpha=0.5, encoder_type='lstm'
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print("Training model for SHAP analysis...")
    train_data = (x1_train, x2_train, x3_train)
    train_adj = (adj1_train, adj2_train, adj3_train)
    for epoch in range(50):  # increased for better convergence
        loss, _, _, _, _ = train_epoch(model, train_data, train_adj, y_train_t,
                                       optimizer, (1.0, 0.1, 0.1, 0.01))
        scheduler.step(loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/50: Loss={loss:.4f}")

    # 6. Prepare wrapper and background data
    dims = (X1.shape[1], X2.shape[1], X3.shape[1])
    wrapper = SHAPModelWrapper(model, dims, device, threshold=0.7, metric='cosine')

    bg_idx = np.random.choice(len(X1_res), min(20, len(X1_res)), replace=False)
    X_bg = np.hstack([X1_res[bg_idx], X2_res[bg_idx], X3_res[bg_idx]])

    n_exp = min(20, len(X1_te))
    exp_idx = np.random.choice(len(X1_te), n_exp, replace=False)
    X_exp = np.hstack([X1_te[exp_idx], X2_te[exp_idx], X3_te[exp_idx]])

    print(f"Background size: {len(X_bg)}, Explanation size: {len(X_exp)}")
    print(f"X_exp shape: {X_exp.shape}")

    # 7. Compute SHAP values
    print("Computing SHAP values (this may take a few minutes)...")
    explainer = shap.KernelExplainer(wrapper.predict_proba, X_bg, link="logit")
    shap_values = explainer.shap_values(X_exp, nsamples=30, l1_reg=False)

    # --- FIX: obtain feature names before using them ---
    if feature_names is None:
        fnames_mrna = [f"mRNA_{i}" for i in range(X1.shape[1])]
        fnames_meth = [f"Meth_{i}" for i in range(X2.shape[1])]
        fnames_mirna = [f"miRNA_{i}" for i in range(X3.shape[1])]
        feature_names = fnames_mrna + fnames_meth + fnames_mirna

    # --- FIX: convert shap_values to per‑class list ---
    if isinstance(shap_values, list):
        # Already a list of arrays per class
        shap_per_class = shap_values
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Shape: (samples, features, classes) -> convert to per‑class list
        shap_per_class = [shap_values[:, :, c] for c in range(num_classes)]
    else:
        raise ValueError(f"Unexpected shap_values type: {type(shap_values)} with shape {shap_values.shape}")

    # Save per‑class SHAP values
    for c in range(num_classes):
        class_shap = shap_per_class[c]   # expected shape (n_samples, n_features)
        # Safety: if class_shap is 3D, squeeze extra dim
        if class_shap.ndim == 3:
            class_shap = class_shap.squeeze(0)
        df_shap_class = pd.DataFrame(class_shap, columns=feature_names)
        df_shap_class.to_csv(os.path.join(output_dir, f'shap_values_class_{c}.csv'), index=False)

    # Choose which class to use for summary plots
    if num_classes == 2:
        shap_plot_data = shap_per_class[1]   # positive class
    else:
        shap_plot_data = shap_per_class[0]   # class 0 for summary

    # ====== SHAPE ADAPTATION ======
    print(f"Raw shap_plot_data shape: {shap_plot_data.shape}")
    if shap_plot_data.ndim == 2:
        # If features are first (n_features, n_samples), transpose
        if shap_plot_data.shape[0] == X_exp.shape[1] and shap_plot_data.shape[1] <= X_exp.shape[0]:
            shap_plot_data = shap_plot_data.T
            print(f"   Transposed to (samples, features): {shap_plot_data.shape}")
        # Align sample count
        if shap_plot_data.shape[0] != X_exp.shape[0]:
            min_samples = min(shap_plot_data.shape[0], X_exp.shape[0])
            print(f" Sample count mismatch: SHAP has {shap_plot_data.shape[0]}, X_exp has {X_exp.shape[0]}. Using first {min_samples} samples.")
            shap_plot_data = shap_plot_data[:min_samples, :]
            X_exp = X_exp[:min_samples, :]
        # Final feature dimension check
        if shap_plot_data.shape[1] != X_exp.shape[1]:
            if shap_plot_data.shape[0] == X_exp.shape[1] and shap_plot_data.shape[1] == X_exp.shape[0]:
                shap_plot_data = shap_plot_data.T
                print(f"   Transposed again to match features: {shap_plot_data.shape}")
            else:
                raise ValueError(f"Feature dimension mismatch: SHAP has {shap_plot_data.shape[1]}, X_exp has {X_exp.shape[1]}")
    else:
        raise ValueError(f"Unexpected shap_plot_data dimension: {shap_plot_data.ndim}")

    # Final check
    if shap_plot_data.shape != X_exp.shape:
        raise ValueError(f"After adaptation, shapes still mismatch: {shap_plot_data.shape} vs {X_exp.shape}")

    # Trim y_exp accordingly
    y_exp_trimmed = y_te[exp_idx][:X_exp.shape[0]]

    # ====== SAVE SHAP VALUES AND IMPORTANCE ======
    os.makedirs(output_dir, exist_ok=True)

    # Save the full SHAP matrix (samples × features)
    df_shap_full = pd.DataFrame(shap_plot_data, columns=feature_names)
    df_shap_full.to_csv(os.path.join(output_dir, 'shap_values_full.csv'), index=False)

    # Save the mean absolute SHAP importance per feature (global importance)
    mean_abs_shap = np.abs(shap_plot_data).mean(axis=0)
    df_importance = pd.DataFrame([mean_abs_shap], columns=feature_names)
    df_importance.to_csv(os.path.join(output_dir, 'shap_feature_importance.csv'), index=False)

    # Save explanation samples (feature values and true labels)
    df_exp = pd.DataFrame(X_exp, columns=feature_names)
    df_exp['true_label'] = y_exp_trimmed
    df_exp.to_csv(os.path.join(output_dir, 'shap_explanation_samples.csv'), index=False)

    # ====== PLOT INDIVIDUAL FIGURES ======
    print("Saving individual SHAP plots...")
    import matplotlib.gridspec as gridspec  # not needed for individual plots, but kept

    # 1. Beeswarm summary plot
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    plt.sca(ax1)
    shap.summary_plot(shap_plot_data, X_exp, feature_names=feature_names,
                      max_display=20, show=False)
    ax1.set_xlabel("SHAP value (impact on model output)", fontsize=14)
    ax1.set_title("SHAP Summary Plot", fontsize=18, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_beeswarm.jpg'), dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # 2. Top-20 feature importance bar chart
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    top_idx = np.argsort(mean_abs_shap)[-20:]
    top_features = [feature_names[i] for i in top_idx]
    top_values = mean_abs_shap[top_idx]

    ax2.barh(range(len(top_idx)), top_values, color='#87CEEB', height=0.7)
    ax2.set_yticks(range(len(top_idx)))
    ax2.set_yticklabels(top_features, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean Absolute SHAP Value", fontsize=14)
    ax2.set_title("Top 20 Features - Global Importance", fontsize=18, weight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_top10_bar.jpg'), dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # 3. Heatmap of SHAP values for top 20 features across samples
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    shap_values_top = shap_plot_data[:, top_idx]
    sns.heatmap(shap_values_top.T, cmap='coolwarm', center=0,
                yticklabels=top_features, xticklabels=False,
                cbar_kws={'label': 'SHAP Value'}, ax=ax3)
    ax3.set_xlabel("Samples", fontsize=14)
    ax3.set_title("Heatmap of Top 20 Features Across Samples", fontsize=18, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_heatmap.jpg'), dpi=300, bbox_inches='tight')
    plt.close(fig3)

    print(f" SHAP analysis completed. Results saved to: {output_dir}")
    return {
        'model': model,
        'shap_values': shap_values,
        'X_exp': X_exp,
        'feature_names': feature_names,
        'y_exp': y_exp_trimmed
    }


# ============================================================
# ------------------- MAIN EXECUTION (for testing) -----------
# ============================================================
def main():
    # ===== CONFIGURATION =====
    data_dir = "D:/Article/dataset/datanew/PAAD/"
    mrna_path = data_dir + "exptransp.csv"
    meth_path = data_dir + "methytransp.csv"
    mirna_path = data_dir + "mirnatransp.csv"
    label_path = data_dir + "labels.csv"

    # MODIFIED: use balanced feature counts
    N_FEATURES_MRNA = 1000
    N_FEATURES_METH = 1000
    N_FEATURES_MIRNA = 200

    RUN_MAIN_CV = False
    RUN_GRAPH_COMPARISON = False
    RUN_ENCODER_COMPARISON = False
    RUN_SMOTE_COMPARISON = False
    RUN_GRAPH_PERTURBATION = False
    RUN_HYPERPARAMETER_SENSITIVITY = False
    RUN_FEATURE_ORDER_SENSITIVITY = False
    RUN_SHAP = False

    print("=" * 60)
    print("  DCMFANet: Multi-Omics Cancer Classification (FINAL REVISION)")
    print("=" * 60)

    X1, X2, X3, y, processors = load_omics_data(
        mrna_path, meth_path, mirna_path, label_path,
        n_features_mrna=N_FEATURES_MRNA,
        n_features_meth=N_FEATURES_METH,
        n_features_mirna=N_FEATURES_MIRNA,
        batch_correct=True, batch_labels=None,
        feature_orders=None,
        use_mi=False   # set to True to use mutual information selection
    )

    num_classes = len(np.unique(y))
    print(f"\nData shape: mRNA={X1.shape}, Meth={X2.shape}, miRNA={X3.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Number of classes: {num_classes}")

    output_base = data_dir + "results_final/"
    os.makedirs(output_base, exist_ok=True)

    if RUN_MAIN_CV:
        out_dir = os.path.join(output_base, "main_cv")
        os.makedirs(out_dir, exist_ok=True)
        print("\n" + "=" * 60)
        print("  RUNNING DCMFANet 5-FOLD CROSS-VALIDATION (final settings)")
        print("=" * 60)

        loss_weights = (1.0, 0.1, 0.1, 0.01)

        results, fused_reps, train_histories, summary, all_test_probs, all_test_labels = run_dcmf_cv(
            X1, X2, X3, y,
            n_folds=5,
            n_trials=15,
            epochs=200,
            loss_weights=loss_weights,
            encoder_type='mlp',          # change to 'lstm' if desired
            graph_metric='pearson',
            use_smote=True,
            use_focal=True,
            focal_gamma=1.0,              # MODIFIED: reduced from 2.0
            early_stopping_patience=15,   # MODIFIED: increased patience
            record_train_val=True,
            save_dir=out_dir
        )

        # Save results
        df_results = pd.DataFrame(results)
        df_results.to_csv(os.path.join(out_dir, "dcmfanet_cv_results.csv"), index=False)
        with pd.ExcelWriter(os.path.join(out_dir, "dcmfanet_cv_results.xlsx")) as writer:
            df_results.to_excel(writer, sheet_name="DCMFANet", index=False)
        summary_df = pd.DataFrame(summary).T
        summary_df.to_csv(os.path.join(out_dir, "summary_metrics.csv"))

        # Plot ROC curves
        plot_roc_curves_cv(all_test_probs, all_test_labels, out_dir)

        # Per-fold performance bar plot
        metrics_to_plot = ['accuracy', 'f1', 'auc']
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 5))
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            ax.bar(df_results['fold'], df_results[metric],
                   yerr=df_results[metric].std() / 2,
                   capsize=5, alpha=0.7, color='skyblue')
            ax.axhline(df_results[metric].mean(), color='red', linestyle='--', label='Mean')
            ax.set_xlabel('Fold')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'DCMFANet {metric.upper()} per fold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "dcmfanet_per_fold.jpg"), dpi=300, bbox_inches='tight')
        plt.close()

        # Training curves (last fold)
        if train_histories:
            hist = train_histories[-1]
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss', color='tab:red')
            ax1.plot(hist['train_loss'], color='tab:red', label='Train Loss')
            ax1.tick_params(axis='y', labelcolor='tab:red')
            ax2 = ax1.twinx()
            ax2.set_ylabel('Accuracy', color='tab:blue')
            ax2.plot(hist['train_acc'], color='tab:blue', linestyle='--', label='Train Acc')
            ax2.plot(hist['val_acc'], color='tab:green', linestyle='-.', label='Val Acc')
            ax2.tick_params(axis='y', labelcolor='tab:blue')
            plt.title(f'Training Curves (Fold {hist["fold"]}) with Early Stopping')
            fig.tight_layout()
            plt.savefig(os.path.join(out_dir, "training_curves.jpg"), dpi=300)
            plt.close()

        print(f"\n✅ Main CV results saved to {out_dir}")

    print("\n" + "=" * 60)
    print("  ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print(f"  Results saved to: {output_base}")
    print("=" * 60)


if __name__ == "__main__":
    main()