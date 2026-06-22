#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation 2: ExoMarker-AI Ablation + Benchmark Testing
=======================================================
JEV Review Paper - Chapter 6 Computational Validation
Focus: Multi-modal fusion with ablation studies

Structure:
  1. Five ablation experiments
  2. Missing modality robustness test
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (roc_auc_score, accuracy_score, confusion_matrix,
                             classification_report, precision_recall_fscore_support)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ============================================================
# Global Configuration
# ============================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

OUTPUT_DIR = "/mnt/agents/output/SCI_论文/路径二_综述论文/计算验证/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ============================================================
# Data Simulation
# ============================================================
def simulate_multimodal_data(n_samples=401, n_rna=1000, n_protein=108, n_classes=3):
    """
    Simulate multi-modal EV data with realistic complexity:
    - RNA-seq features (1000-dim): miRNA, lncRNA, circRNA expression
    - Protein features (108-dim): surface markers, cargo proteins
    - 3 classes: Healthy, Cancer Type A, Cancer Type B
    - Moderate class overlap and noise for realistic ablation comparison
    """
    np.random.seed(SEED)

    # Class distribution
    n_per_class = [134, 134, 133]  # ~401 total
    n_samples = sum(n_per_class)

    # RNA features: different signature genes for each class
    rna_data = []
    labels = []

    # Class-specific RNA signatures (smaller effect size for realistic difficulty)
    for cls in range(n_classes):
        # Base expression (high variance background)
        expr = np.random.normal(5.0, 2.5, size=(n_per_class[cls], n_rna))

        # Add class-specific upregulated markers (moderate effect, some overlap)
        marker_start = cls * 40
        marker_end = marker_start + 40
        expr[:, marker_start:marker_end] += np.random.normal(1.8, 1.2, size=(n_per_class[cls], 40))

        # Add some shared cancer markers (classes 1 and 2) - weak signal
        if cls > 0:
            expr[:, 200:220] += np.random.normal(1.0, 0.8, size=(n_per_class[cls], 20))

        # Substantial noise for realistic difficulty
        expr += np.random.normal(0, 1.5, size=expr.shape)
        expr = np.clip(expr, 0, 15)
        expr = np.nan_to_num(expr, nan=5.0)

        rna_data.append(expr)
        labels.extend([cls] * n_per_class[cls])

    X_rna = np.vstack(rna_data)

    # Protein features: 108 proteins (surface markers + cargo)
    protein_data = []
    for cls in range(n_classes):
        # Base expression (higher variance)
        expr = np.random.normal(4.0, 2.0, size=(n_per_class[cls], n_protein))

        # Class-specific protein markers (moderate effect)
        marker_start = cls * 12
        marker_end = marker_start + 12
        expr[:, marker_start:marker_end] += np.random.normal(1.5, 0.9, size=(n_per_class[cls], 12))

        # Shared markers (weak)
        if cls > 0:
            expr[:, 50:58] += np.random.normal(0.8, 0.6, size=(n_per_class[cls], 8))

        # Add noise
        expr += np.random.normal(0, 1.0, size=expr.shape)
        expr = np.clip(expr, 0, 12)
        expr = np.nan_to_num(expr, nan=4.0)

        protein_data.append(expr)

    X_protein = np.vstack(protein_data)
    y = np.array(labels)

    # Normalize
    X_rna = (X_rna - X_rna.mean(axis=0)) / (X_rna.std(axis=0) + 1e-6)
    X_protein = (X_protein - X_protein.mean(axis=0)) / (X_protein.std(axis=0) + 1e-6)
    X_rna = np.nan_to_num(X_rna, nan=0.0)
    X_protein = np.nan_to_num(X_protein, nan=0.0)

    return X_rna, X_protein, y


# ============================================================
# Model Definitions
# ============================================================
class RNAEncoder(nn.Module):
    def __init__(self, input_dim=1000, latent_dim=128):
        super(RNAEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


class ProteinEncoder(nn.Module):
    def __init__(self, input_dim=108, latent_dim=64):
        super(ProteinEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


class QCAnchorModule(nn.Module):
    """Quality Control Anchor Module: adds QC-guided attention"""
    def __init__(self, latent_dim=128):
        super(QCAnchorModule, self).__init__()
        self.qc_predictor = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.qc_embedding = nn.Linear(1, latent_dim)

    def forward(self, z):
        qc_score = self.qc_predictor(z)
        qc_embedded = self.qc_embedding(qc_score)
        # QC-guided residual connection
        z_enhanced = z + 0.3 * qc_embedded
        return z_enhanced, qc_score


class DynamicFusionModule(nn.Module):
    """Dynamic modality fusion with learnable weights"""
    def __init__(self, rna_dim=128, prot_dim=64, output_dim=128):
        super(DynamicFusionModule, self).__init__()
        self.fusion_weight = nn.Sequential(
            nn.Linear(rna_dim + prot_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
        self.fusion_proj = nn.Linear(rna_dim + prot_dim, output_dim)

    def forward(self, z_rna, z_prot):
        concat = torch.cat([z_rna, z_prot], dim=1)
        weights = self.fusion_weight(concat)  # [w_rna, w_prot]
        w_rna = weights[:, 0:1]
        w_prot = weights[:, 1:2]

        # Weighted concatenation
        z_fused = torch.cat([w_rna * z_rna, w_prot * z_prot], dim=1)
        z_fused = self.fusion_proj(z_fused)
        return z_fused, weights


# ============================================================
# 5 Ablation Model Variants
# ============================================================
class ExoMarkerAI_Full(nn.Module):
    """Complete ExoMarker-AI model"""
    def __init__(self, n_rna=1000, n_protein=108, n_classes=3, use_qc=True, use_dynamic_fusion=True):
        super(ExoMarkerAI_Full, self).__init__()
        self.rna_encoder = RNAEncoder(n_rna, 128)
        self.protein_encoder = ProteinEncoder(n_protein, 64)
        self.use_qc = use_qc
        self.use_dynamic_fusion = use_dynamic_fusion

        if use_qc:
            self.qc_rna = QCAnchorModule(128)
            self.qc_protein = QCAnchorModule(64)

        if use_dynamic_fusion:
            self.fusion = DynamicFusionModule(128, 64, 128)
            classifier_input = 128
        else:
            classifier_input = 128 + 64  # Simple concatenation

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x_rna, x_protein):
        z_rna = self.rna_encoder(x_rna)
        z_prot = self.protein_encoder(x_protein)

        if self.use_qc:
            z_rna, qc_rna = self.qc_rna(z_rna)
            z_prot, qc_prot = self.qc_protein(z_prot)

        if self.use_dynamic_fusion:
            z_fused, fusion_weights = self.fusion(z_rna, z_prot)
        else:
            z_fused = torch.cat([z_rna, z_prot], dim=1)
            fusion_weights = None

        logits = self.classifier(z_fused)
        return logits, fusion_weights


class RNAOnlyModel(nn.Module):
    """Ablation: RNA single encoder only"""
    def __init__(self, n_rna=1000, n_classes=3):
        super(RNAOnlyModel, self).__init__()
        self.encoder = RNAEncoder(n_rna, 128)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x_rna, x_protein=None):
        z = self.encoder(x_rna)
        return self.classifier(z), None


class ProteinOnlyModel(nn.Module):
    """Ablation: Protein single encoder only"""
    def __init__(self, n_protein=108, n_classes=3):
        super(ProteinOnlyModel, self).__init__()
        self.encoder = ProteinEncoder(n_protein, 64)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_classes)
        )

    def forward(self, x_rna, x_protein):
        z = self.encoder(x_protein)
        return self.classifier(z), None


# ============================================================
# Training and Evaluation Functions
# ============================================================
def train_model(model, train_loader, val_loader, n_epochs=30, lr=0.001, device=DEVICE):
    """Train a model and return history"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    best_val_acc = 0
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_rna, batch_prot, batch_y in train_loader:
            batch_rna = batch_rna.to(device)
            batch_prot = batch_prot.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs, _ = model(batch_rna, batch_prot)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch_rna, batch_prot, batch_y in val_loader:
                batch_rna = batch_rna.to(device)
                batch_prot = batch_prot.to(device)
                batch_y = batch_y.to(device)
                outputs, _ = model(batch_rna, batch_prot)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def evaluate_model(model, test_loader, n_classes=3, device=DEVICE):
    """Evaluate model and return metrics"""
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_rna, batch_prot, batch_y in test_loader:
            batch_rna = batch_rna.to(device)
            batch_prot = batch_prot.to(device)
            outputs, _ = model(batch_rna, batch_prot)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_probs.append(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Accuracy
    acc = accuracy_score(all_labels, all_preds)

    # AUC (macro-average one-vs-rest)
    labels_binarized = label_binarize(all_labels, classes=list(range(n_classes)))
    try:
        auc_macro = roc_auc_score(labels_binarized, all_probs, average='macro', multi_class='ovr')
        auc_per_class = roc_auc_score(labels_binarized, all_probs, average=None, multi_class='ovr')
    except:
        auc_macro = 0.5
        auc_per_class = [0.5] * n_classes

    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': acc,
        'auc_macro': auc_macro,
        'auc_per_class': auc_per_class.tolist() if hasattr(auc_per_class, 'tolist') else auc_per_class,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist()
    }


# ============================================================
# Part 1: 5 Ablation Experiments
# ============================================================
def run_ablation_experiments():
    print("\n" + "="*60)
    print("Part 1: 5 Ablation Experiments")
    print("="*60)

    # Generate data
    X_rna, X_protein, y = simulate_multimodal_data()
    n_classes = 3

    print(f"\n[Dataset]")
    print(f"  Total samples: {len(y)}")
    print(f"  RNA features: {X_rna.shape[1]}")
    print(f"  Protein features: {X_protein.shape[1]}")
    print(f"  Classes: {n_classes}")
    for c in range(n_classes):
        print(f"    Class {c}: {np.sum(y==c)} samples")

    # Train-test split
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=y)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=SEED, stratify=y[test_idx])

    # Create datasets
    def create_loader(idx, batch_size=32, shuffle=False):
        dataset = TensorDataset(
            torch.FloatTensor(X_rna[idx]),
            torch.FloatTensor(X_protein[idx]),
            torch.LongTensor(y[idx])
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = create_loader(train_idx, batch_size=32, shuffle=True)
    val_loader = create_loader(val_idx, batch_size=32, shuffle=False)
    test_loader = create_loader(test_idx, batch_size=32, shuffle=False)

    # Define 5 ablation configurations
    ablation_configs = {
        'RNA_only': {'name': '① RNA Single Encoder', 'model_type': 'rna_only'},
        'Protein_only': {'name': '② Protein Single Encoder', 'model_type': 'protein_only'},
        'No_QC': {'name': '③ w/o QC Anchor', 'model_type': 'no_qc'},
        'No_Dynamic_Fusion': {'name': '④ w/o Dynamic Fusion', 'model_type': 'no_dynamic'},
        'Full': {'name': '⑤ Full ExoMarker-AI', 'model_type': 'full'}
    }

    results = {}
    histories = {}

    for key, config in ablation_configs.items():
        print(f"\n{'-'*50}")
        print(f"Running: {config['name']}")
        print(f"{'-'*50}")

        if config['model_type'] == 'rna_only':
            model = RNAOnlyModel(n_rna=1000, n_classes=n_classes)
        elif config['model_type'] == 'protein_only':
            model = ProteinOnlyModel(n_protein=108, n_classes=n_classes)
        elif config['model_type'] == 'no_qc':
            model = ExoMarkerAI_Full(n_rna=1000, n_protein=108, n_classes=n_classes,
                                      use_qc=False, use_dynamic_fusion=True)
        elif config['model_type'] == 'no_dynamic':
            model = ExoMarkerAI_Full(n_rna=1000, n_protein=108, n_classes=n_classes,
                                      use_qc=True, use_dynamic_fusion=False)
        else:  # full
            model = ExoMarkerAI_Full(n_rna=1000, n_protein=108, n_classes=n_classes,
                                      use_qc=True, use_dynamic_fusion=True)

        # Train
        trained_model, history = train_model(model, train_loader, val_loader, n_epochs=30)
        histories[key] = history

        # Evaluate
        metrics = evaluate_model(trained_model, test_loader, n_classes=n_classes)
        results[key] = {
            'config_name': config['name'],
            **metrics
        }

        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  AUC (macro): {metrics['auc_macro']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")

    # Summary comparison
    print(f"\n{'='*60}")
    print("Ablation Experiment Summary")
    print(f"{'='*60}")
    print(f"{'Model':<30} {'Accuracy':>10} {'AUC':>10} {'F1':>10}")
    print(f"{'-'*62}")
    for key, config in ablation_configs.items():
        r = results[key]
        print(f"{r['config_name']:<28} {r['accuracy']:>10.4f} {r['auc_macro']:>10.4f} {r['f1_score']:>10.4f}")

    # Plot ablation comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    model_names = [results[k]['config_name'] for k in ablation_configs.keys()]
    x_pos = np.arange(len(model_names))

    # Bar chart: Accuracy and AUC
    accs = [results[k]['accuracy'] for k in ablation_configs.keys()]
    aucs = [results[k]['auc_macro'] for k in ablation_configs.keys()]
    f1s = [results[k]['f1_score'] for k in ablation_configs.keys()]

    width = 0.25
    axes[0].bar(x_pos - width, accs, width, label='Accuracy', color='#2E86AB')
    axes[0].bar(x_pos, aucs, width, label='AUC', color='#A23B72')
    axes[0].bar(x_pos + width, f1s, width, label='F1 Score', color='#F18F01')
    axes[0].set_ylabel('Score', fontsize=11)
    axes[0].set_title('Ablation Study: Performance Comparison', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(model_names, rotation=15, ha='right', fontsize=8)
    axes[0].legend(fontsize=9)
    axes[0].set_ylim([0, 1.1])
    axes[0].grid(axis='y', alpha=0.3)

    # Training curves for full model
    axes[1].plot(histories['Full']['val_acc'], color='#2E86AB', lw=2, label='Full Model')
    axes[1].plot(histories['RNA_only']['val_acc'], color='#A23B72', lw=1.5, linestyle='--', label='RNA Only')
    axes[1].plot(histories['Protein_only']['val_acc'], color='#F18F01', lw=1.5, linestyle='--', label='Protein Only')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Validation Accuracy', fontsize=11)
    axes[1].set_title('Training Curves: Validation Accuracy', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot confusion matrices for full model
    cm = np.array(results['Full']['confusion_matrix'])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('ExoMarker-AI (Full): Confusion Matrix', fontsize=12, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, ['Healthy', 'Cancer A', 'Cancer B'], fontsize=10)
    plt.yticks(tick_marks, ['Healthy', 'Cancer A', 'Cancer B'], fontsize=10)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'full_model_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return results


# ============================================================
# Part 2: Missing Modality Robustness Test
# ============================================================
def run_missing_modality_test():
    print("\n" + "="*60)
    print("Part 2: Missing Modality Robustness Test")
    print("="*60)

    # Generate data
    X_rna, X_protein, y = simulate_multimodal_data()
    n_classes = 3

    # Train-test split
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=y)

    # Train full model
    def create_loader(idx, batch_size=32, shuffle=False):
        dataset = TensorDataset(
            torch.FloatTensor(X_rna[idx]),
            torch.FloatTensor(X_protein[idx]),
            torch.LongTensor(y[idx])
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = create_loader(train_idx, batch_size=32, shuffle=True)
    val_idx, _ = train_test_split(test_idx, test_size=0.5, random_state=SEED, stratify=y[test_idx])
    val_loader = create_loader(val_idx, batch_size=32, shuffle=False)

    # Train full model
    full_model = ExoMarkerAI_Full(n_rna=1000, n_protein=108, n_classes=n_classes,
                                   use_qc=True, use_dynamic_fusion=True)
    trained_model, _ = train_model(full_model, train_loader, val_loader, n_epochs=30)

    # Test with varying modality missing rates
    missing_rates = [0.0, 0.25, 0.50, 0.75]
    results_missing = {
        'missing_rates': missing_rates,
        'rna_missing': {},
        'protein_missing': {}
    }

    for modality in ['rna', 'protein']:
        print(f"\n[Testing {modality.upper()} modality missing...]")
        for rate in missing_rates:
            # Create masked test set
            X_rna_test = X_rna[test_idx].copy()
            X_prot_test = X_protein[test_idx].copy()

            if modality == 'rna':
                # Mask RNA features
                n_features = X_rna_test.shape[1]
                n_mask = int(n_features * rate)
                for i in range(len(X_rna_test)):
                    mask_indices = np.random.choice(n_features, n_mask, replace=False)
                    X_rna_test[i, mask_indices] = 0  # Zero out
            else:
                # Mask protein features
                n_features = X_prot_test.shape[1]
                n_mask = int(n_features * rate)
                for i in range(len(X_prot_test)):
                    mask_indices = np.random.choice(n_features, n_mask, replace=False)
                    X_prot_test[i, mask_indices] = 0

            test_dataset = TensorDataset(
                torch.FloatTensor(X_rna_test),
                torch.FloatTensor(X_prot_test),
                torch.LongTensor(y[test_idx])
            )
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            metrics = evaluate_model(trained_model, test_loader, n_classes=n_classes)

            key = f"missing_{int(rate*100)}pct"
            results_missing[f'{modality}_missing'][key] = {
                'missing_rate': rate,
                'accuracy': metrics['accuracy'],
                'auc_macro': metrics['auc_macro'],
                'f1_score': metrics['f1_score']
            }

            print(f"  {modality.upper()} missing {int(rate*100)}%: Acc={metrics['accuracy']:.4f}, AUC={metrics['auc_macro']:.4f}")

    # Plot degradation curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, modality in enumerate(['rna', 'protein']):
        accs = [results_missing[f'{modality}_missing'][f'missing_{int(r*100)}pct']['accuracy']
                for r in missing_rates]
        aucs = [results_missing[f'{modality}_missing'][f'missing_{int(r*100)}pct']['auc_macro']
                for r in missing_rates]
        f1s = [results_missing[f'{modality}_missing'][f'missing_{int(r*100)}pct']['f1_score']
               for r in missing_rates]

        axes[idx].plot([int(r*100) for r in missing_rates], accs, 'o-', color='#2E86AB', lw=2, markersize=8, label='Accuracy')
        axes[idx].plot([int(r*100) for r in missing_rates], aucs, 's-', color='#A23B72', lw=2, markersize=8, label='AUC')
        axes[idx].plot([int(r*100) for r in missing_rates], f1s, '^-', color='#F18F01', lw=2, markersize=8, label='F1 Score')
        axes[idx].set_xlabel(f'{modality.upper()} Missing Rate (%)', fontsize=11)
        axes[idx].set_ylabel('Performance Score', fontsize=11)
        axes[idx].set_title(f'{modality.upper()} Modality Degradation', fontsize=12, fontweight='bold')
        axes[idx].set_xticks([int(r*100) for r in missing_rates])
        axes[idx].legend(fontsize=9)
        axes[idx].grid(alpha=0.3)
        axes[idx].set_ylim([0, 1.1])

    plt.suptitle('ExoMarker-AI: Robustness to Missing Modalities', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'missing_modality_robustness.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Summary table
    print(f"\n{'='*60}")
    print("Missing Modality Robustness Summary")
    print(f"{'='*60}")
    for modality in ['rna', 'protein']:
        print(f"\n{modality.upper()} Modality Missing:")
        for rate in missing_rates:
            key = f'missing_{int(rate*100)}pct'
            r = results_missing[f'{modality}_missing'][key]
            print(f"  {int(rate*100):>3}% missing: Acc={r['accuracy']:.4f}, AUC={r['auc_macro']:.4f}, F1={r['f1_score']:.4f}")

    return results_missing


# ============================================================
# Main Execution
# ============================================================
def main():
    print("="*60)
    print("ExoMarker-AI: Ablation + Benchmark Validation Suite")
    print("Journal of Extracellular Vesicles - Review Paper")
    print("="*60)

    # Part 1: Ablation experiments
    ablation_results = run_ablation_experiments()

    # Part 2: Missing modality robustness
    missing_results = run_missing_modality_test()

    # Compile all results
    all_results = {
        "validation_name": "ExoMarker-AI Ablation + Benchmark",
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": SEED,
        "device": str(DEVICE),
        "ablation_experiments": ablation_results,
        "missing_modality_robustness": missing_results
    }

    # Save JSON results
    output_path = os.path.join(OUTPUT_DIR, 'validation_2_exomarker_ai_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "="*60)
    print("Validation Complete!")
    print(f"Results saved to: {output_path}")
    print("="*60)

    # Print summary
    print("\n[Key Results]")
    full = ablation_results['Full']
    rna_only = ablation_results['RNA_only']
    prot_only = ablation_results['Protein_only']
    no_qc = ablation_results['No_QC']
    no_dyn = ablation_results['No_Dynamic_Fusion']

    print(f"  Full Model AUC: {full['auc_macro']:.4f}")
    print(f"  w/o QC AUC: {no_qc['auc_macro']:.4f} (delta: {full['auc_macro']-no_qc['auc_macro']:+.4f})")
    print(f"  w/o Dynamic Fusion AUC: {no_dyn['auc_macro']:.4f} (delta: {full['auc_macro']-no_dyn['auc_macro']:+.4f})")
    print(f"  RNA Only AUC: {rna_only['auc_macro']:.4f}")
    print(f"  Protein Only AUC: {prot_only['auc_macro']:.4f}")

    return all_results


if __name__ == "__main__":
    results = main()
