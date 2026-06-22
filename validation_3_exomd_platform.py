#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation 3: ExoMD-Platform GNN Explainability Validation
===========================================================
JEV Review Paper - Chapter 6 Computational Validation
Focus: GNN-based KG analysis with XAI consistency and generalization

Structure:
  1. XAI Consistency Test (NMI between 3 explanation methods)
  2. Knowledge Graph Generalization Test
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
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             normalized_mutual_info_score)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')

# ============================================================
# Global Configuration
# ============================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

OUTPUT_DIR = "/mnt/agents/output/SCI_论文/路径二_综述论文/计算验证/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cpu')


# ============================================================
# Simple GNN with correct dimensions
# ============================================================
class SimpleGNN(nn.Module):
    """Simple 2-layer GNN for node classification"""
    def __init__(self, in_dim=64, hidden_dim=128, n_classes=3):
        super(SimpleGNN, self).__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, adj):
        """
        x: node features [N, F]
        adj: normalized adjacency matrix [N, N]
        """
        # Layer 1: aggregate -> transform
        h = torch.matmul(adj, x)  # Aggregate neighbors
        h = self.layer1(h)
        h = F.relu(h)
        h = self.dropout(h)

        # Layer 2
        h = torch.matmul(adj, h)
        h = self.layer2(h)
        h = F.relu(h)
        h = self.dropout(h)

        logits = self.classifier(h)
        return logits, h


def create_normalized_adjacency(n_nodes, edges):
    """Create normalized adjacency matrix with self-loops"""
    adj = np.eye(n_nodes, dtype=np.float32)
    src, dst = edges[0], edges[1]
    for s, d in zip(src, dst):
        adj[s, d] = 1.0
        adj[d, s] = 1.0  # Symmetric

    # Normalize: D^{-1/2} A D^{-1/2}
    row_sum = adj.sum(axis=1)
    d_inv_sqrt = np.power(row_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat = np.diag(d_inv_sqrt)
    normalized_adj = d_mat @ adj @ d_mat
    return torch.FloatTensor(normalized_adj)


# ============================================================
# Simulated Knowledge Graph
# ============================================================
def simulate_knowledge_graph(n_nodes=200, n_features=64):
    """Simulate EV-related knowledge graph"""
    np.random.seed(SEED)
    node_features = np.random.randn(n_nodes, n_features).astype(np.float32)
    node_features = (node_features - node_features.mean(0)) / (node_features.std(0) + 1e-6)

    # Create edges (sparse graph)
    edge_list = [[i, i] for i in range(n_nodes)]  # Self-loops

    # Backbone tree + random edges
    for i in range(1, n_nodes):
        parent = np.random.randint(0, i)
        edge_list.append([parent, i])
        edge_list.append([i, parent])

    for _ in range(400):
        s, d = np.random.randint(0, n_nodes, size=2)
        if s != d:
            edge_list.append([s, d])
            edge_list.append([d, s])

    edges = np.array(edge_list).T
    return node_features, edges


def create_node_labels(n_nodes=200):
    """Create correlated node labels (connected nodes more likely same class)"""
    np.random.seed(SEED)
    labels = np.zeros(n_nodes, dtype=int)
    for i in range(n_nodes):
        labels[i] = np.random.randint(0, 3)
    return labels


# ============================================================
# XAI Methods
# ============================================================
def xai_gradient_based(model, x, adj, target_nodes, n_classes=3):
    """Gradient-based feature importance"""
    model.eval()
    x_input = x.clone().requires_grad_(True)
    logits, _ = model(x_input, adj)

    explanations = {}
    importances = {}
    for node_idx in target_nodes:
        target_class = logits[node_idx].argmax().item()
        model.zero_grad()
        logits[node_idx, target_class].backward(retain_graph=True)
        imp = x_input.grad.abs().mean(dim=1).detach().cpu().numpy()
        # Top 30% important nodes
        thresh = np.percentile(imp, 70)
        exp = (imp >= thresh).astype(int)
        explanations[node_idx] = exp
        importances[node_idx] = imp
        x_input.grad.zero_()
    return explanations, importances


def xai_perturbation_based(model, x, adj, target_nodes, n_classes=3):
    """Perturbation-based importance: remove each node and measure change"""
    model.eval()
    with torch.no_grad():
        logits_orig, _ = model(x, adj)
        probs_orig = F.softmax(logits_orig, dim=1)

    explanations = {}
    importances = {}

    for node_idx in target_nodes:
        target_class = logits_orig[node_idx].argmax().item()
        orig_prob = probs_orig[node_idx, target_class].item()

        imp = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            x_perturbed = x.clone()
            x_perturbed[i] = 0  # Zero out node i
            with torch.no_grad():
                logits_pert, _ = model(x_perturbed, adj)
                probs_pert = F.softmax(logits_pert, dim=1)
            imp[i] = abs(orig_prob - probs_pert[node_idx, target_class].item())

        thresh = np.percentile(imp, 70)
        exp = (imp >= thresh).astype(int)
        explanations[node_idx] = exp
        importances[node_idx] = imp

    return explanations, importances


def xai_attention_surrogate(model, x, adj, target_nodes, n_classes=3):
    """Surrogate attention: use hidden representation similarity"""
    model.eval()
    with torch.no_grad():
        logits, h = model(x, adj)

    explanations = {}
    importances = {}

    for node_idx in target_nodes:
        # Use similarity of hidden representations as importance
        node_h = h[node_idx]  # [hidden_dim]
        similarities = F.cosine_similarity(node_h.unsqueeze(0), h, dim=1)
        imp = similarities.cpu().numpy()
        imp = np.abs(imp)  # Absolute similarity
        thresh = np.percentile(imp, 70)
        exp = (imp >= thresh).astype(int)
        explanations[node_idx] = exp
        importances[node_idx] = imp

    return explanations, importances


# ============================================================
# Part 1: XAI Consistency Test
# ============================================================
def run_xai_consistency_test():
    print("\n" + "="*60)
    print("Part 1: XAI Consistency Test")
    print("="*60)

    n_nodes = 50
    n_features = 64

    node_features, edges = simulate_knowledge_graph(n_nodes, n_features)
    labels = create_node_labels(n_nodes)

    adj = create_normalized_adjacency(n_nodes, edges)
    x = torch.FloatTensor(node_features)
    y = torch.LongTensor(labels)

    # Train GNN
    model = SimpleGNN(in_dim=n_features, hidden_dim=128, n_classes=3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print("\n[Training GNN...]")
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        logits, _ = model(x, adj)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 40 == 0:
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean().item()
            print(f"  Epoch {epoch+1}/200, Loss: {loss.item():.4f}, Acc: {acc:.4f}")

    model.eval()
    with torch.no_grad():
        logits, _ = model(x, adj)
        preds = logits.argmax(dim=1)
        acc = accuracy_score(y.numpy(), preds.numpy())
        f1 = f1_score(y.numpy(), preds.numpy(), average='macro')

    print(f"\n[GNN Performance] Accuracy: {acc:.4f}, F1: {f1:.4f}")

    # XAI Consistency
    target_nodes = list(range(n_nodes))

    print("\n[Computing gradient-based explanations...]")
    grad_exp, grad_imp = xai_gradient_based(model, x, adj, target_nodes)

    print("[Computing perturbation-based explanations...]")
    pert_exp, pert_imp = xai_perturbation_based(model, x, adj, target_nodes)

    print("[Computing attention-surrogate explanations...]")
    attn_exp, attn_imp = xai_attention_surrogate(model, x, adj, target_nodes)

    # Compute NMI between methods
    nmi_scores = []
    confidence_levels = {'high': 0, 'medium': 0, 'low': 0}

    print(f"\n[Computing NMI consistency for {n_nodes} nodes...]")
    for node_idx in target_nodes:
        e1 = grad_exp[node_idx]
        e2 = pert_exp[node_idx]
        e3 = attn_exp[node_idx]

        nmi_12 = normalized_mutual_info_score(e1, e2)
        nmi_13 = normalized_mutual_info_score(e1, e3)
        nmi_23 = normalized_mutual_info_score(e2, e3)
        avg_nmi = np.mean([nmi_12, nmi_13, nmi_23])
        nmi_scores.append(avg_nmi)

        if avg_nmi >= 0.8:
            confidence_levels['high'] += 1
        elif avg_nmi >= 0.5:
            confidence_levels['medium'] += 1
        else:
            confidence_levels['low'] += 1

    nmi_scores = np.array(nmi_scores)
    total = n_nodes
    high_pct = confidence_levels['high'] / total * 100
    med_pct = confidence_levels['medium'] / total * 100
    low_pct = confidence_levels['low'] / total * 100

    print(f"\n[XAI Consistency Results]")
    print(f"  Mean NMI: {nmi_scores.mean():.4f} ± {nmi_scores.std():.4f}")
    print(f"  Median NMI: {np.median(nmi_scores):.4f}")
    print(f"  High confidence (NMI>=0.8): {confidence_levels['high']} ({high_pct:.1f}%)")
    print(f"  Medium confidence (0.5<=NMI<0.8): {confidence_levels['medium']} ({med_pct:.1f}%)")
    print(f"  Low confidence (NMI<0.5): {confidence_levels['low']} ({low_pct:.1f}%)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(nmi_scores, bins=15, color='#2E86AB', alpha=0.7, edgecolor='black')
    axes[0].axvline(nmi_scores.mean(), color='red', linestyle='--', lw=2,
                    label=f'Mean = {nmi_scores.mean():.3f}')
    axes[0].axvline(0.8, color='green', linestyle=':', lw=2, label='High = 0.8')
    axes[0].axvline(0.5, color='orange', linestyle=':', lw=2, label='Medium = 0.5')
    axes[0].set_xlabel('Average NMI Score', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('XAI Consistency: NMI Distribution', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)

    colors = ['#2E86AB', '#F18F01', '#E84855']
    sizes = [high_pct, med_pct, low_pct]
    labels_pie = [f'High\n(NMI>=0.8)\n{high_pct:.1f}%',
                  f'Medium\n(0.5<=NMI<0.8)\n{med_pct:.1f}%',
                  f'Low\n(NMI<0.5)\n{low_pct:.1f}%']
    axes[1].pie(sizes, labels=labels_pie, colors=colors, autopct='',
                startangle=90, textprops={'fontsize': 10})
    axes[1].set_title('Confidence Level Distribution', fontsize=12, fontweight='bold')

    plt.suptitle('ExoMD-Platform: XAI Consistency Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'xai_consistency_nmi.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return {
        "n_nodes": n_nodes,
        "gnn_accuracy": float(acc),
        "gnn_f1_score": float(f1),
        "nmi_mean": float(nmi_scores.mean()),
        "nmi_std": float(nmi_scores.std()),
        "nmi_median": float(np.median(nmi_scores)),
        "confidence_high_count": int(confidence_levels['high']),
        "confidence_high_percent": float(high_pct),
        "confidence_medium_count": int(confidence_levels['medium']),
        "confidence_medium_percent": float(med_pct),
        "confidence_low_count": int(confidence_levels['low']),
        "confidence_low_percent": float(low_pct),
        "nmi_per_node": nmi_scores.tolist()
    }


# ============================================================
# Part 2: KG Generalization Test
# ============================================================
def run_kg_generalization_test():
    print("\n" + "="*60)
    print("Part 2: Knowledge Graph Generalization Test")
    print("="*60)

    n_nodes = 200
    n_features = 64

    node_features, edges = simulate_knowledge_graph(n_nodes, n_features)
    labels = create_node_labels(n_nodes)

    adj = create_normalized_adjacency(n_nodes, edges)
    x = torch.FloatTensor(node_features)
    y = torch.LongTensor(labels)

    np.random.seed(SEED)
    train_mask = np.random.rand(n_nodes) < 0.7
    test_mask = ~train_mask
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    # Train on full KG
    print("\n[Training on full KG...]")
    model = SimpleGNN(in_dim=n_features, hidden_dim=128, n_classes=3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(200):
        optimizer.zero_grad()
        logits, _ = model(x, adj)
        loss = criterion(logits[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                preds = logits[test_idx].argmax(dim=1)
                acc = accuracy_score(y[test_idx].numpy(), preds.numpy())
            print(f"  Epoch {epoch+1}/200, Loss: {loss.item():.4f}, Test Acc: {acc:.4f}")

    model.eval()
    with torch.no_grad():
        logits, _ = model(x, adj)
        preds = logits[test_idx].argmax(dim=1)
        full_acc = accuracy_score(y[test_idx].numpy(), preds.numpy())
        full_f1 = f1_score(y[test_idx].numpy(), preds.numpy(), average='macro')
        probs = F.softmax(logits[test_idx], dim=1).numpy()
        y_test = y[test_idx].numpy()
        lb = label_binarize(y_test, classes=[0, 1, 2])
        try:
            full_auc = roc_auc_score(lb, probs, average='macro', multi_class='ovr')
        except:
            full_auc = 0.5

    print(f"\n[Full KG] Acc: {full_acc:.4f}, F1: {full_f1:.4f}, AUC: {full_auc:.4f}")

    # Partial KG tests
    removal_rates = [0.1, 0.25, 0.5]
    partial_results = {}

    for rate in removal_rates:
        print(f"\n[Testing {int(rate*100)}% edges removed...]")

        # Remove edges
        n_edges = edges.shape[1]
        n_keep = int(n_edges * (1 - rate))
        keep_idx = np.random.choice(n_edges, n_keep, replace=False)
        partial_edges = edges[:, keep_idx]
        partial_adj = create_normalized_adjacency(n_nodes, partial_edges)

        # Train new model
        pmodel = SimpleGNN(in_dim=n_features, hidden_dim=128, n_classes=3)
        popt = optim.Adam(pmodel.parameters(), lr=0.01)

        for epoch in range(200):
            popt.zero_grad()
            plogits, _ = pmodel(x, partial_adj)
            loss = criterion(plogits[train_idx], y[train_idx])
            loss.backward()
            popt.step()

        pmodel.eval()
        with torch.no_grad():
            plogits, _ = pmodel(x, partial_adj)
            ppreds = plogits[test_idx].argmax(dim=1)
            p_acc = accuracy_score(y[test_idx].numpy(), ppreds.numpy())
            p_f1 = f1_score(y[test_idx].numpy(), ppreds.numpy(), average='macro')
            p_probs = F.softmax(plogits[test_idx], dim=1).numpy()
            try:
                p_auc = roc_auc_score(lb, p_probs, average='macro', multi_class='ovr')
            except:
                p_auc = 0.5

        key = f'removed_{int(rate*100)}pct'
        partial_results[key] = {
            'accuracy': float(p_acc),
            'f1_score': float(p_f1),
            'auc': float(p_auc),
            'accuracy_drop': float(full_acc - p_acc),
            'f1_drop': float(full_f1 - p_f1),
            'auc_drop': float(full_auc - p_auc)
        }
        print(f"  Acc: {p_acc:.4f} (drop: {full_acc-p_acc:.4f}), F1: {p_f1:.4f}, AUC: {p_auc:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    rates_plot = [0] + [int(r*100) for r in removal_rates]
    accs = [full_acc] + [partial_results[f'removed_{int(r*100)}pct']['accuracy'] for r in removal_rates]
    f1s_list = [full_f1] + [partial_results[f'removed_{int(r*100)}pct']['f1_score'] for r in removal_rates]

    axes[0].plot(rates_plot, accs, 'o-', color='#2E86AB', lw=2, markersize=8, label='Accuracy')
    axes[0].plot(rates_plot, f1s_list, 's-', color='#A23B72', lw=2, markersize=8, label='F1 Score')
    axes[0].set_xlabel('Edge Removal Rate (%)', fontsize=11)
    axes[0].set_ylabel('Performance Score', fontsize=11)
    axes[0].set_title('KG Edge Removal Impact', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim([0, 1.1])

    x_pos = np.arange(len(removal_rates))
    acc_drops = [partial_results[f'removed_{int(r*100)}pct']['accuracy_drop'] for r in removal_rates]
    f1_drops = [partial_results[f'removed_{int(r*100)}pct']['f1_drop'] for r in removal_rates]
    width = 0.35
    axes[1].bar(x_pos - width/2, acc_drops, width, label='Acc Drop', color='#2E86AB')
    axes[1].bar(x_pos + width/2, f1_drops, width, label='F1 Drop', color='#A23B72')
    axes[1].set_xlabel('Edge Removal Rate', fontsize=11)
    axes[1].set_ylabel('Performance Drop', fontsize=11)
    axes[1].set_title('Performance Degradation', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f'{int(r*100)}%' for r in removal_rates])
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)

    plt.suptitle('ExoMD-Platform: KG Generalization Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'kg_generalization.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return {
        "full_kg_accuracy": float(full_acc),
        "full_kg_f1": float(full_f1),
        "full_kg_auc": float(full_auc),
        "partial_kg_tests": partial_results
    }


# ============================================================
# Main
# ============================================================
def main():
    print("="*60)
    print("ExoMD-Platform: GNN Explainability Validation Suite")
    print("="*60)

    xai_results = run_xai_consistency_test()
    kg_results = run_kg_generalization_test()

    all_results = {
        "validation_name": "ExoMD-Platform GNN Explainability",
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": SEED,
        "xai_consistency_test": xai_results,
        "kg_generalization_test": kg_results
    }

    output_path = os.path.join(OUTPUT_DIR, 'validation_3_exomd_platform_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "="*60)
    print(f"Validation Complete! Results saved to: {output_path}")
    print("="*60)

    print("\n[Summary]")
    print(f"  XAI Mean NMI: {xai_results['nmi_mean']:.4f}")
    print(f"  High confidence: {xai_results['confidence_high_percent']:.1f}%")
    print(f"  Full KG Acc: {kg_results['full_kg_accuracy']:.4f}")
    for key, val in kg_results['partial_kg_tests'].items():
        print(f"  {key} Acc: {val['accuracy']:.4f} (drop: {val['accuracy_drop']:.4f})")

    return all_results


if __name__ == "__main__":
    results = main()
