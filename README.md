# Exosome-AI Seven-Module Computational Ecosystem
## Academic Reproducibility Package

**Version:** 2.1.0  
**Date:** 2026-06-22  
**License:** MIT License (Academic Use)  
**Corresponding Author:** Wei Lian (lianwubio@163.com)

---

## Overview

This repository contains the complete computational validation scripts, benchmark datasets, and analysis pipelines for the manuscript:

> **"AI End-to-End Computational Ecosystem for MISEV2023-Compliant sEV Liquid Biopsy: A Systematic Review & Open Computational Resource Suite"**  
> Submitted to *Journal of Extracellular Vesicles* (JEV)

All validation experiments described in Chapter 6 of the manuscript can be fully reproduced using the scripts provided here. No proprietary software is required—only open-source Python packages.

---

## System Requirements

- **OS:** Linux (Ubuntu 20.04+), macOS (12+), Windows 10/11 (WSL2 recommended)
- **Python:** 3.9 or higher
- **RAM:** Minimum 4 GB (8 GB recommended)
- **Storage:** ~500 MB for code + synthetic datasets
- **GPU:** Optional (CPU-only execution supported)

---

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/LivegenoBiotech/exosome-ai-ecosystem.git
cd exosome-ai-ecosystem

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run all validations (sequential, ~15 minutes on CPU)
python run_all_validations.py

# 4. Run individual validation modules
python validation_1_exoqc_pro.py --output results/v1
python validation_2_exomarker_ai.py --output results/v2
python validation_3_exomd_platform.py --output results/v3
python validation_4_pipeline.py --output results/v4

# 5. Generate all figures
python generate_all_figures.py
```

---

## Repository Structure

```
exosome-ai-ecosystem/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Container specification
├── run_all_validations.py             # Master script: run all 4 validations
├── generate_all_figures.py           # Master script: generate all 12 figures
│
├── validation_1_exoqc_pro.py          # ExoQC-Pro 3D QC validation
│   ├── MPCI marker protein consistency (ROC, AUC, sensitivity, specificity)
│   ├── RLDF RNA length distribution fidelity (KS test, EMD)
│   ├── EMII membrane integrity CNN (accuracy, confusion matrix, kappa)
│   └── Multi-center batch correction (kBET, silhouette, variance ratio)
│
├── validation_2_exomarker_ai.py      # ExoMarker-AI validation
│   ├── 4-group ablation study (Full, w/o QC, w/o Fusion, w/o KG, w/o Attention)
│   ├── DeLong test vs. 3 baselines (LASSO, Random Forest, SVM)
│   ├── Modality missing robustness (0%/25%/50%/75% RNA/Protein dropout)
│   └── Bootstrap 95% CI for all metrics
│
├── validation_3_exomd_platform.py    # ExoMD-Platform GNN validation
│   ├── GNN classification accuracy
│   ├── Triple XAI consistency (NMI distribution)
│   ├── Knowledge graph robustness (10%/25%/50% edge dropout)
│   └── Confidence stratification
│
├── validation_4_pipeline.py          # End-to-end pipeline simulation
│   ├── 7-stage execution timing
│   ├── QC pass/fail gate statistics
│   ├── Diagnostic accuracy
│   └── Throughput benchmark
│
├── utils.py                           # Shared utility functions
├── statistical_tests.py              # DeLong, bootstrap CI, kBET, Cohen's d
│
├── results/                           # Output directory (auto-created)
│   ├── validation_1_exoqc_pro_results.json
│   ├── validation_2_exomarker_ai_results.json
│   ├── validation_3_exomd_platform_results.json
│   ├── validation_4_pipeline_results.json
│   └── results_final_with_stats.json  # Merged master results
│
├── figures/                           # Generated figures (auto-created)
│   ├── Figure_PRISMA.png
│   ├── Figure_ExoQC_Architecture.png
│   ├── Figure_ExoMarker_Architecture.png
│   ├── Figure_ExoMD_Architecture.png
│   ├── Figure_Ecosystem_Pipeline.png
│   ├── Figure_RoB_Traffic_Light.png
│   ├── Figure_ROC_Curves.png
│   ├── Figure_Ablation_Bar.png
│   ├── Figure_MPCI_Distribution.png
│   ├── Figure_Batch_Correction.png
│   ├── Figure_XAI_NMI.png
│   └── Figure_Efficiency_Comparison.png
│
└── supplementary/
    ├── S1_PRISMA_materials.md        # PRISMA checklist + search strategies
    ├── S2_exoqc_code.md              # ExoQC-Pro algorithm pseudocode
    ├── S3_ablation_raw_data.csv      # Complete ablation numerical results
    ├── S4_gnn_xai_scripts.py         # GNN + XAI computation scripts
    ├── S5_pipeline_reports/          # End-to-end example reports
    └── S6_tool_comparison.csv        # Open-source EV tool benchmark table
```

---

## Validation Summary

### Validation 1: ExoQC-Pro Three-Dimensional Quality Control

| Metric | Value | 95% CI | Test |
|--------|-------|--------|------|
| MPCI AUC | 1.000 | [1.000, 1.000] | Bootstrap (n=1000) |
| MPCI Sensitivity | 1.000 | — | Threshold optimization |
| MPCI Specificity | 1.000 | — | — |
| RLDF KS statistic | 0.793 | [0.783, 0.807] | Bootstrap |
| RLDF p-value | <0.001 | — | Mann-Whitney U |
| EMII Accuracy | 1.000 | [1.000, 1.000] | CNN classification |
| kBET (before correction) | 0.000 | — | k-nearest neighbor batch test |
| kBET (after correction) | 0.967 | — | — |

### Validation 2: ExoMarker-AI Ablation & Benchmark

| Configuration | AUC | Accuracy | F1 | Delta vs Full | DeLong p |
|--------------|-----|----------|-----|---------------|----------|
| Full Model | 0.995 | 0.927 | 0.927 | — | — |
| w/o QC Gating | 0.994 | 0.902 | 0.903 | -0.001 | 1.000 |
| w/o Attention | 0.986 | 0.927 | 0.928 | -0.009 | 1.000 |
| w/o Fusion | 0.999 | 0.951 | 0.951 | +0.004 | 1.000 |
| w/o KG | 0.989 | 0.927 | 0.927 | -0.005 | 0.579 |

### Validation 3: ExoMD-Platform GNN Explainability

| Metric | Value |
|--------|-------|
| GNN Accuracy | 1.000 [1.000, 1.000] |
| XAI Mean NMI | 0.072 ± 0.042 |
| KG 10% edge drop | +0.041 accuracy |
| KG 25% edge drop | +0.068 accuracy |

### Validation 4: End-to-End Pipeline

| Metric | Value |
|--------|-------|
| QC Pass Rate | 91.0% |
| Diagnostic Accuracy | 89.0% |
| Processing Time | 90.9 ms/sample |
| Throughput | 39,604 samples/hour |
| Speedup vs Traditional | 1,984,158× |

---

## Datasets

All validation experiments use **publicly available datasets**:

| Dataset | URL | Usage |
|---------|-----|-------|
| ExoRBase v2 | http://www.exorbase.org | RNA length distribution profiles |
| Vesiclepedia | http://microvesicles.org | Protein marker reference catalogs |
| EV-TRACK | https://evtrack.org | Methodology quality scores |
| STRING v12 | https://string-db.org | Protein-protein interactions |
| DisGeNET v7 | https://www.disgenet.org | Disease-gene associations |

**No proprietary or restricted-access data is used.** All synthetic data generators are included in the validation scripts with fixed random seeds (SEED=42) for full reproducibility.

---

## Docker Support

```bash
# Build container
docker build -t exosome-ai-ecosystem:2.1 .

# Run all validations
docker run -v $(pwd)/results:/app/results exosome-ai-ecosystem:2.1

# Interactive mode
docker run -it exosome-ai-ecosystem:2.1 bash
```

---

## Citation

If you use these validation scripts or software platforms in your research, please cite:

> Lian W. AI End-to-End Computational Ecosystem for MISEV2023-Compliant sEV Liquid Biopsy: A Systematic Review & Open Computational Resource Suite. *J Extracell Vesicles*. 2026;XX(X):eXXXXX.

---

## Contact

For academic research collaboration or test version requests:
- **Email:** lianwubio@163.com
- **Company:** Shanghai Livegeno Biotech CO.,LTD
- **Address:** Room 518, Bldg 1, Lane 88, Haiyang 2nd Rd, Pudong New Area, Shanghai 201306, China

---

## Software Copyright Notices

| Software | Registration Flow Number |
|----------|-------------------------|
| ExoMarker-AI | 2026R11L1933104 |
| ExoQC-Pro | 2026R11L1935402 |
| ExoScale | 2026R11L1935741 |
| ExoScreen | 2026R11L1936020 |
| ExoMD-Platform | 2026R11L1935024 |
| ExoLIMS | 2026R11L1931790 |
| CAED-Platform | 2026R11L1931190 |

All software copyright applications were filed with the China Copyright Protection Center on 2026-06-16.
