# Greenwashing Detection — Hierarchical Multi-Label ESG Claim Classification

**MSc Computer Science (Data Analytics) — Capstone Project**  
**Gautami Thakur | University of Galway, 2025**

---

## Overview

This project proposes a two-stage NLP pipeline for detecting and classifying Environmental, Social, and Governance (ESG) sustainability claims from corporate text, with a focus on identifying greenwashing in financial disclosures.

- **Stage 1** — Binary claim detection: does a sentence make a verifiable environmental claim?
- **Stage 2** — Hierarchical multi-label classification: which ESG categories and subcategories does the claim belong to?

The pipeline is built on transformer-based models (ClimateBERT, RoBERTa, DeBERTa-v3), fine-tuned on an expert-annotated dataset aligned with the SASB (Sustainability Accounting Standards Board) taxonomy.

---

## Repository Structure

```
├── ClimateBERT_S1.ipynb        # Stage 1: Binary claim detection with ClimateBERT
├── RoBERTa_S1.ipynb            # Stage 1: Binary claim detection with RoBERTa-base
├── DeBERTa_S1.ipynb            # Stage 1: Binary claim detection with DeBERTa-v3-small
├── ClimateBERT_S2.ipynb        # Stage 2: Hierarchical ESG classification (ClimateBERT gate)
├── RoBERTa_S2.ipynb            # Stage 2: Hierarchical ESG classification (RoBERTa gate)
├── DeBERTa_S2.ipynb            # Stage 2: Hierarchical ESG classification (DeBERTa gate)
├── InterAnnotatorScore.ipynb   # Inter-annotator agreement analysis
├── ESG Annotation Guidelines - SASB aligned.pdf
└── dataset/
```

---

## Dataset

Built on the expert-annotated corpus from [Stammbach et al. (2023)](https://arxiv.org/abs/2209.00507), which contains 3,000+ sentences from corporate sustainability reports, earnings calls, and annual reports. Each sentence was originally labelled as `yes` (claim), `no` (non-claim), or `tie` (ambiguous).

### Extension for this project

The 667 claim-positive (`yes`) sentences were re-annotated with a hierarchical multi-label taxonomy aligned with **SASB standards**, introducing five ESG parent categories and fine-grained subcategories. Subcategories are only assigned when their parent is also present, enforcing logical consistency.

For example:
> *"Reducing our Scope 2 emissions means mainly focusing on sourcing low-carbon electricity"*  
> → **Environment** → GHG Emissions, Energy Management

The resulting dataset supports hierarchical multi-label classification with strong multi-label co-occurrence patterns (e.g., emissions claims frequently overlap with energy management). The dataset exhibits a long-tail distribution — broad categories like **Environment** dominate, while subcategories such as *Critical Incident Risk Management* and *Access and Affordability* are sparsely represented.

---

## Methodology

### Stage 1 — Binary Claim Detection

Three transformer backbones were fine-tuned and evaluated:

| Model | Backbone |
|---|---|
| ClimateBERT | `distilroberta-base-climate-f` |
| RoBERTa | `roberta-base` |
| DeBERTa | `deberta-v3-small` |

**Training setup:** Hyperparameter search over `lr ∈ {1e-5, 2e-5}`, `batch size ∈ {8, 16}`, `epochs ∈ {4, 6}`, with weight decay 0.01, warmup ratio 0.06, max sequence length 256.

**Threshold selection:** Monte-Carlo dropout (12 stochastic passes) with a post-hoc validation threshold sweep. The best threshold is fixed and applied unchanged to the held-out test set.

### Stage 2 — Hierarchical ESG Classification

The Stage 2 encoder is **DeBERTa-v3-small** with a **label-graph head** (masked self-attention over label embeddings), trained with:

- Positive-class–weighted Binary Cross-Entropy (BCE)
- Hierarchy penalty enforcing child ≤ parent (λ = 0.01)
- Contrastive regularisation ablation: `none`, `ntxent_any`, `supcon_any`, `supcon_parent`
- Post-hoc temperature calibration + per-label threshold sweep (parent thresholds constrained by children)
- Light k-NN refinement in embedding space (cosine neighbours from training labels)

Each Stage 2 model runs behind a Stage 1 gate (ClimateBERT, RoBERTa, or DeBERTa), forming three full S1→S2 pipelines.

---

## Results

### Stage 1 — Test Set Performance

| Model | lr / batch / epochs | Threshold | Val F1 | Test P | Test R | Test F1 | Test Acc |
|---|---|---|---|---|---|---|---|
| **ClimateBERT** | 2e-5 / 16 / 4 | 0.30 | 0.857 | 84.0 | 89.0 | **86.4** | **93.0** |
| RoBERTa-base | 1e-5 / 16 / 6 | 0.30 | 0.856 | 81.7 | 85.0 | 83.3 | 91.5 |
| DeBERTa-v3-small | 2e-5 / 8 / 6 | 0.30 | 0.839 | 81.1 | 86.0 | 83.5 | 91.5 |

> Threshold selected on validation set via MC-dropout (12 passes); applied unchanged to test.

#### Comparison against published baselines (Stammbach et al.)

| Model | Source | P | R | F1 | Acc |
|---|---|---|---|---|---|
| ClimateBERT | **This work** | **84.0** | 89.0 | **86.4** | **93.0** |
| ClimateBERT | Published baseline | 76.5 | **92.5** | 83.8 | 90.9 |
| RoBERTa-base | **This work** | **81.7** | 85.0 | **83.3** | **91.5** |
| RoBERTa-base | Published baseline | 73.3 | **94.0** | 82.4 | 89.8 |

ClimateBERT outperforms the published baseline by **+2.6 F1** and **+2.1 Acc**, with notably higher precision (+7.5). It also exceeds the published study's best single model (RoBERTa-large, F1 = 84.9) by **+1.5 F1** and **+1.3 Acc**.

---

### Stage 2 — Hierarchical ESG Classification

Best configurations per pipeline (selected on validation micro-F1 after calibration):

| Pipeline (S1 → S2) | Contrastive | kNN | Val micro-F1 | Test micro-F1 | Test macro-F1 | Test LRAP |
|---|---|---|---|---|---|---|
| **ClimateBERT → DeBERTa** | ntxent_any | Yes | **64.85** | 58.42 | 18.04 | **71.59** |
| DeBERTa → DeBERTa | supcon_any | Yes | 64.85 | 57.63 | 16.85 | 71.20 |
| RoBERTa → DeBERTa | ntxent_any | Yes | 59.79 | **59.85** | **17.79** | 71.39 |

> kNN params: k=25, α=0.1, τ=0.1 (ClimateBERT); k=15, α=0.35 (DeBERTa, RoBERTa); p=2 for RoBERTa.  
> LRAP = Label Ranking Average Precision. Parent/child constraints enforced at decode.

The **ClimateBERT → DeBERTa** pipeline is the final selected model based on strongest validation score, cleanest parent-group t-SNE separation, and highest LRAP.

**Final model config:** DeBERTa-v3-small encoder + label-graph head · positive-class–weighted BCE + hierarchy penalty (λ=0.01) · ntxent_any contrastive term (τ=0.20, coeff=0.10) · temperature calibration · per-label thresholds · kNN refinement (k=25, α=0.1, τ=0.1).

---

## Key Findings

1. **ClimateBERT is the strongest Stage 1 gate** — most stable under MC-dropout, best test F1 and accuracy, outperforms published baselines.
2. **Contrastive regularisation consistently helps** — ntxent_any or supcon_any adds ~3–6 points in validation micro-F1 over no regularisation, with tighter parent-level clusters in t-SNE.
3. **Label-graph head + hierarchy penalty improves logical consistency** — discourages child-without-parent label predictions.
4. **Post-hoc calibration is essential** — temperature scaling and per-label thresholding are necessary for reliable multilabel decisions and fair validation-based model selection.
5. **kNN refinement sharpens decision boundaries** — improves ClimateBERT→DeBERTa (TEST 58.4, LRAP 71.6) and RoBERTa→DeBERTa (TEST ≈ 59.9); small test-time drop observed for DeBERTa→DeBERTa.
6. **Parent categories are well-modelled; rare child labels remain recall-limited** — targeted augmentation or class-balanced training are the most promising avenues for improvement.

---

## Tech Stack

- Python, PyTorch, Hugging Face Transformers
- ClimateBERT (`distilroberta-base-climate-f`), RoBERTa-base, DeBERTa-v3-small
- Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn
- SASB taxonomy for hierarchical annotation

---

## Citation

If you use this work, please cite:

```
Thakur, G. (2025). Greenwashing Detection: Hierarchical Multi-Label ESG Claim Classification.
MSc Capstone Project, University of Galway, Ireland.
```

Dataset originally from:
> Stammbach, D., Webersinke, N., Bingler, J. A., Kraus, M., & Leippold, M. (2023).
> *A Dataset for Detecting Real-World Environmental Claims.*
> arXiv:2209.00507.
