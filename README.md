# Uncertainty-Aware Classification with Conformal Prediction

This repository demonstrates an **uncertainty-aware ML workflow** using
**Conformal Prediction** to produce reliable, coverage-guaranteed outputs.

The project is motivated by **operational decision-making under ambiguity**, where inputs
may be noisy, or heterogeneous, and where knowing when not to trust a model
is as important as raw accuracy.

---

## Project Focus

The core focus is on:
- Producing prediction sets with formal coverage guarantees
- Evaluating reliability vs informativeness trade-offs
- Measuring subgroup robustness using Mondrian (group-conditional) conformal prediction

Rather than optimizing for a single metric, the emphasis is on calibration, stability,
and failure awareness.

---

## Repository Overview
- `notebooks/`: end-to-end analysis and experiments
- `src/`: reusable conformal prediction and evaluation utilities
- `slides/`: presentation summarizing methodology and findings
- `data/`: input dataset

## Methods

- **Baseline models**
  - Logistic Regression
  - Random Forest

- **Uncertainty estimation**
  - Split Conformal Prediction (global coverage control)

- **Subgroup reliability**
  - Mondrian Conformal Prediction (group-conditional coverage)

- **Evaluation**
  - Empirical coverage
  - Prediction set size
  - Subgroup-wise reliability diagnostics

---

## Key Notebook

The main analysis can be found here:

notebooks/03_conformal_prediction_reliability_report.ipynb 

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook
