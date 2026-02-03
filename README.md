# Uncertainty-Aware Classification with Conformal Prediction

This repository demonstrates an **uncertainty-aware machine learning workflow** using
**Conformal Prediction** to produce reliable, coverage-guaranteed outputs.

The project is motivated by **operational decision-making under ambiguity**, where inputs
may be noisy, or heterogeneous, and where knowing when not to trust a model
is as important as raw accuracy.

---

## Project Focus

The core focus is on:
- Producing **prediction sets** with formal coverage guarantees
- Evaluating **reliability vs informativeness** trade-offs
- Measuring **subgroup robustness** using Mondrian (group-conditional) conformal prediction

Rather than optimizing for a single metric, the emphasis is on **calibration, stability,
and failure awareness**.

---

## Repository Structure
.
├── notebooks/
│ └── 03_conformal_prediction_reliability_report.ipynb
├── src/
│ ├── conformal.py
│ └── metrics.py
├── slides/
│ └── reliability_conformal_prediction_operational_decisioning.pptx
├── data/
├── requirements.txt
└── README.md

- `notebooks/`: End-to-end analysis and experiments
- `src/`: Reusable conformal prediction and evaluation utilities
- `slides/`: Presentation summarizing methodology and findings
- `data/`: Input dataset

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

This notebook walks through:
1. Data preparation and train/calibration/test splits  
2. Baseline model training  
3. Construction of conformal prediction sets  
4. Reliability and subgroup coverage evaluation  

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook
