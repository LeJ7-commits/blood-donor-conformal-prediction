# Blood Donor Availability Prediction (Conformal Prediction)

This repository contains a machine learning project on predicting blood donor availability, with a focus on uncertainty-aware outputs using **Conformal Prediction** (Split CP + Mondrian CP).

## Contents
- Notebook: `notebooks/Group1_Project2_Notebook.ipynb`
- Slides: `slides/` (project presentation)
- Data: `data/` (dataset files, if included)

## Methods
- Baseline models: Logistic Regression, Random Forest
- Uncertainty: Split Conformal Prediction (global coverage)
- Subgroup reliability: Mondrian Conformal Prediction (group-conditional coverage)

## How to run
```bash
pip install -r requirements.txt
jupyter notebook
