# DX799---Milestone-One-# Flood Claim Severity and Storm Damage Modeling (2017)

Weekly Jupyter notebooks for Milestone One of the Boston University Data Science Capstone (Module C, Fall 2026). Each week applies one group of machine learning methods to two 2017 natural disaster datasets, the FEMA National Flood Insurance Program (NFIP) redacted claims and the NOAA Storm Events database.

The project asks what drives the severity of flood-related losses. Earlier work showed that NFIP payouts are almost perfectly predicted by the damage estimate, which is circular, so these notebooks try to explain losses from information that exists apart from the loss itself, such as the building, the policy, the flood, and the storm. Four questions run through the notebooks.

- How large is an NFIP building payout, and does it exceed $50,000?
- How much property damage does a storm event cause when it causes some, and does it report any damage at all?

## Notebooks

There are two notebooks per week, one for each dataset, plus a combined Week 4 notebook that covers both.

| Week | Topic | Notebooks | Main result |
|---|---|---|---|
| 1 | Linear regression with polynomial and interaction terms, multicollinearity and VIF, categorical and continuous features | `Week1-NFIP-Luu-Austin.ipynb`, `Week1-Storm-Luu-Austin.ipynb` | Test R-squared of 0.407 (NFIP log payout) and 0.414 (Storm Events log damage) |
| 2 | Lasso, ridge, and elastic net regression | `Week2-NFIP-Luu-Austin.ipynb`, `Week2-Storm-Luu-Austin.ipynb` | Regularization steadies correlated coefficients but cannot add signal the features do not carry |
| 3 | Forward and backward selection, principal component regression (PCR), partial least squares regression (PLSR) | `Week3-NFIP-Luu-Austin.ipynb`, `Week3-Storm-Luu-Austin.ipynb` | PLSR matches the full model with 6 components while PCR needs 30 or more |
| 4 | Logistic regression and feature scaling | `Week4-NFIP-Luu-Austin.ipynb`, `Week4-Storm-Luu-Austin.ipynb`, `Week4-Combined-Luu-Austin.ipynb` | Test ROC-AUC of 0.809 (NFIP) and 0.821 (Storm Events), plus a close look at scaling, outliers, and non-independent samples |
| 5 | Support vector machines, the kernel trick, and SVM regularization | `Week5-NFIP-Luu-Austin.ipynb`, `Week5-Storm-Luu-Austin.ipynb` | RBF kernel reaches 0.834 and 0.841, and random splits overstate Storm Events scores |
| 6 | Decision trees and random forests | `Week6-NFIP-Luu-Austin.ipynb`, `Week6-Storm-Luu-Austin.ipynb` | Random forests reach 0.875 and 0.917, the best results of the six weeks |

In every week except Week 2, Storm Events models are trained and tested with whole storm episodes kept on one side of the split, since events from the same storm share weather and reporting and would otherwise inflate the scores.

## Data

The raw data files are not included in this repository because of their size (the NFIP file is about 1 GB). To run the notebooks, download them and place them in the same folder as the notebooks.

- **FEMA NFIP redacted claims (v2)**, saved as `FimaNfipClaimsV2.csv`, from https://www.fema.gov/openfema-data-page/fima-nfip-redacted-claims-v2
- **NOAA Storm Events details for 2017**, saved as `StormEvents_details-ftp_v1.0_d2017_c20260519.csv`, from https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/ (unzip the 2017 details file; the `c` date in the file name may differ in newer downloads)

The notebooks filter the NFIP data to losses in 2017.

## Running the notebooks

The notebooks were run with Python 3.14, pandas 2.3, NumPy 2.4, matplotlib 3.10, and scikit-learn 1.8.

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
jupyter notebook
```

Every notebook is saved with its outputs, so the results and figures can be read on GitHub without running anything. A full run of the NFIP notebooks takes a few minutes each, mostly for loading the claims file and for the grid searches.

## Author

Austin Luu, Boston University, Fall 2026
