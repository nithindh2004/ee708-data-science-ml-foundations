# Exploratory Data Analysis of a Three-Class Dataset

A systematic EDA of a 342-sample, three-class, four-feature dataset: summary statistics, IQR, and univariate, bivariate and multivariate plots, with attention to class separability and redundant features.

## Approach
Class and sex frequencies, describe / IQR, histogram, box, violin, scatter, KDE contours, hexbin, correlation heat map (pooled and within-class), and a pair plot.

## Result
Class B is separable on a single feature; classes A and C differ mainly in Feature 1; Features 3 and 4 are strongly correlated (r = 0.87); Features 1 and 2 show a **Simpson's paradox** (pooled r = −0.24, within each class +0.39 to +0.65).

## Files
| File | Contents |
|---|---|
| `exploratory_data_analysis.py` | full analysis |
| `exploratory_data_analysis.ipynb` | notebook with outputs |
| `data/eda_dataset.csv` | dataset |
| `report.pdf` | paper-style write-up |
| `figures/` | all plots |

## Data
`data/eda_dataset.csv` is included.

## Run
```bash
pip install pandas seaborn matplotlib
python exploratory_data_analysis.py
```

The full write-up (method, figures, results and discussion) is in [`report.pdf`](report.pdf).
