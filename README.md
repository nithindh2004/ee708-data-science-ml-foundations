# Data Science & Machine Learning Projects

Six projects covering the core machine-learning toolkit: deep networks for images and sequences, neural networks written from scratch, regression and regularisation, clustering, decision trees, and exploratory data analysis.

**Course:** EE708 Fundamentals of Data Science and Machine Intelligence, IIT Kanpur · **Instructor:** Dr. Rajesh M. Hegde

Each project folder contains runnable code (`.py` + `.ipynb`), its dataset, a **paper-style PDF report** (abstract, method, results, discussion, references), the figures and a short README. Projects are ordered from the most advanced models to the foundations.

## Projects

| # | Project | Techniques | Key finding | Report |
|---|---|---|---|---|
| 01 | [Deep learning: CNN + LSTM](01-deep-learning-cnn-lstm) | CNN image classification (tf_flowers), stacked-LSTM price forecasting | Found train/test leakage in a `tf.data` split; the LSTM loses to a naive baseline because test prices exceed the training range | [PDF](01-deep-learning-cnn-lstm/report.pdf) |
| 02 | [Neural networks from scratch](02-neural-networks-from-scratch) | Perceptron rule, hand-derived backpropagation, gradient check | 43-parameter MLP: 98.3 % train / 100 % test on Iris | [PDF](02-neural-networks-from-scratch/report.pdf) |
| 03 | [Regression models](03-regression-models) | Polynomial fits, gradient-descent logistic regression, LASSO / ridge | Penalty strength decides whether regularisation helps (test MSE 4.64) or hurts (21.8) | [PDF](03-regression-models/report.pdf) |
| 04 | [Clustering](04-clustering) | K-means + elbow / silhouette, agglomerative clustering from scratch | k = 5 by both criteria, yet the data contain 7 groups; from-scratch linkage matches SciPy | [PDF](04-clustering/report.pdf) |
| 05 | [Decision trees](05-decision-trees) | CART regression tree, Gini tree with depth pruning | Pruned depth-3 tree: half the leaves, same 95.6 % test accuracy | [PDF](05-decision-trees/report.pdf) |
| 06 | [Exploratory data analysis](06-exploratory-data-analysis) | Summary statistics, distribution and correlation plots | Simpson's paradox: r = −0.24 pooled, +0.39 to +0.65 within classes | [PDF](06-exploratory-data-analysis/report.pdf) |

## Repository layout

```
├── 01-deep-learning-cnn-lstm/          deep_learning.py · .ipynb · data/
├── 02-neural-networks-from-scratch/    neural_networks_from_scratch.py · .ipynb · data/
├── 03-regression-models/               regression_models.py · .ipynb · data/
├── 04-clustering/                      clustering.py · .ipynb · data/
├── 05-decision-trees/                  decision_trees.py · .ipynb · data/
├── 06-exploratory-data-analysis/       exploratory_data_analysis.py · .ipynb · data/
├── requirements.txt
└── README.md
```

Every project folder also contains `report.pdf`, `figures/` and a `README.md`.

## Setup

```bash
git clone <this-repo-url>
cd EE708-Data-Science-Machine-Intelligence
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd 04-clustering && python clustering.py      # each script writes its plots to ./figures
```

All datasets are included in each project's `data/` folder, except tf_flowers and Iris, which load automatically. Projects 02–06 run on a CPU in seconds. The CNN in project 01 is best run on a GPU.

## Reproducibility notes

- Projects 02–06 and the LSTM part of 01 were re-run from the released code; the reports use those outputs.
- While preparing this repository, a few issues in the original code were found and corrected. Each one is explained in its report:
  - **01:** the CNN train/test split re-shuffled every epoch (leakage).
  - **01:** the LSTM scaler is now fitted on the training period only.
  - **02:** the MLP's MSE gradient through the softmax.
  - **03:** the LASSO penalty mapped to scikit-learn's scaling; the row-index column is excluded from the logistic regression.
  - **05:** the pruning depth is taken from the correct end of the minimal-ΔSSE interval.
- The CNN results in project 01 come from the original run and are reported with the leakage caveat; re-running the fixed pipeline gives an honest held-out accuracy.

## Tech stack

Python · NumPy · pandas · scikit-learn · SciPy · TensorFlow / Keras · Matplotlib · seaborn · LaTeX

## Author

**Nithin D H** — B.Tech, IIT Kanpur
