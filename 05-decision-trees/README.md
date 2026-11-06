# Decision Trees: Regression Tree + Depth-Pruned Classification Tree

A fully grown squared-error regression tree on a small dataset, and a Gini classification tree whose depth is chosen from the test-set SSE of its predicted probabilities.

## Approach
CART regression tree with no depth limit (root split checked by enumerating every candidate threshold); Gini classification trees with maximum depth 1–6, pruned at the depth after which the change in SSE becomes negligible.

## Result
| Tree | Result |
|---|---|
| Regression | depth 3, 8 leaves, zero training error; root split x1 ≤ 4.5 (SSE 82.75) |
| Classification | full: depth 6, 16 leaves, 95.6 % test; **pruned: depth 3, 8 leaves, same 95.6 % test** |

## Files
| File | Contents |
|---|---|
| `decision_trees.py` | both trees |
| `decision_trees.ipynb` | notebook with outputs |
| `data/` | tree_train.csv, tree_test.csv |
| `report.pdf` | paper-style write-up |
| `figures/` | all plots |

## Data
Both CSV files are included in `data/`.

## Run
```bash
pip install numpy pandas scikit-learn matplotlib
python decision_trees.py
```

The full write-up (method, figures, results and discussion) is in [`report.pdf`](report.pdf).
