# Clustering: K-Means Model Selection + Agglomerative Clustering from Scratch

K-means on 2,360 2-D points with the number of clusters chosen by the elbow method and the silhouette score, plus a from-scratch agglomerative clustering (single, average and complete linkage) verified against SciPy.

## Approach
K-means++ with k = 1–15 on standardised features (WCSS, relative drop and silhouette); a naive O(n³) agglomerative algorithm that builds a SciPy-format linkage matrix and draws dendrograms.

## Result
| Part | Finding |
|---|---|
| K-means | elbow and silhouette both pick k = 5 (silhouette 0.580), but a second elbow and the scatter show 7 groups |
| Hierarchical | all merge heights match SciPy exactly; single linkage chains (final merge 10.7 vs 94.9 for complete) |

## Files
| File | Contents |
|---|---|
| `clustering.py` | both parts |
| `clustering.ipynb` | notebook with outputs |
| `data/` | kmeans_2d.csv, hierarchical_1d.csv |
| `report.pdf` | paper-style write-up |
| `figures/` | all plots |

## Data
Both CSV files are included in `data/`.

## Run
```bash
pip install numpy pandas scipy scikit-learn matplotlib
python clustering.py
```

The full write-up (method, figures, results and discussion) is in [`report.pdf`](report.pdf).
