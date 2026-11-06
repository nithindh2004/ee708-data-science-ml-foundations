# Regression Models: Capacity, Gradient Descent and Regularisation

Three regression studies: polynomial degree vs. generalisation on noisy linear data, logistic regression trained by hand-written gradient descent, and ordinary least squares vs. LASSO vs. ridge with correctly mapped penalty strengths.

## Approach
Polynomials of degree 1–3 on 100 synthetic samples; logistic regression with batch gradient descent (learning rates 0.01 and 0.05, 20 iterations, plus a 3,000-iteration reference run); OLS / LASSO / ridge on 10 standardised features, including a LASSO regularisation path.

## Result
| Study | Finding |
|---|---|
| Polynomial | test MSE 0.654 / 0.636 / 0.642 for degrees 1–3: extra terms only fit noise |
| Logistic GD | 20 iterations reach only the 74 % majority rate; 3,000 iterations reach 88 % |
| Regularisation | correctly mapped LASSO test MSE 4.64 vs OLS 4.72; a 160× too-strong LASSO zeroes 6/10 coefficients and reaches 21.8 |

## Files
| File | Contents |
|---|---|
| `regression_models.py` | all three studies |
| `regression_models.ipynb` | notebook with outputs |
| `data/` | logistic_data.csv, regularisation_data.csv |
| `report.pdf` | paper-style write-up |
| `figures/` | all plots |

## Data
Both CSV files are included in `data/`.

## Run
```bash
pip install numpy pandas scikit-learn matplotlib
python regression_models.py
```

The full write-up (method, figures, results and discussion) is in [`report.pdf`](report.pdf).
