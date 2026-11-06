# Neural Networks from Scratch: Perceptron + Backpropagation MLP

A perceptron and a one-hidden-layer neural network written in plain NumPy, with every forward and backward step coded by hand and the gradient verified by finite differences.

## Approach
**Perceptron:** step activation, learning rate 0.01, 20 passes, three random initialisations.  
**MLP:** 4 → 5 (tanh) → 3 (softmax), MSE loss with the gradient taken through the softmax Jacobian, per-sample SGD, learning rate 0.01, 1,000 epochs on Iris.

## Result
| Model | Result |
|---|---|
| Perceptron | 100 % / 100 % / 95 % for seeds 42 / 7 / 123: convergence speed depends on initialisation |
| MLP (43 params) | 98.3 % train, **100 % test** (30/30); gradient check rel. error 2e-9 |

## Files
| File | Contents |
|---|---|
| `neural_networks_from_scratch.py` | both models |
| `neural_networks_from_scratch.ipynb` | notebook with outputs |
| `data/perceptron_2d.csv` | 20-point 2-D dataset |
| `report.pdf` | paper-style write-up |
| `figures/` | all plots |

## Data
`data/perceptron_2d.csv` is included; Iris loads from scikit-learn.

## Run
```bash
pip install numpy pandas scikit-learn matplotlib
python neural_networks_from_scratch.py
```

The full write-up (method, figures, results and discussion) is in [`report.pdf`](report.pdf).
