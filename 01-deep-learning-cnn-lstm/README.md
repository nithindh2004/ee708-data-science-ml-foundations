# Deep Learning for Images and Sequences: CNN + LSTM

A three-block CNN trained from scratch on tf_flowers (3,670 images, 5 classes) and a stacked LSTM that forecasts the next day's closing price from the previous 60 days, each evaluated with a critical eye: a data-leakage check for the CNN and a naive baseline for the LSTM.

## Approach
**CNN:** 3 × (Conv 3×3 → MaxPool) with 32/64/128 filters → Dense 128 → Dropout 0.5 → softmax (3.3 M params), Adam, 10 epochs.  
**LSTM:** LSTM(50, return_sequences) → LSTM(50) → Dense 25 → Dense 1 (31.9 k params), 60-day windows, chronological 80/20 split, 20 epochs.

## Result
| Model | Result | Note |
|---|---|---|
| CNN | 92.6 % "test" accuracy | the original `take`/`skip` split re-shuffled every epoch, so all 734 test images were also trained on; the released code uses a fixed split |
| LSTM | RMSE 5.1–10.4 (across seeds) | worse than the naive "tomorrow = today" forecast (RMSE 2.98): test prices exceed the training range |

## Files
| File | Contents |
|---|---|
| `deep_learning.py` | CNN and LSTM (run `python deep_learning.py lstm` for the LSTM only) |
| `deep_learning.ipynb` | notebook version |
| `data/stock_close.csv` | 2,014 daily closing prices |
| `report.pdf` | paper-style write-up |
| `figures/` | all plots |

## Data
`data/stock_close.csv` is included. tf_flowers is downloaded automatically by `tensorflow_datasets`.

## Run
```bash
pip install tensorflow tensorflow-datasets scikit-learn seaborn matplotlib pandas
python deep_learning.py lstm   # ~1 min on CPU
python deep_learning.py cnn    # downloads tf_flowers (~220 MB); GPU recommended
```

The full write-up (method, figures, results and discussion) is in [`report.pdf`](report.pdf).
