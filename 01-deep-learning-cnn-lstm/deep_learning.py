# %% [markdown]
# # Deep Learning for Images and Sequences: CNN Flower Classification and LSTM Price Forecasting
#   A. A three-block CNN trained from scratch on tf_flowers (3,670 images, 5 classes)
#   B. A two-layer stacked LSTM that predicts the next day's closing price from the previous 60 days,
#      compared with a naive "tomorrow = today" baseline
#
# Run from this folder:  python deep_learning.py            (both parts)
#                        python deep_learning.py lstm       (only the LSTM part; no download needed)
# Part A downloads tf_flowers (~220 MB) through tensorflow_datasets; a GPU is recommended.

# %%
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
PARTS = [a for a in sys.argv[1:] if a in ("cnn", "lstm")] or ["cnn", "lstm"]

SEED = 42
tf.keras.utils.set_random_seed(SEED)
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#6b6b6b"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "font.size": 10})

# %% [markdown]
# ## A. CNN on tf_flowers

# %%
def run_cnn():
    import seaborn as sns
    import tensorflow_datasets as tfds
    from sklearn.metrics import classification_report, confusion_matrix

    IMG, BATCH, EPOCHS = 128, 32, 10
    # Fixed, disjoint 80/20 split defined by tfds slicing. (Shuffling a tf.data pipeline and then using
    # take/skip re-draws the split every epoch, which leaks test images into training.)
    (ds_tr, ds_te), info = tfds.load("tf_flowers", split=["train[:80%]", "train[80%:]"],
                                     as_supervised=True, with_info=True, shuffle_files=False)
    names = info.features["label"].names

    def prep(img, lab):
        return tf.cast(tf.image.resize(img, [IMG, IMG]), tf.float32) / 255.0, lab

    AT = tf.data.AUTOTUNE
    ds_tr = ds_tr.map(prep, num_parallel_calls=AT).cache().shuffle(3000, seed=SEED).batch(BATCH).prefetch(AT)
    ds_te = ds_te.map(prep, num_parallel_calls=AT).cache().batch(BATCH).prefetch(AT)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(IMG, IMG, 3)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"), tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"), tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"), tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(names), activation="softmax"),
    ])
    model.summary()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(ds_tr, validation_data=ds_te, epochs=EPOCHS, verbose=2)

    y_true = np.concatenate([y.numpy() for _, y in ds_te])
    y_pred = model.predict(ds_te, verbose=0).argmax(1)
    print(classification_report(y_true, y_pred, target_names=names, digits=3))

    ep = np.arange(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(ep, hist.history["accuracy"], "o-", color=BLUE, label="train")
    axes[0].plot(ep, hist.history["val_accuracy"], "s--", color=ORANGE, label="test")
    axes[0].set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    axes[1].plot(ep, hist.history["loss"], "o-", color=BLUE, label="train")
    axes[1].plot(ep, hist.history["val_loss"], "s--", color=ORANGE, label="test")
    axes[1].set(xlabel="epoch", ylabel="cross-entropy", title="Loss")
    for a in axes:
        a.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "cnn_training.png"), dpi=160)
    plt.show()

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set(xlabel="predicted", ylabel="true", title="CNN confusion matrix (test)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "cnn_confusion_matrix.png"), dpi=160)
    plt.show()


# %% [markdown]
# ## B. Stacked LSTM for next-day price prediction

# %%
def run_lstm():
    from sklearn.preprocessing import MinMaxScaler

    SEQ, EPOCHS, BATCH = 60, 20, 32
    prices = pd.read_csv(os.path.join(DATA, "stock_close.csv"))[["Close"]].values.astype(np.float32)
    n = len(prices)
    n_windows = n - SEQ
    split = int(0.8 * n_windows)                     # first 80% of windows for training (chronological)
    train_end = split + SEQ                           # last price index that any training target uses

    scaler = MinMaxScaler().fit(prices[:train_end])  # fit on the training period only
    s = scaler.transform(prices).ravel()

    X = np.stack([s[i - SEQ:i] for i in range(SEQ, n)])[..., None]
    y = s[SEQ:]
    X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]
    print(f"{n} prices -> {len(X)} windows | train {len(X_tr)} | test {len(X_te)}")

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(SEQ, 1)),
        tf.keras.layers.LSTM(50, return_sequences=True),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(25, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.summary()
    model.compile(optimizer="adam", loss="mse")
    hist = model.fit(X_tr, y_tr, validation_data=(X_te, y_te), epochs=EPOCHS, batch_size=BATCH, verbose=2)

    pred = scaler.inverse_transform(model.predict(X_te, verbose=0)).ravel()
    true = scaler.inverse_transform(y_te.reshape(-1, 1)).ravel()
    naive = scaler.inverse_transform(X_te[:, -1, :]).ravel()      # tomorrow = today

    for name, p in [("LSTM", pred), ("naive (tomorrow = today)", naive)]:
        err = p - true
        print(f"{name:<26} RMSE {np.sqrt(np.mean(err ** 2)):7.3f} | MAE {np.mean(np.abs(err)):7.3f} | "
              f"MSE {np.mean(err ** 2):8.3f} | mean error (bias) {np.mean(err):+7.3f}")
    up_true = np.sign(true[1:] - true[:-1])
    up_pred = np.sign(pred[1:] - true[:-1])
    print(f"LSTM direction accuracy (up/down vs today): {np.mean(up_pred == up_true) * 100:.1f}%")
    print(f"training-period price range: {prices[:train_end].min():.2f} - {prices[:train_end].max():.2f} | "
          f"test-period range: {true.min():.2f} - {true.max():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 3.9), gridspec_kw={"width_ratios": [2.3, 1]})
    axes[0].plot(true, color="black", lw=1.4, label="actual")
    axes[0].plot(pred, color=ORANGE, lw=1.4, label="LSTM prediction")
    axes[0].set(xlabel="trading day in test period", ylabel="closing price", title="Next-day forecast on the test period")
    axes[0].legend(frameon=False)
    ep = np.arange(1, EPOCHS + 1)
    axes[1].plot(ep, hist.history["loss"], "o-", ms=4, color=BLUE, label="train")
    axes[1].plot(ep, hist.history["val_loss"], "s--", ms=4, color=ORANGE, label="test")
    axes[1].set(yscale="log", xlabel="epoch", ylabel="MSE (scaled prices)", title="Training curve")
    axes[1].legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "lstm_forecast.png"), dpi=160)
    plt.show()

    zoom = slice(len(true) - 60, len(true))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(np.arange(len(true))[zoom], true[zoom], "o-", ms=3, color="black", lw=1.2, label="actual")
    ax.plot(np.arange(len(true))[zoom], pred[zoom], "s-", ms=3, color=ORANGE, lw=1.2, label="LSTM")
    ax.plot(np.arange(len(true))[zoom], naive[zoom], "-", color=GREY, lw=1, label="naive")
    ax.set(xlabel="trading day in test period", ylabel="closing price", title="Last 60 test days")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "lstm_zoom.png"), dpi=160)
    plt.show()


if __name__ == "__main__":
    if "lstm" in PARTS:
        run_lstm()
    if "cnn" in PARTS:
        run_cnn()
