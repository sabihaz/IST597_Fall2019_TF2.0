#!/usr/bin/env python3
# log_reg.py -- Logistic regression on Fashion-MNIST using TF2 eager (no Keras model).
# Features:
#  - manual training loop with tf.GradientTape
#  - choice of optimizer: sgd / adam / rmsprop
#  - train/val split, batch-size, epochs, learning-rate
#  - plot images, loss/accuracy curves, visualize weights
#  - compare with sklearn RandomForest and SVM
#  - t-SNE and k-means clustering on final weight vectors
#  - CPU vs GPU timing per epoch (use --device '/CPU:0' or '/GPU:0')
#
# Example runs:
# python log_reg.py --epochs 20 --optimizer adam --lr 0.001 --batch_size 128 --device /CPU:0
# python log_reg.py --epochs 20 --optimizer sgd --lr 0.1 --batch_size 256 --device /GPU:0

import os
import time
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import joblib
import csv

os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)

# -------------------------
# Utilities
# -------------------------
def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def prepare_dataset(split_ratio=0.9, batch_size=128, seed=42):
    # Load Fashion MNIST via tensorflow_datasets
    ds_train, ds_info = tfds.load('fashion_mnist', split='train', as_supervised=True, with_info=True)
    ds_test = tfds.load('fashion_mnist', split='test', as_supervised=True)

    # Convert to NumPy arrays (small dataset so OK)
    X_train = []
    y_train = []
    for img, label in tfds.as_numpy(ds_train):
        X_train.append(img)
        y_train.append(label)
    X_train = np.stack(X_train).astype(np.float32) / 255.0  # shape (60000,28,28)
    y_train = np.array(y_train).astype(np.int32)

    X_test = []
    y_test = []
    for img, label in tfds.as_numpy(ds_test):
        X_test.append(img)
        y_test.append(label)
    X_test = np.stack(X_test).astype(np.float32) / 255.0
    y_test = np.array(y_test).astype(np.int32)

    # shuffle train
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    # split train/val
    n_train = int(split_ratio * len(X_train))
    X_tr, X_val = X_train[:n_train], X_train[n_train:]
    y_tr, y_val = y_train[:n_train], y_train[n_train:]

    # flatten images for logistic regression
    X_tr_flat = X_tr.reshape((X_tr.shape[0], -1))
    X_val_flat = X_val.reshape((X_val.shape[0], -1))
    X_test_flat = X_test.reshape((X_test.shape[0], -1))

    # create tf.data
    train_ds = tf.data.Dataset.from_tensor_slices((X_tr_flat, y_tr)).shuffle(10000, seed=seed).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val_flat, y_val)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test_flat, y_test)).batch(batch_size)

    return train_ds, val_ds, test_ds, (X_tr_flat, y_tr, X_val_flat, y_val, X_test_flat, y_test), ds_info

def plot_images_grid(X, y, classes, fname="plots/sample_images.png", n=25):
    plt.figure(figsize=(6,6))
    idx = np.random.choice(len(X), n, replace=False)
    for i, k in enumerate(idx):
        plt.subplot(5,5,i+1)
        plt.imshow(X[k].reshape(28,28), cmap='gray')
        plt.title(classes[y[k]])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def plot_curves(history, fname_prefix="plots/logreg"):
    plt.figure()
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.title('Loss vs Epoch')
    plt.savefig(f"{fname_prefix}_loss.png")
    plt.close()

    plt.figure()
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.title('Accuracy vs Epoch')
    plt.savefig(f"{fname_prefix}_acc.png")
    plt.close()

def plot_weights(W, fname="plots/weights.png", classes=None):
    # W shape (num_features, num_classes)
    num_classes = W.shape[1]
    ncols = 5
    nrows = int(np.ceil(num_classes / ncols))
    plt.figure(figsize=(ncols*2, nrows*2))
    for i in range(num_classes):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(W[:, i].reshape(28,28), cmap='seismic', vmin=-np.max(np.abs(W)), vmax=np.max(np.abs(W)))
        title = classes[i] if classes is not None else str(i)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

# -------------------------
# Logistic Regression model (no Keras)
# -------------------------
class LogisticRegressionTF:
    def __init__(self, input_dim, num_classes, W_init=0.0, b_init=0.0):
        # W: shape (input_dim, num_classes)
        self.W = tf.Variable(tf.random.normal([input_dim, num_classes], stddev=0.01), dtype=tf.float32)
        self.b = tf.Variable(tf.zeros([num_classes], dtype=tf.float32))
        # store dims
        self.input_dim = input_dim
        self.num_classes = num_classes

    def logits(self, X):  # X is (batch, input_dim)
        return tf.matmul(X, self.W) + self.b  # (batch, num_classes)

    def predict_proba(self, X):
        return tf.nn.softmax(self.logits(X), axis=-1)

    def predict(self, X):
        return tf.argmax(self.predict_proba(X), axis=-1)

    def loss(self, X, y):
        logits = self.logits(X)
        y_onehot = tf.one_hot(y, depth=self.num_classes)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=logits))
        return loss

# -------------------------
# Training loop
# -------------------------
def train_loop(model, train_ds, val_ds, optimizer, epochs=20, device="/CPU:0", seed=42):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    times = []
    with tf.device(device):
        for epoch in range(1, epochs+1):
            t0 = time.time()
            # train
            train_losses = []
            train_correct = 0
            train_total = 0
            for X_batch, y_batch in train_ds:
                X_batch = tf.cast(X_batch, tf.float32)
                y_batch = tf.cast(y_batch, tf.int32)
                with tf.GradientTape() as tape:
                    loss_val = model.loss(X_batch, y_batch)
                grads = tape.gradient(loss_val, [model.W, model.b])
                optimizer.apply_gradients(zip(grads, [model.W, model.b]))
                train_losses.append(loss_val.numpy())
                preds = tf.argmax(model.predict_proba(X_batch), axis=1).numpy()
                train_correct += np.sum(preds == y_batch.numpy())
                train_total += len(y_batch)
            train_loss = float(np.mean(train_losses))
            train_acc = float(train_correct) / train_total

            # val
            val_losses = []
            val_correct = 0
            val_total = 0
            for X_batch, y_batch in val_ds:
                X_batch = tf.cast(X_batch, tf.float32)
                y_batch = tf.cast(y_batch, tf.int32)
                loss_val = model.loss(X_batch, y_batch)
                val_losses.append(loss_val.numpy())
                preds = tf.argmax(model.predict_proba(X_batch), axis=1).numpy()
                val_correct += np.sum(preds == y_batch.numpy())
                val_total += len(y_batch)
            val_loss = float(np.mean(val_losses))
            val_acc = float(val_correct) / val_total

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            times.append(time.time() - t0)

            print(f"Epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_acc={train_acc*100:.2f}% val_acc={val_acc*100:.2f}% time={times[-1]:.3f}s")

    return history, times

# -------------------------
# Helper to compute final test accuracy
# -------------------------
def evaluate_model(model, test_ds):
    correct = 0; total = 0
    for X_batch, y_batch in test_ds:
        preds = tf.argmax(model.predict_proba(tf.cast(X_batch, tf.float32)), axis=1).numpy()
        correct += np.sum(preds == y_batch.numpy())
        total += len(y_batch)
    return float(correct) / total

# -------------------------
# Main experiment orchestration
# -------------------------
def main(args):
    set_seed(args.seed)
    print("Preparing dataset...")
    train_ds, val_ds, test_ds, arrays, ds_info = prepare_dataset(split_ratio=args.train_split, batch_size=args.batch_size, seed=args.seed)
    classes = ds_info.features['label'].names
    X_tr_flat, y_tr, X_val_flat, y_val, X_test_flat, y_test = arrays

    # show sample images
    plot_images_grid(X_tr_flat, y_tr, classes, fname="plots/fmnist_samples.png", n=25)

    # build model
    input_dim = X_tr_flat.shape[1]
    num_classes = len(classes)
    model = LogisticRegressionTF(input_dim, num_classes)

    # choose optimizer
    if args.optimizer.lower() == 'sgd':
        optimizer = tf.optimizers.SGD(learning_rate=args.lr)
    elif args.optimizer.lower() == 'adam':
        optimizer = tf.optimizers.Adam(learning_rate=args.lr)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer = tf.optimizers.RMSprop(learning_rate=args.lr)
    else:
        raise ValueError("Unsupported optimizer")

    # train
    print("Training model...")
    history, times = train_loop(model, train_ds, val_ds, optimizer, epochs=args.epochs, device=args.device, seed=args.seed)

    # save history plots
    plot_curves(history, fname_prefix="plots/logreg")

    # evaluate on test set
    test_acc = evaluate_model(model, test_ds)
    print(f"Test accuracy (TF logistic): {test_acc*100:.2f}%")

    # visualize weights
    W_np = model.W.numpy()  # shape (784,10)
    plot_weights(W_np, fname="plots/logreg_weights.png", classes=classes)

    # compare with sklearn models (train on flattened X_tr_flat and X_val_flat concat, evaluate on test set)
    print("Training sklearn baselines (this may take ~1-2 minutes)...")
    X_train_skl = np.vstack([X_tr_flat, X_val_flat])
    y_train_skl = np.hstack([y_tr, y_val])
    rf = RandomForestClassifier(n_estimators=100, random_state=args.seed, n_jobs=-1)
    rf.fit(X_train_skl, y_train_skl)
    rf_preds = rf.predict(X_test_flat)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"RandomForest accuracy: {rf_acc*100:.2f}%")

    svm = SVC(kernel='rbf', gamma='scale', random_state=args.seed)
    svm.fit(X_train_skl, y_train_skl)
    svm_preds = svm.predict(X_test_flat)
    svm_acc = accuracy_score(y_test, svm_preds)
    print(f"SVM accuracy: {svm_acc*100:.2f}%")

    # t-SNE on class weight vectors (use class weight vectors = W.T (10 vectors of 784 dims))
    print("Running t-SNE and k-means on weight vectors...")
    w_vectors = W_np.T  # shape (10, 784)
    tsne = TSNE(n_components=2, random_state=args.seed, init='pca', perplexity=3)
    w2d = tsne.fit_transform(w_vectors)
    kmeans = KMeans(n_clusters=10, random_state=args.seed).fit(w2d)
    plt.figure(figsize=(6,5))
    plt.scatter(w2d[:,0], w2d[:,1], c=kmeans.labels_, cmap='tab10', s=100)
    for i, txt in enumerate(classes):
        plt.annotate(txt, (w2d[i,0], w2d[i,1]))
    plt.title("t-SNE of class weight vectors (logistic)")
    plt.savefig("plots/logreg_weights_tsne.png")
    plt.close()

    # Save numeric summary
    summary = {
        'test_acc_tf_logistic': float(test_acc),
        'rf_acc': float(rf_acc),
        'svm_acc': float(svm_acc),
        'avg_epoch_time': float(np.mean(times)),
        'epochs': int(args.epochs),
        'optimizer': args.optimizer,
        'lr': args.lr,
        'batch_size': args.batch_size
    }
    summary_file = "results/logreg_summary.csv"
    with open(summary_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for k, v in summary.items():
            writer.writerow([k, v])
    print(f"Summary saved to {summary_file}")
    print("Plots saved under plots/*.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="adam", choices=['sgd', 'adam', 'rmsprop'])
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=sum([ord(c) for c in "Noorus"]))  # unique seed derived from first name
    parser.add_argument("--device", type=str, default="/CPU:0")  # or "/GPU:0"
    args = parser.parse_args()
    main(args)
