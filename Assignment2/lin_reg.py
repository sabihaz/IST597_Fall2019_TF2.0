# lin_reg.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# -----------------------------
# Create folders for results
# -----------------------------
os.makedirs("plots", exist_ok=True)
os.makedirs("loss_curves", exist_ok=True)

# -----------------------------
# Utility Functions
# -----------------------------
def generate_data(n_samples=10000, noise_type='gaussian', noise_level=1.0, seed=42):
    np.random.seed(seed)
    x = np.random.rand(n_samples, 1) * 10
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level, size=(n_samples, 1))
    elif noise_type == 'uniform':
        noise = np.random.uniform(-noise_level, noise_level, size=(n_samples, 1))
    elif noise_type == 'laplace':
        noise = np.random.laplace(0, noise_level, size=(n_samples, 1))
    else:
        noise = 0
    y = 3 * x + 2 + noise
    return x.astype(np.float32), y.astype(np.float32)

def plot_results(x, y, W, b, title="Linear Fit", fname=None):
    plt.figure(figsize=(6,4))
    plt.scatter(x, y, s=5, alpha=0.3)
    plt.plot(x, W * x + b, color='red', linewidth=2)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    if fname:
        plt.savefig(fname)
    plt.close()

def plot_loss(losses_dict, fname=None):
    plt.figure(figsize=(6,4))
    for key in losses_dict:
        plt.plot(losses_dict[key], label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    if fname:
        plt.savefig(fname)
    plt.close()

# -----------------------------
# Model and Loss Functions
# -----------------------------
class LinearRegression:
    def __init__(self, W_init=0.0, b_init=0.0):
        self.W = tf.Variable([[W_init]], dtype=tf.float32)
        self.b = tf.Variable([[b_init]], dtype=tf.float32)

    def predict(self, x):
        return tf.matmul(x, self.W) + self.b

    def loss(self, y_true, y_pred, loss_type='mse', alpha=0.5):
        if loss_type == 'mse':
            return tf.reduce_mean(tf.square(y_true - y_pred))
        elif loss_type == 'mae':
            return tf.reduce_mean(tf.abs(y_true - y_pred))
        elif loss_type == 'hybrid':
            return alpha * tf.reduce_mean(tf.abs(y_true - y_pred)) + (1-alpha) * tf.reduce_mean(tf.square(y_true - y_pred))
        else:
            raise ValueError("Invalid loss_type")

# -----------------------------
# Training Function
# -----------------------------
def train(model, x_train, y_train, epochs=100, lr=0.01, patience=5,
          loss_type='mse', noise_in_weights=0.0, noise_in_lr=0.0, seed=42, device="/GPU:0"):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    optimizer = tf.optimizers.SGD(lr)
    losses = []
    best_loss = float('inf')
    wait = 0
    times_per_epoch = []

    with tf.device(device):
        for epoch in range(epochs):
            start_time = time.time()
            with tf.GradientTape() as tape:
                y_pred = model.predict(x_train)
                loss_val = model.loss(y_train, y_pred, loss_type=loss_type)
            
            gradients = tape.gradient(loss_val, [model.W, model.b])

            # Add noise to gradients (simulate noisy weights)
            if noise_in_weights > 0:
                gradients = [g + tf.random.normal(g.shape, stddev=noise_in_weights) for g in gradients]
            
            # Add noise to learning rate
            lr_noisy = lr
            if noise_in_lr > 0:
                lr_noisy += np.random.normal(0, noise_in_lr)
                optimizer.learning_rate = max(lr_noisy, 1e-6)

            optimizer.apply_gradients(zip(gradients, [model.W, model.b]))

            losses.append(loss_val.numpy())

            # Patience scheduling
            if loss_val < best_loss - 1e-6:
                best_loss = loss_val
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    lr *= 0.5
                    optimizer.learning_rate = lr
                    wait = 0

            times_per_epoch.append(time.time() - start_time)

    return losses, times_per_epoch

# -----------------------------
# Experiments
# -----------------------------
if __name__ == "__main__":
    # Seed from first name
    seed = sum([ord(c) for c in "Noorus"])
    
    noise_types = ['gaussian', 'uniform', 'laplace']
    noise_levels = [0.5, 1.0, 2.0]
    loss_types = ['mse', 'mae', 'hybrid']
    
    results_summary = []

    for noise_type in noise_types:
        for noise_level in noise_levels:
            x_train, y_train = generate_data(noise_type=noise_type, noise_level=noise_level, seed=seed)
            losses_dict = {}
            times_dict = {}

            for loss_type in loss_types:
                model = LinearRegression(W_init=0.0, b_init=0.0)
                losses, times_per_epoch = train(
                    model, x_train, y_train,
                    epochs=100, lr=0.01, patience=10,
                    loss_type=loss_type, noise_in_weights=0.1, noise_in_lr=0.001, seed=seed
                )
                losses_dict[loss_type] = losses
                times_dict[loss_type] = times_per_epoch

                # Save plots
                fname_plot = f"plots/fit_{loss_type}_{noise_type}_{noise_level}.png"
                plot_results(x_train, y_train, model.W.numpy()[0][0], model.b.numpy()[0][0],
                             title=f"{loss_type} | {noise_type} noise {noise_level}", fname=fname_plot)
            
            # Save loss curves
            fname_loss = f"loss_curves/loss_{noise_type}_{noise_level}.png"
            plot_loss(losses_dict, fname=fname_loss)

            # Store summary
            results_summary.append({
                "noise_type": noise_type,
                "noise_level": noise_level,
                "loss_final": {lt: losses_dict[lt][-1] for lt in loss_types},
                "avg_epoch_time": {lt: np.mean(times_dict[lt]) for lt in loss_types}
            })
    
    # Print summary
    for res in results_summary:
        print(res)
