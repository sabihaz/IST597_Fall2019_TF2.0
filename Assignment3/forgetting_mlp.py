"""
CS 599: Foundations of Deep Learning
Assignment: Catastrophic Forgetting in MLP
Author: <Your Name>
Date: <Today's Date>
"""

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Step 1: Reproducibility
# -------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------
# Step 2: Generate Permuted MNIST Tasks
# -------------------------
def generate_permuted_tasks(num_tasks=10, seed=SEED):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    tasks_list = []
    rng = np.random.RandomState(seed)

    for _ in range(num_tasks):
        permutation = rng.permutation(784)
        tasks_list.append((x_train[:, permutation], y_train, x_test[:, permutation], y_test))
    return tasks_list

# -------------------------
# Step 3: Build MLP Model
# -------------------------
def create_mlp(input_dim=784, num_layers=2, units=256, dropout_rate=0.0, num_classes=10):
    model = Sequential()
    for _ in range(num_layers):
        model.add(Dense(units, activation='relu'))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# -------------------------
# Step 4: Metrics Calculation
# -------------------------
def calculate_ACC_BWT(task_matrix):
    num_tasks = task_matrix.shape[0]
    ACC = np.mean(task_matrix[-1, :])
    BWT = np.mean([task_matrix[-1, i] - task_matrix[i, i] for i in range(num_tasks - 1)])
    return ACC, BWT

# -------------------------
# Step 5: Sequential Task Training
# -------------------------
def train_on_tasks(tasks, num_layers=2, units=256, dropout_rate=0.0, optimizer_choice='adam', loss_fn='sparse_categorical_crossentropy'):
    model = create_mlp(num_layers=num_layers, units=units, dropout_rate=dropout_rate)
    model.compile(optimizer=optimizer_choice, loss=loss_fn, metrics=['accuracy'])

    n_tasks = len(tasks)
    task_matrix = np.zeros((n_tasks, n_tasks))
    val_accuracies = []

    for t, (x_tr, y_tr, x_te, y_te) in enumerate(tasks):
        print(f"\n--- Training Task {t+1} ---")
        epochs = 50 if t == 0 else 20
        history = model.fit(x_tr, y_tr, epochs=epochs, validation_split=0.1, verbose=2)
        val_accuracies.append(history.history['val_accuracy'])

        # Evaluate on all seen tasks
        for i in range(t + 1):
            xi_tr, yi_tr, xi_te, yi_te = tasks[i]
            acc = model.evaluate(xi_te, yi_te, verbose=0)[1]
            task_matrix[t, i] = acc
            print(f"Accuracy on Task {i+1}: {acc:.4f}")

    ACC, BWT = calculate_ACC_BWT(task_matrix)
    print("\n=== Final Results ===")
    print(f"ACC = {ACC:.4f}, BWT = {BWT:.4f}")
    return task_matrix, val_accuracies

# -------------------------
# Step 6: Plot Validation Accuracy
# -------------------------
def plot_validation(val_accuracies):
    plt.figure(figsize=(10,6))
    for idx, acc in enumerate(val_accuracies):
        plt.plot(acc, label=f'Task {idx+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy over Epochs for Each Task')
    plt.legend()
    plt.show()

# -------------------------
# Step 7: Main Execution
# -------------------------
if __name__ == "__main__":
    tasks_data = generate_permuted_tasks(num_tasks=10)

    # Experiment parameters
    num_layers = 3
    dropout_rate = 0.3
    optimizer_choice = 'adam'
    loss_fn = 'sparse_categorical_crossentropy'

    task_matrix, val_accs = train_on_tasks(tasks_data, num_layers=num_layers,
                                           dropout_rate=dropout_rate, optimizer_choice=optimizer_choice,
                                           loss_fn=loss_fn)
    plot_validation(val_accs)
