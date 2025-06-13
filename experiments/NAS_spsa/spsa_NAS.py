import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import backend as K
import random
import tensorflow as tf
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

def load_wine_data():

    # Load UCI-Wine Quality dataset from the website 
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')

    # Create the different training, validation and test sets
    X = df.drop("quality", axis=1).values
    y = df["quality"].values
    y = y - y.min()
    num_classes = len(np.unique(y))
    y = to_categorical(y, num_classes)
    X = StandardScaler().fit_transform(X)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes


def mapping_arch(x):

    # Map our continous architecture representation to discrete layer sizes
    x = np.clip(x, 0.0, 1.0)
    l1 = int(32 + x[0] * (1024 - 32))
    l2 = int(32 + x[1] * (1024 - 32))
    l3 = int(x[2] * 512)
    return [l1, l2, l3]

def build_model(arch, num_classes):

    # Build a 3 layer MLP with the current given architecture
    model = Sequential()
    model.add(Dense(arch[0], activation='relu', input_shape=(11,)))
    model.add(Dropout(0.3))
    model.add(Dense(arch[1], activation='relu'))
    model.add(Dropout(0.3))
    if arch[2] > 0:
        model.add(Dense(arch[2], activation='relu'))
        model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(0.001), loss=CategoricalCrossentropy(), metrics=['accuracy'])
    return model

def evaluate_architecture(x, X_train, y_train, X_val, y_val, num_classes):

    # Evaluate the architecture by training the model and returning its loss
    arch = mapping_arch(x)
    model = build_model(arch, num_classes)
    model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=0)
    loss, _ = model.evaluate(X_val, y_val, verbose=0)
    K.clear_session()
    return loss

### Implementation of the SPSA algorithm adapted to our needs ###
def spsa_NAS_(x0, a, X_train, y_train, X_val, y_val, num_classes, c=0.2, alpha=0.602, gamma=0.101, num_iter=25, beta=0.8, plot=True):
    
    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_loss = float('inf')
    grad_avg = np.zeros_like(x)
    losses, best_losses = [], []

    for k in range(1, num_iter + 1):

        # Compute the learning rate (ak) from the magnitude rate a and its decay parameter alpha
        ak = a / (k ** alpha)

        # Compute the perturbation size (ck) from the magnitude rate c and its decay parameter gamma
        ck = max(c / (k ** gamma), 0.05)

        # Random ±1 perturbation vector
        delta = np.random.choice([-1, 1], size=len(x))

        # Evaluate the two points to estimate the gradient 
        x_plus = np.clip(x + ck * delta, 0.0, 1.0)
        x_minus = np.clip(x - ck * delta, 0.0, 1.0)
        y_plus = evaluate_architecture(x_plus, X_train, y_train, X_val, y_val, num_classes)
        y_minus = evaluate_architecture(x_minus, X_train, y_train, X_val, y_val, num_classes)

        # Compute the gradient approximation
        gk = (y_plus - y_minus) / (2.0 * ck * delta)
        gk = np.nan_to_num(gk)

        # Smooth out gradient estimates with momentum
        grad_avg = beta * grad_avg + (1 - beta) * gk

        # Update the architecture parameters
        x = np.clip(x - ak * grad_avg, 0.0, 1.0)

        # Evaluate the model with the new architecture
        current_loss = evaluate_architecture(x, X_train, y_train, X_val, y_val, num_classes)
        losses.append(current_loss)

        # Update the best architecture if the current loss is better
        if current_loss < best_loss:
            best_loss = current_loss
            best_x = x.copy()
        best_losses.append(best_loss)

        print(f"Iter {k:02d}: Loss={current_loss:.4f}, Best Loss={best_loss:.4f}, Arch={mapping_arch(x)}, Best Arch={mapping_arch(best_x)}")

    # Plot the evloution of the current loss and best loss
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(losses, label="Current loss")
        plt.plot(best_losses, label="Best loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("SPSA Accuracy Optimization")
        plt.legend()
        plt.grid(True)
        plt.show()

    return mapping_arch(best_x)

def run_NAS_spsa(X_train, X_val, y_train, y_val, num_classes, x0=[0.5, 0.5, 0.5], a=0.3, num_iter=25, plot=True):

    # Run the SPSA algorithm
    best_arch = spsa_NAS_(x0, a, X_train, y_train, X_val, y_val, num_classes, num_iter=num_iter, plot=plot)

    return best_arch


def final_accuracy(X_train, X_test, y_train, y_test, num_classes, best_arch, epochs=10):

    # Train the model with the best found architecture and evaluate on the test set
    final_model = build_model(best_arch, num_classes)
    final_model.fit(X_train, y_train, epochs=5, batch_size=128, verbose=0)
    _, accuracy= final_model.evaluate(X_test, y_test, verbose=0)

    return accuracy


### Grid search ###

def grid_search(X_train, X_val, y_train, y_val, num_classes):

    # Configurations to test
    layer_nb_units = [32, 64, 128, 256, 512, 1024]  
    last_layer_nb_units = [0, 64, 128, 256]

    best_loss = float('inf')
    worst_loss = float('-inf')
    best_arch = None
    worst_arch = None
    step = 0

    for l1 in layer_nb_units:
        for l2 in layer_nb_units:
            for l3 in last_layer_nb_units :

                # Build the model
                arch = [l1, l2, l3]
                model = build_model(arch, num_classes)

                # Train and evaluate the model
                model.fit(X_train, y_train, epochs=5, batch_size=128, verbose=0)
                loss, _ = model.evaluate(X_val, y_val, verbose=0)
                step += 1
                K.clear_session()
                print(f"Configuration {step:02d}: Loss={loss:.4f}, Arch={arch}")

                # Update best and worst architectures
                if loss < best_loss:
                    best_loss = loss
                    best_arch = arch
                if loss > worst_loss:
                    worst_loss = loss
                    worst_arch = arch

    print(f"\nGrid search best loss: {best_loss:.4f} - Best arch: {best_arch}")
    print(f"\nGrid search worst loss: {worst_loss:.4f} - Worst arch: {worst_arch}")

    return best_arch, worst_arch 

