

# x: input
# w: weight of the connection
# b: bias of the neuron
# y: output of the neuron
# y = w * x + b, equation of the line

# improve performance with dropout, batch normalization
#wider network: add units to learn linear relationships
#deeper network: add layers to learn non linear relationships
#overfitting due to the network learning spurious patterns in the training data: add dropout
#correct training that is slow or unstable: add batch normalization
#Models with batchnorm tend to need fewer epochs to complete training. Moreover, batchnorm can also fix various problems that can cause the training to get "stuck".

#classification: 
# * use sigmoid activation function for binary classification
# * softmax activation function for multi-class classification
# classification loss: cross-entropy loss rather than accuracy (soft changes easier to optimize)
# classification metric: accuracy, precision, recall, F1-score, ROC-AUC


from tensorflow import keras
from tensorflow.keras import layers, callbacks
import pandas as pd

def train_model(X_train, y_train, X_valid, y_valid, input_shape, epochs=10, batch_size=256):
    # Early stopping callback
    early_stopping = callbacks.EarlyStopping(
        min_delta=0.001,  # minimum amount of change to count as an improvement
        patience=20,  # how many epochs to wait before stopping
        restore_best_weights=True,  # restore the best weights after stopping
    )

    # Model architecture
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=[input_shape]),
        layers.Dropout(rate=0.3),  # apply 30% dropout to the next layer
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dense(512),
        layers.BatchNormalization(),  # batchnorm after dense layer before activation
        layers.Activation('relu'),
        layers.Dense(1, activation='sigmoid'),
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy'],
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],  # put your callbacks in a list
        verbose=0,  # turn off training log
    )

    # Convert training history to DataFrame
    history_df = pd.DataFrame(history.history)
    
    # Plot training history starting from epoch 5
    history_df.loc[5:, ['loss', 'val_loss']].plot()
    history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()

    # Print best validation results
    print(("Best Validation Loss: {:0.4f}" +\
          "\nBest Validation Accuracy: {:0.4f}")\
          .format(history_df['val_loss'].min(), 
                  history_df['val_binary_accuracy'].max()))

    return model, history_df


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.output = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.3)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.batchnorm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.batchnorm2(self.fc2(x)))
        x = self.relu(self.batchnorm3(self.fc3(x)))
        x = self.sigmoid(self.output(x))
        return x

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            model.load_state_dict(self.best_model)
            return True
        return False

def train_model(X_train, y_train, X_valid, y_valid, input_size, epochs=10, batch_size=256, learning_rate=0.001):
    # Convert data to PyTorch tensors
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    valid_data = TensorDataset(torch.tensor(X_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.float32))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, optimizer, and early stopping
    model = SimpleNN(input_size=input_size)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=20, min_delta=0.001)

    # Track history for plotting
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in valid_loader:
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        val_loss /= len(valid_loader.dataset)
        val_acc = correct / total

        history['loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(epoch_acc)
        history['val_accuracy'].append(val_acc)

        print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc:.4f}')

        if early_stopping(val_loss, model):
            print("Early stopping")
            break

    history_df = pd.DataFrame(history)

    # Plot loss and accuracy
    history_df.loc[5:, ['loss', 'val_loss']].plot()
    history_df.loc[5:, ['accuracy', 'val_accuracy']].plot()

    print(("Best Validation Loss: {:0.4f}" +\
          "\nBest Validation Accuracy: {:0.4f}")\
          .format(history_df['val_loss'].min(), 
                  history_df['val_accuracy'].max()))

    return model, history_df
