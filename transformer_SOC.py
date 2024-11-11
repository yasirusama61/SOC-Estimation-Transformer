#!/usr/bin/env python
# coding: utf-8

"""
Battery SOC Estimation using Transformer Model

Author: Usama Yasir Khan
Date: 2024-11-11

Description:
This script performs State of Charge (SOC) estimation for battery data using a Transformer model. It includes functions
for loading data, creating sequences, building the Transformer model, and training the model. The model can be evaluated
on different temperature conditions and the results visualized.

Dependencies:
- TensorFlow
- NumPy
- Pandas
- Scipy
- Plotly
- Matplotlib
"""

# Import libraries
import scipy.io
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers, models, regularizers
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Load and preprocess data
def load_mat_file(file_path, input_columns, target_column):
    """Loads data from a .mat file and returns input and target dataframes."""
    mat_file = scipy.io.loadmat(file_path)
    X = mat_file['X'].T
    Y = mat_file['Y'].T
    df_X = pd.DataFrame(X, columns=input_columns)
    df_Y = pd.DataFrame(Y, columns=[target_column])
    return pd.concat([df_X, df_Y], axis=1)

def create_sequences(X, y, timesteps):
    """Creates sequences from the data for training the model."""
    X_seq, y_seq = [], []
    for i in range(len(X) - timesteps):
        X_seq.append(X[i:i + timesteps])
        y_seq.append(y[i + timesteps])
    return np.array(X_seq), np.array(y_seq)

# Positional encoding for Transformer model
def positional_encoding(position, d_model):
    """Generates positional encoding for Transformer model."""
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads, dtype=tf.float32)

# Transformer model components
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    @tf.function
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self._split_heads(self.query_dense(inputs), batch_size)
        key = self._split_heads(self.key_dense(inputs), batch_size)
        value = self._split_heads(self.value_dense(inputs), batch_size)
        scale = tf.math.sqrt(tf.cast(self.projection_dim, tf.float32))
        attention_scores = tf.nn.softmax(tf.matmul(query, key, transpose_b=True) / scale, axis=-1)
        out = tf.matmul(attention_scores, value)
        out = self._combine_heads(out, batch_size)
        return self.combine_heads(out)

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    @tf.function
    def call(self, inputs, training=False):
        attn_output = self.attention(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Build Transformer model
def build_transformer_model(seq_len, num_features, embed_dim=64, num_heads=2, ff_dim=64, num_layers=1):
    inputs = layers.Input(shape=(seq_len, num_features))
    x = layers.Dense(embed_dim)(inputs)
    x += positional_encoding(seq_len, embed_dim)
    for _ in range(num_layers):
        x = TransformerEncoder(embed_dim, num_heads, ff_dim)(x, training=False)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(1)(x)
    return models.Model(inputs=inputs, outputs=outputs)

# Load and prepare training and validation data
train_file = 'TRAIN_LGHG2@n10degC_to_25degC_Norm_5Inputs.mat'
validation_file = '01_TEST_LGHG2@n10degC_Norm_(05_Inputs).mat'
input_columns = ['Voltage', 'Current', 'Temperature', 'Avg_voltage', 'Avg_current']
target_column = 'SOC'

# Load training and validation data
df_train = load_mat_file(train_file, input_columns, target_column)
df_val = load_mat_file(validation_file, input_columns, target_column)

# Create sequences for model input
timesteps = 100
X_train_seq, y_train_seq = create_sequences(df_train[input_columns].values, df_train[target_column].values, timesteps)
X_val_seq, y_val_seq = create_sequences(df_val[input_columns].values, df_val[target_column].values, timesteps)

# Train the Transformer model
transformer_model = build_transformer_model(seq_len=timesteps, num_features=len(input_columns))
transformer_model.compile(optimizer=Adam(learning_rate=5e-7, clipnorm=1.0), loss="mean_squared_error", metrics=["mae"])
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-8)

history = transformer_model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=50,
    batch_size=256,
    callbacks=[early_stopping, reduce_lr]
)

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

plot_training_history(history)

# Save the model
transformer_model.save('transformer_soc_model.h5')
print("Model saved as transformer_soc_model.h5")

# Evaluation on test data
test_file_path = 'Test/02_TEST_LGHG2@0degC_Norm_(05_Inputs).mat'
X_test_seq, y_test_seq = create_sequences(load_mat_file(test_file_path, input_columns, target_column)[input_columns].values, load_mat_file(test_file_path, input_columns, target_column)[target_column].values, timesteps)
test_loss, test_mae = transformer_model.evaluate(X_test_seq, y_test_seq)
print(f"Test MAE: {test_mae}")

# Plot actual vs predicted
y_pred_val = transformer_model.predict(X_test_seq)
plt.figure(figsize=(12, 6))
plt.plot(y_pred_val, label='Predicted')
plt.plot(y_test_seq, label='Actual')
plt.title('Actual vs Predicted SOC')
plt.xlabel('Index')
plt.ylabel('SOC')
plt.legend()
plt.show()
