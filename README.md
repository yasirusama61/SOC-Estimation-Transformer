# ⚡ SOC Estimation using Transformer Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20Project-yellow)

This repository contains an initial implementation of a Transformer-based model for **State of Charge (SOC) estimation** in lithium-ion batteries. The project explores the potential of Transformer architectures for time-series forecasting tasks, specifically for SOC prediction.

## 🌟 Project Overview

SOC estimation is critical for **Battery Management Systems (BMS)** to monitor the performance, lifespan, and safety of lithium-ion batteries. Traditional approaches often rely on **LSTM** or other recurrent networks. This project investigates whether **Transformer models**, which have shown impressive results in other time-series applications, can be effective for SOC estimation.

### Key Features
- Utilizes a custom Transformer model for time-series SOC estimation.
- Explores performance across different temperature conditions.
- Includes data processing scripts and model training with evaluation metrics.
- Flexible model architecture for future experimentation and improvements.

## 📊 Dataset

The data used in this project comes from the publicly available **LG 18650HG2 Li-ion Battery Dataset**. The dataset includes normalized features, focusing on the following key measurements:

- **Voltage [V]**: Voltage of the battery cell.
- **Current [A]**: Current drawn from the battery.
- **Temperature [°C]**: Measured ambient temperature.
- **Rolling Average of Voltage**: Moving average for voltage to capture trends.
- **Rolling Average of Current**: Moving average for current to capture trends.

The **target variable** is **SOC (State of Charge)**, which has been normalized.

## 📂 Repository Structure

```plaintext
├── data/                     # Folder containing the dataset (not included)
├── src/                      # Python scripts for data loading, preprocessing, and model training
├── notebooks/                # Jupyter notebooks for exploration and testing
├── results/                  # Model predictions and evaluation plots
├── README.md                 # Project documentation
└── requirements.txt          # Required dependencies
```

### 📊 Data Processing

To effectively utilize the data, we created sequences of length 100, enabling the Transformer model to learn temporal dependencies over a fixed window. This approach captures the operational patterns of SOC over time, allowing the model to predict SOC based on historical data within each sequence.

## 🏗️ Model Architecture

### Transformer Model

The Transformer model consists of several core components:

1. **Input Embedding Layer**: Transforms input features to a higher dimensional space for better representation.
2. **Positional Encoding**: Injects positional information into the input sequence to help the model distinguish the order of observations.
3. **Multi-head Self-Attention Layers**: Captures complex temporal dependencies across the sequence.
4. **Feedforward Layers**: Applies dense layers with ReLU activation and dropout for regularization, reducing the likelihood of overfitting.
5. **Output Layer**: A single neuron layer to output the SOC prediction.

**Key Hyperparameters:**
- **Embedding Dimension**: 64
- **Number of Attention Heads**: 4
- **Feedforward Dimension**: 128
- **Dropout Rate**: 0.4
- **Regularization**: L2 penalty to prevent overfitting

## 🏋️ Training and Evaluation

The model was trained with the following settings:

- **Optimizer**: Adam with gradient clipping for stable training.
- **Learning Rate**: 0.000005
- **Loss Function**: Mean Squared Error (MSE)
- **Batch Size**: 256

### 📉 Results

Despite extensive hyperparameter tuning (adjusting dropout rate, adding L2 regularization, etc.), the Transformer model displayed challenges with overfitting, evidenced by a higher validation loss compared to training loss. This suggests that further optimization, such as additional data preprocessing, data augmentation, or alternative model architectures, may be necessary to improve generalization and model performance.

### Model History and Plotting

We save the training history to visualize and compare training and validation losses over epochs. The current performance indicates room for improvement, and future iterations could involve data augmentation or exploring hybrid Transformer models.

## 🚀 Getting Started

### Requirements

- Python 3.8+
- TensorFlow
- NumPy
- Matplotlib
- Plotly

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Model
1. Preprocess the data: Ensure the data is preprocessed and sequenced.

2. Train the Model:

```bash
   python src/train_transformer.py
```
3. Evaluate and Plot: The script saves evaluation plots in the plots/ directory.

### 🧩 Future Work
- **Model Optimization**: Further tuning of Transformer architecture and hyperparameters.
- **Data Augmentation**: Synthesizing more training data to improve generalization.
- **Hybrid Models**: Investigate hybrid LSTM-Transformer models for enhanced performance.

### 🙋 Contributions

Contributions are welcome. Feel free to fork this repository, raise issues, or submit pull requests.