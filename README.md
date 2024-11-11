# SOC Estimation using Transformer Model

This repository contains an initial implementation of a Transformer-based model for State of Charge (SOC) estimation in lithium-ion batteries. The project explores the potential of Transformer architectures for time-series forecasting tasks, specifically for SOC prediction.

## Project Overview

SOC estimation is critical for battery management systems (BMS) to monitor the performance, lifespan, and safety of lithium-ion batteries. Traditional approaches often rely on LSTM or other recurrent networks. This project investigates whether Transformer models, which have shown impressive results in other time-series applications, can be effective for SOC estimation.

## Dataset

The data used in this project comes from the publicly available **LG 18650HG2 Li-ion Battery Dataset**. The dataset includes normalized features:

- **Voltage [V]**
- **Current [A]**
- **Temperature [°C]**
- **Rolling Average of Voltage**
- **Rolling Average of Current**

The target variable is **SOC (State of Charge)**, which has been normalized.

### Data Processing

To utilize the data effectively, we created sequences of length 100. This approach allows the Transformer model to learn temporal dependencies over a fixed window, simulating the operational patterns of SOC over time.

## Model Architecture

### Transformer Model

The Transformer model consists of:

1. **Input Embedding Layer**: Maps input features to a higher dimensional space.
2. **Positional Encoding**: Adds positional information to the input sequence.
3. **Multi-head Self-Attention Layers**: Captures temporal dependencies.
4. **Feedforward Layers**: Applies dense layers with ReLU activation and dropout for regularization.
5. **Output Layer**: A single neuron layer to predict SOC.

**Hyperparameters:**
- Embedding Dimension: 64
- Number of Attention Heads: 4
- Feedforward Dimension: 128
- Dropout Rate: 0.4
- Regularization: L2 penalty

## Training and Evaluation

The model was trained using the following settings:

- **Optimizer**: Adam with gradient clipping.
- **Learning Rate**: 0.000005
- **Loss Function**: Mean Squared Error
- **Batch Size**: 256

### Results

Despite hyperparameter tuning efforts (dropout, L2 regularization, etc.), the Transformer model displayed overfitting and struggled to generalize effectively. Validation loss was significantly higher than training loss, suggesting further optimization and data augmentation are needed.

### Model History and Plotting

We save the training history to visualize and compare training and validation losses over epochs. The current performance indicates room for improvement, and future iterations could involve data augmentation or exploring hybrid Transformer models.

## File Structure

- `data/`: Folder containing the dataset files (if using external datasets, include download instructions here).
- `notebooks/`: Jupyter notebooks for data exploration and model experimentation.
- `src/`: Core Python scripts for model building, training, and evaluation.
- `plots/`: Folder to save prediction plots and training history.
- `transformer_soc_model.h5`: Saved model after training.

## Getting Started

### Requirements

- Python 3.8+
- TensorFlow
- NumPy
- Matplotlib
- Plotly

Install dependencies:

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

### Future Work
- **Model Optimization**: Further tuning of Transformer architecture and hyperparameters.
- **Data Augmentation**: Synthesizing more training data to improve generalization.
- **Hybrid Models**: Investigate hybrid LSTM-Transformer models for enhanced performance.

### Contributions

Contributions are welcome. Feel free to fork this repository, raise issues, or submit pull requests.