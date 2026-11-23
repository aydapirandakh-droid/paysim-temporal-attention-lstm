# PaySim Temporal Attention LSTM

## Project Overview

This project presents an improved deep learning approach for financial fraud detection using the PaySim synthetic transaction dataset.

It combines domain-specific feature engineering with a Temporal Attention-based LSTM architecture to identify fraudulent transactions in highly imbalanced financial data.

The project also explores threshold optimization and temporal attention weight analysis.

## Dataset

- **Dataset:** PaySim
- **Source:** [Kaggle - PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **Total Transactions:** 6,362,620
- **Fraudulent Transactions:** 8,213
- **Original Features:** 11
- **Engineered Input Features:** 14
- **Target Variable:** `isFraud`

The dataset contains a highly imbalanced class distribution, with fraudulent transactions representing approximately 0.1291% of all transactions.

## Feature Engineering

Three domain-specific financial features were introduced:

- `errorBalanceOrig`: Measures discrepancies in origin account balances.
- `errorBalanceDest`: Measures discrepancies in destination account balances.
- `amount_to_oldbalanceOrg_ratio`: Represents the transaction amount relative to the origin account's previous balance.

Additional preprocessing steps include:

- Removing transaction identifiers and the `isFlaggedFraud` feature.
- One-hot encoding transaction types.
- Chronological sorting of transactions.
- StandardScaler normalization.

## Temporal Data Preparation

The dataset was divided chronologically:

- Training: 70%
- Validation: 15%
- Testing: 15%

A sliding window approach was used to construct temporal sequences.

- **Window Size:** 5 consecutive transactions
- **Input Shape:** (Batch Size, 5, 14)
- **Batch Size:** 4096

The StandardScaler was fitted exclusively on the training data.

## Model Architecture

### Temporal Attention LSTM

The proposed architecture consists of:

- Two-layer LSTM network.
- Hidden dimension of 64.
- Temporal Attention mechanism.
- Weighted context vector representation.
- Fully connected classification layers.
- ReLU activation and Dropout.

The attention mechanism assigns weights to different positions within each input sequence before generating the final fraud classification output.

## Training Configuration

- **Framework:** PyTorch
- **Optimizer:** AdamW
- **Learning Rate:** 0.001
- **Weight Decay:** 0.0001
- **Training Epochs:** 7
- **Loss Function:** BCEWithLogitsLoss
- **Class Imbalance Strategy:** Calibrated positive class weighting
- **Execution Device:** CUDA GPU

## Threshold Optimization

The decision threshold was optimized on the validation set by maximizing the F1-score.

- **Optimal Threshold:** 0.9385
- **Validation Precision:** 0.9495
- **Validation Recall:** 0.8025
- **Validation F1-score:** 0.8698

## Model Evaluation

Final evaluation was performed on the unseen test set.

| Evaluation Metric | Test Result |
|---|---:|
| PR-AUC (Average Precision) | 0.9368 |
| ROC-AUC | 0.9978 |
| Fraud Precision | 0.9728 |
| Fraud Recall | 0.8473 |
| Fraud F1-score | 0.9057 |
| Accuracy | 0.9993 |

The reported classification metrics use the validation-optimized threshold of 0.9385.

## Temporal Attention Analysis

An additional attention-weight profiling analysis was conducted to examine the distribution of attention across the five temporal positions.

Mean attention weights were calculated separately for fraudulent and legitimate transaction sequences.

This analysis provides insight into how the trained model distributes attention across the input window. Attention weights are used as an auxiliary interpretation method and should not be considered a complete explanation of individual predictions.

## Technologies & Libraries

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- KaggleHub

## Repository Structure

```text
paysim-temporal-attention-lstm/
│
├── README.md
└── paysim.Temporal2.ipynb
```

## How to Run

1. Clone or download this repository.
2. Open the notebook in Google Colab or Jupyter Notebook.
3. Install the required Python libraries.
4. Access the PaySim dataset through KaggleHub.
5. Enable GPU acceleration for faster training.
6. Run the notebook cells sequentially.

## Project Scope

This project is an educational implementation focused on financial fraud detection, temporal deep learning, domain-specific feature engineering, class imbalance handling, and attention-based analysis.

---

**Author:** Ayda Pirandakh
