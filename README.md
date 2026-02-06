# NCD_NS

A neuro-symbolic learning-based cognitive diagnosis model implementation that combines logical rule constraints for assessing student knowledge mastery.

## Project Structure

```
NCD_NS
├── main.py                 # Training and evaluation main program
├── models.py              # NeuroSymbolicCD model implementation
├── data_utils.py          # Data loading and preprocessing utilities
├── divide_data.py         # Dataset splitting script
├── data/                  # Data directory
│   ├── Assist/           # ASSISTments dataset
│   ├── Math/             # Math dataset
│   └── Junyi/            # Junyi dataset
└── result/               # Model results saving directory
    ├── best_model.pth
    └── full_model_and_results.pth
```

## Features

- **Neuro-Symbolic Learning**: Integrating deep learning with logical rules
- **Multiple Rule Constraints**:
  - Prerequisite rules
  - Similarity pairing
  - Compositional rules
  - Smoothness and monotonicity constraints
- **Multi-Dataset Support**: Compatible with ASSISTments, Math datasets, etc.
- **Early Stopping Mechanism**: Prevents overfitting
- **Comprehensive Evaluation Metrics**: Accuracy, AUC, RMSE

## Quick Start

### 1. Data Preparation

Modify `data_dir` to your data directory (e.g., `'data/Assist'`)

### 2. Train Model

```bash
python main.py
```

### 3. Key Parameter Configuration

Edit hyperparameters in main.py:

```python
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
LAMBDA_LOGIC = 0.05      # Logical rule loss weight
PATIENCE = 3             # Early stopping patience
```

## Data Format

### Training Set (train_set.json)

```json
[
  {
    "user_id": 1,
    "exer_id": 7,
    "score": 1,
    "knowledge_code": [1, 2]
  }
]
```

### Validation/Test Set (val_set.json / test_set.json)

```json
[
  {
    "user_id": 1,
    "log_num": 4,
    "logs": [
      {
        "exer_id": 7,
        "score": 1,
        "knowledge_code": [1, 2]
      }
    ]
  }
]
```

## Output Results

After training completes, results are saved in the `result/` directory:

- `best_model.pth`: Best model weights
- `full_model_and_results.pth`: Complete model + configuration + evaluation results

## Performance Metrics

Model outputs on test set:

- **Accuracy**: Binary classification accuracy
- **AUC**: Area under ROC curve
- **RMSE**: Root mean square error