# ML Group Project Submission

This project contains a reproducible local training/evaluation pipeline for the sign-language image classification task.

## Folder structure

```text
PROJECT_SUBMISSION_FINAL/
  README.md
  requirements.txt
  src/
    __init__.py
    dataset.py
    models.py
    train_utils.py
  notebooks/
    01_mlp_baseline.ipynb
    02_mlp_improved.ipynb
    03_evaluation.ipynb
  saved_models/
  plots/
```

## Dataset location

The notebooks expect the dataset here:

```text
PROJECT_SUBMISSION_FINAL/sign_lang_train/sign_lang_train/
  labels.csv
  image files
```

If your dataset is stored elsewhere, edit this line in each notebook:

```python
DATA_ROOT = PROJECT_ROOT / "sign_lang_train" / "sign_lang_train"
```

## How to run

1. Activate your virtual environment.
2. Install requirements:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Run the notebooks in this order:

```text
notebooks/01_mlp_baseline.ipynb
notebooks/02_mlp_improved.ipynb
notebooks/03_evaluation.ipynb
```

## Methods

### Baseline MLP

The baseline uses a simple MLP on flattened 64x64 grayscale images. It uses a fixed number of epochs, no class weights, no early stopping, and no data augmentation.

### Improved MLP

The improved MLP adds:
- more training epochs
- early stopping based on validation loss
- class-weighted CrossEntropyLoss for class imbalance
- dropout
- light data augmentation

## Metrics

Accuracy is sample-based, so frequent classes influence it more strongly. Macro-F1 computes F1 per class and averages all classes equally. Macro-F1 is therefore useful for imbalanced datasets.

## Future work

A CNN would likely be better suited for image classification, because it can exploit local spatial structure. Due to time and computational constraints, this submission focuses on a reliable Logistic Regression / MLP comparison and mentions CNNs as a possible improvement.

## AI assistance

AI assistance was used for code structuring, debugging support, and documentation wording. The final code was reviewed, adapted, executed, and checked by the authors. The implemented training pipeline, model architecture, evaluation procedure, and reported results can be explained by the authors.