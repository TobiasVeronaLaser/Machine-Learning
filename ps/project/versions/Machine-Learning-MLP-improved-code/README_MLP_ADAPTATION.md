# MLP adaptation for the Sign Language project

This version adapts the course notebook section **Multiclass Classification with PyTorch (Addendum)** to the project dataset.

Main changes compared to the MNIST addendum:

- MNIST is replaced by a custom `SignLanguageDataset`.
- The input size is `64 * 64 = 4096` because images are resized to 64x64 grayscale.
- The output size is `36` because the labels are digits 0-9 and letters a-z.
- The loss is `nn.CrossEntropyLoss(weight=class_weights_tensor)` because this is single-label multiclass classification with imbalanced classes.
- Class weights are computed from the training split and penalize mistakes on rare classes more strongly.
- The model returns logits, so no softmax is used inside the network.
- A stratified train/validation/test split is used because the classes are imbalanced.
- The training transform includes `RandomRotation(10)` and a small `RandomAffine` augmentation.
- The notebook includes a linear baseline and an MLP.
- The MLP now uses `HIDDEN_SIZE = 512`.
- The MLP optimizer is `Adam` instead of SGD.
- `train_model` implements early stopping based on validation loss and restores the best model state before final evaluation.
- The final saved MLP checkpoint is the best validation-loss model, not necessarily the last epoch.

Copy these files into `Machine-Learning/ps/project/`. The dataset folder should remain at:

```text
Machine-Learning/ps/project/sign_lang_train/sign_lang_train/
```
