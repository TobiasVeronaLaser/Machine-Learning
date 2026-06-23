# MLP adaptation for the Sign Language project

This version adapts the course notebook section **Multiclass Classification with PyTorch (Addendum)** to the project dataset.

Main changes compared to the MNIST addendum:

- MNIST is replaced by a custom `SignLanguageDataset`.
- The input size is `64 * 64 = 4096` because images are resized to 64x64 grayscale.
- The output size is `36` because the labels are digits 0-9 and letters a-z.
- The loss is `nn.CrossEntropyLoss()` because this is single-label multiclass classification.
- The model returns logits, so no softmax is used inside the network.
- A stratified train/validation/test split is used because the classes are imbalanced.
- The notebook includes a linear baseline and an MLP.
- The final MLP is saved and loaded again for reproducible evaluation.

Copy these files into `Machine-Learning/ps/project/`. The dataset folder should remain at:

```text
Machine-Learning/ps/project/sign_lang_train/sign_lang_train/
```
