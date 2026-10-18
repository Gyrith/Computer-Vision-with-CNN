# Computer Vision with CNN

Face recognition on the Olivetti faces dataset (40 people, 10 grayscale 64x64 images each) using convolutional neural networks in Keras.

This repository contains a four-part lab that walks through the full modeling process: preparing the data, building a baseline CNN, experimenting with the convolution window size, and optimizing a final model with regularization.

## Parts

| Part | Notebook | Description |
|------|----------|-------------|
| 1 | `Libraries & Preprocessing.ipynb` | Libraries, data loading, sample image, 50/50 stratified train/test split (5 training and 5 test images per person), one-hot encoding, reshaping to `(n, 64, 64, 1)` |
| 2 | `Computer Vision Model 1.ipynb` | Model 1: 16 filters, 3x3 window, same padding, max pooling, softmax output. Adam, 20 epochs, batch size 10. Test loss/accuracy and feature maps |
| 3 | `Model 2 Implementation.ipynb` | Model 2: same as Model 1 with a 9x9 window. Test loss/accuracy and feature maps for comparison |
| 4 | `Model Comparison & Optimization.ipynb` | Comparison of Models 1 and 2, then an optimized Model 3 (two conv blocks, dropout, L2 regularization, early stopping) and a written reflection |

## Dataset

[Olivetti faces](https://scikit-learn.org/stable/datasets/real_world.html#the-olivetti-faces-dataset), loaded with `sklearn.datasets.fetch_olivetti_faces`. It is downloaded automatically the first time you run a notebook.

Run the notebooks in order (Part 1 to Part 4). Each notebook includes its own data-loading cell, and every split uses `random_state=42`, so the train/test sets are identical across notebooks.

## Tech stack

Python, TensorFlow/Keras, scikit-learn, NumPy, Matplotlib, Jupyter