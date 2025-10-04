# Deepfake Face Detector 🕵️‍♂️🧠

[![TensorFlow](https://img.shields.io/badge/Made%20with-TensorFlow-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue?logo=kaggle)](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A binary image classifier that distinguishes **real** vs **fake** faces using EfficientNetB0.  
Built with TensorFlow/Keras and trained on the Kaggle dataset:  
➡️ [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

---

## 🚀 Quick Start (Google Colab)

1. Open `notebooks/deepfake_training.ipynb` in Google Colab.
2. Set **Runtime → GPU**.
3. Upload your `kaggle.json` API key (from [Kaggle Account Settings](https://www.kaggle.com/account)).
4. Run all cells to:
   - Install dependencies
   - Download the dataset
   - Train EfficientNetB0
   - Save training curves, classification report, and confusion matrix

---

