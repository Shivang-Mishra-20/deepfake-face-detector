import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from preprocess import get_data_generators

# ✅ Correct validation path
TRAIN_DIR = "/content/data/real_vs_fake/real-vs-fake/train"
VAL_DIR   = "/content/data/real_vs_fake/real-vs-fake/valid"


IMG_SIZE = (224,224)
BATCH_SIZE = 16

MODEL_PATH = 'models/deepfake_efficientnetb0.h5'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    # ✅ Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Model file not found at {MODEL_PATH}")
        print("👉 Please run `python src\\train.py` first to train and save the model.")
        return

    # ✅ Check class indices file exists
    if not os.path.exists('models/class_indices.json'):
        print("⚠️ class_indices.json not found in models/")
        print("👉 This file is saved automatically after training. Please retrain.")
        return

    with open('models/class_indices.json') as f:
        class_indices = json.load(f)

    print("📂 Loading validation data...")
    _, val_gen = get_data_generators(TRAIN_DIR, VAL_DIR, IMG_SIZE, BATCH_SIZE)

    print("⚡ Loading model...")
    model = load_model(MODEL_PATH)

    print("🚀 Running predictions...")
    preds = model.predict(val_gen, verbose=1)
    y_pred = (preds > 0.5).astype(int).ravel()
    y_true = val_gen.classes

    report = classification_report(y_true, y_pred, target_names=class_indices.keys())
    print("\n📊 Classification Report:\n", report)

    with open(os.path.join(RESULTS_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))

    print("✅ Results saved in:", RESULTS_DIR)

if __name__ == '__main__':
    main()
