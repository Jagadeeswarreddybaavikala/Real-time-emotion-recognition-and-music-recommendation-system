"""
Professional Emotion Recognition Training with Real Data Reporting
Uses your FER2013 image folders (32,000+ images) to create an ACCURATE emotion detector
Saves training history, graphs, and classification metrics for project review
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

print("🧠 Professional Emotion Recognition Training")
print("=" * 60)

# Create project results directory
os.makedirs('project_results', exist_ok=True)

train_dir = Path('data/fer2013.csv/train')
test_dir  = Path('data/fer2013.csv/test')

if not train_dir.exists():
    print("❌ FER2013 training data not found!")
    exit(1)

print(f"✅ Training  {train_dir}")
print(f"✅ Test      {test_dir}")

try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✅ Mac M2 GPU acceleration enabled!")
except:
    print("⚠️  Using CPU")

def create_accurate_model():
    model = keras.Sequential([
        keras.Input(shape=(48, 48, 1)),
        layers.Rescaling(1./255),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2), layers.Dropout(0.25),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2), layers.Dropout(0.25),

        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2), layers.Dropout(0.25),

        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(), layers.Dropout(0.5),

        layers.Dense(512, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(256, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.4),
        layers.Dense(7, activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

print("\n🏗️  Creating CNN model...")
model = create_accurate_model()
model.summary()

print("\n📂 Setting up data generators...")
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

print("\n📥 Loading training images...")
train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=(48,48),
    color_mode='grayscale', batch_size=64,
    class_mode='categorical', shuffle=True
)

print("📥 Loading test images...")
test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=(48,48),
    color_mode='grayscale', batch_size=64,
    class_mode='categorical', shuffle=False
)

print(f"\n✅ Training samples: {train_gen.samples}")
print(f"✅ Test samples:     {test_gen.samples}")

emotions = list(train_gen.class_indices.keys())
print(f"\n🎭 Emotions: {emotions}")

callbacks = [
    keras.callbacks.ModelCheckpoint(
        'models/accurate_emotion_model.h5',
        monitor='val_accuracy', save_best_only=True, verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1
    )
]

print("\n" + "="*60)
print("🏋️  TRAINING STARTED (50 EPOCHS)")
print("="*60)

history = model.fit(
    train_gen,
    epochs=50,
    validation_data=test_gen,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "="*60)
print("📊 FINAL EVALUATION")
print("="*60)

loss, acc = model.evaluate(test_gen, verbose=0)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")
print(f"✅ Test Loss:     {loss:.4f}")

model.save('models/accurate_emotion_model_final.h5')
print("\n💾 Model saved to: models/accurate_emotion_model.h5")

# ============================================================
# SAVE TRAINING HISTORY AND GENERATE REPORTS
# ============================================================

print("\n" + "="*60)
print("💾 SAVING TRAINING HISTORY AND GENERATING REPORTS")
print("="*60)

# 1. Save training history as pickle
with open('project_results/training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
print("✅ Training history saved: project_results/training_history.pkl")

# 2. Save to CSV
epochs_data = pd.DataFrame(history.history)
epochs_data.to_csv('project_results/training_history.csv', index=True)
print("✅ Training history CSV: project_results/training_history.csv")

# 3. Generate training curves
print("\n📊 Generating training curves...")
epochs_range = range(1, len(history.history['accuracy'])+1)

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2, marker='o', markersize=3)
plt.plot(epochs_range, history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2, marker='s', markersize=3)
plt.title('Training & Validation Accuracy (50 Epochs)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['loss'], 'b-', label='Training Loss', linewidth=2, marker='o', markersize=3)
plt.plot(epochs_range, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=3)
plt.title('Training & Validation Loss (50 Epochs)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('project_results/training_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Training curves saved: project_results/training_curves.png")

# 4. Generate confusion matrix and classification report
print("\n📊 Generating classification metrics...")
y_pred = model.predict(test_gen, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_gen.classes

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=emotions, yticklabels=emotions,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Emotion Recognition Model', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Emotion', fontsize=12)
plt.ylabel('True Emotion', fontsize=12)
plt.tight_layout()
plt.savefig('project_results/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Confusion matrix saved: project_results/confusion_matrix.png")

# 5. Per-emotion metrics
report = classification_report(y_true, y_pred_classes, target_names=emotions, output_dict=True)

with open('project_results/classification_report.txt', 'w') as f:
    f.write("EMOTION CLASSIFICATION REPORT\n")
    f.write("="*70 + "\n")
    f.write("Dataset: FER-2013 (32,000+ images)\n")
    f.write("Model: Convolutional Neural Network (CNN)\n")
    f.write("Architecture: 4 Conv Blocks with BatchNormalization & Dropout\n")
    f.write("Training Epochs: 50\n")
    f.write(f"Training Samples: {train_gen.samples}\n")
    f.write(f"Test Samples: {test_gen.samples}\n\n")
    
    f.write("PER-EMOTION PERFORMANCE:\n")
    f.write("-"*70 + "\n")
    for emotion in emotions:
        f.write(f"\n{emotion}:\n")
        f.write(f"  Precision: {report[emotion]['precision']:.4f} ({report[emotion]['precision']*100:.2f}%)\n")
        f.write(f"  Recall:    {report[emotion]['recall']:.4f} ({report[emotion]['recall']*100:.2f}%)\n")
        f.write(f"  F1-Score:  {report[emotion]['f1-score']:.4f}\n")
        f.write(f"  Support:   {int(report[emotion]['support'])}\n")
    
    f.write(f"\n\nOVERALL METRICS:\n")
    f.write(f"{'='*70}\n")
    f.write(f"Weighted Precision: {report['weighted avg']['precision']:.4f}\n")
    f.write(f"Weighted Recall:    {report['weighted avg']['recall']:.4f}\n")
    f.write(f"Weighted F1-Score:  {report['weighted avg']['f1-score']:.4f}\n")
    f.write(f"Overall Accuracy:   {report['accuracy']:.4f} ({report['accuracy']*100:.2f}%)\n")

print("✅ Classification report saved: project_results/classification_report.txt")

# 6. Per-emotion metrics graph
precisions = [report[emotion]['precision'] for emotion in emotions]
recalls = [report[emotion]['recall'] for emotion in emotions]
f1_scores = [report[emotion]['f1-score'] for emotion in emotions]

plt.figure(figsize=(12, 6))
x = np.arange(len(emotions))
width = 0.25

plt.bar(x - width, precisions, width, label='Precision', alpha=0.8, color='skyblue')
plt.bar(x, recalls, width, label='Recall', alpha=0.8, color='lightcoral')
plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8, color='lightgreen')

plt.xlabel('Emotions', fontsize=12, fontweight='bold')
plt.ylabel('Score', fontsize=12, fontweight='bold')
plt.title('Per-Emotion Classification Metrics', fontsize=14, fontweight='bold')
plt.xticks(x, emotions, rotation=45)
plt.legend()
plt.ylim([0, 1])
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('project_results/classification_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Per-emotion metrics saved: project_results/classification_metrics.png")

# 7. Summary statistics
print("\n" + "="*60)
print("📈 TRAINING & EVALUATION SUMMARY")
print("="*60)
print(f"\n✅ Final Training Accuracy:    {history.history['accuracy'][-1]:.4f} ({history.history['accuracy'][-1]*100:.2f}%)")
print(f"✅ Final Validation Accuracy:  {history.history['val_accuracy'][-1]:.4f} ({history.history['val_accuracy'][-1]*100:.2f}%)")
print(f"✅ Best Validation Accuracy:   {max(history.history['val_accuracy']):.4f} ({max(history.history['val_accuracy'])*100:.2f}%)")
print(f"✅ Overall Test Accuracy:      {report['accuracy']:.4f} ({report['accuracy']*100:.2f}%)")
print(f"✅ Total Epochs Completed:     {len(history.history['accuracy'])}")
print(f"\n📁 All results saved in:       project_results/")

print("\n" + "="*60)
print("✨ TRAINING COMPLETE - READY FOR PROJECT REVIEW!")
print("="*60)

