import os
import pickle
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'clean data')
IMAGES_PATH = os.path.join(DATA_DIR, 'images.p')
LABELS_PATH = os.path.join(DATA_DIR, 'labels.p')
MODEL_PATH = os.path.join(BASE_DIR, 'final_model.h5')
CLASSES_PATH = os.path.join(BASE_DIR, 'classes.npy')

print("=" * 60)
print(" FACE RECOGNITION MODEL TRAINING ")
print("=" * 60)

if not os.path.exists(IMAGES_PATH) or not os.path.exists(LABELS_PATH):
    print(f"Error: Missing dataset in '{DATA_DIR}'. Please run 'consolidated_data.py' first.")
    exit(1)

print("Loading dataset...")
with open(IMAGES_PATH, 'rb') as f:
    images = pickle.load(f)

with open(LABELS_PATH, 'rb') as f:
    labels = pickle.load(f)

print(f"Loaded {len(images)} images.")
unique_labels, counts = np.unique(labels, return_counts=True)
print("Classes found:")
for ul, c in zip(unique_labels, counts):
    print(f" - {ul}: {c} samples")

# If only 1 person exists, create synthetic negative class so training can proceed
if len(unique_labels) < 2:
    print("\n" + "!" * 60)
    print("NOTE: Only 1 person class was found.")
    print("A classifier needs at least 2 distinct classes to distinguish faces.")
    print("Generating a synthetic 'Unknown/Background' class for demonstration...")
    print("Tip: Run 'collect_data.py' for other people to build a multi-person dataset.")
    print("!" * 60 + "\n")

    # Generate synthetic negative images (inverted, blurred, noise patterns)
    synthetic_imgs = []
    synthetic_labels = []
    for img in images:
        # Create heavily blurred / noisy negative samples
        noise = np.random.randint(0, 50, (100, 100), dtype=np.uint8)
        neg_img = cv2.bitwise_not(cv2.GaussianBlur(img, (25, 25), 0))
        neg_img = cv2.add(neg_img, noise)
        synthetic_imgs.append(neg_img)
        synthetic_labels.append("Unknown")

    images = np.concatenate([images, np.array(synthetic_imgs, dtype=np.uint8)], axis=0)
    labels = np.concatenate([labels, np.array(synthetic_labels)], axis=0)

# Preprocessing: normalize and reshape
print("\nPreprocessing images...")
X = images.astype('float32') / 255.0
X = np.expand_dims(X, axis=-1)  # Shape: (N, 100, 100, 1)

# Encode string labels to integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
class_names = list(label_encoder.classes_)
num_classes = len(class_names)

# Save class names mapping
np.save(CLASSES_PATH, label_encoder.classes_)
print(f"Classes ({num_classes}): {class_names}")
print(f"Saved class mapping to: {CLASSES_PATH}")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

# Build Convolutional Neural Network
print("\nBuilding CNN Architecture...")
model = Sequential([
    Input(shape=(100, 100, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train Model
print("\nStarting Training...")
EPOCHS = 15
BATCH_SIZE = 16

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("\n" + "=" * 50)
print(f"Evaluation Test Loss:     {loss:.4f}")
print(f"Evaluation Test Accuracy: {accuracy * 100:.2f}%")
print("=" * 50)

# Save trained model
model.save(MODEL_PATH)
print(f"\nModel successfully saved to: {MODEL_PATH}")
print("You can now run 'recognize.py' to test real-time recognition!")
