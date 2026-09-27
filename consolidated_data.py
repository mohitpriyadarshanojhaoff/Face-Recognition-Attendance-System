import os
import pickle
import cv2
import numpy as np

# Use script directory to ensure correct path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'clean data')
IMG_DIR = os.path.join(BASE_DIR, 'images')

os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(IMG_DIR) or len(os.listdir(IMG_DIR)) == 0:
    print(f"Error: No images found in '{IMG_DIR}'. Run collect_data.py first.")
    exit(1)

image_data = []
labels = []

print("Processing images...")
for filename in os.listdir(IMG_DIR):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    
    file_path = os.path.join(IMG_DIR, filename)
    image = cv2.imread(file_path)
    if image is None:
        continue
    
    # Preprocess: Grayscale -> Resize to (100, 100) -> Histogram Equalization
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (100, 100))
    image = cv2.equalizeHist(image)
    
    image_data.append(image)
    # Extract label prefix before underscore, e.g., 'Mohit_0.jpg' -> 'Mohit'
    label_name = filename.split('_')[0]
    labels.append(label_name)

image_data = np.array(image_data, dtype=np.uint8)
labels = np.array(labels)

print(f"Total processed images: {len(image_data)}")
unique_labels, counts = np.unique(labels, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f" - {label}: {count} images")

# Save pickled dataset
images_path = os.path.join(DATA_DIR, "images.p")
labels_path = os.path.join(DATA_DIR, "labels.p")

with open(images_path, 'wb') as f:
    pickle.dump(image_data, f)
print(f"Saved images to {images_path}")

with open(labels_path, 'wb') as f:
    pickle.dump(labels, f)
print(f"Saved labels to {labels_path}")

print("Data consolidation complete!")
