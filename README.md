# Real-Time Face Detection & Recognition

A deep-learning-based Face Detection and Recognition system built with **OpenCV** (Haar Cascades) and **TensorFlow / Keras** (Convolutional Neural Networks). Supports both **built-in / USB webcams** and **IP camera streams** (e.g. mobile IP Webcam apps).

---

## Project Structure

```text
face detection/
├── images/                             # Stored face images for each individual
│   └── Mohit_0.jpg ... Mohit_99.jpg   # 100 face crops per person
├── clean data/                         # Preprocessed serialized dataset
│   ├── images.p                        # Numpy array of normalized (100, 100) images
│   └── labels.p                        # Associated class labels
├── haarcascade_frontalface_default.xml # OpenCV pre-trained Haar Cascade detector
├── collect_data.py                     # Captures 100 face crops per person from camera
├── consolidated_data.py                # Preprocesses images & updates clean data
├── train.py                            # Trains CNN model & exports weights + class map
├── recognize.py                        # Real-time face detection & recognition
├── final_model.h5                      # Saved trained Keras CNN model
└── classes.npy                         # Serialized dynamic class label mappings
```

---

## Requirements

The project runs using standard Python packages:
- `tensorflow`
- `opencv-python`
- `numpy`
- `scikit-learn`

---

## Step-by-Step Usage Guide

### 1. Collect Face Samples (`collect_data.py`)
Run this script to register a new person:
```bash
python collect_data.py
```
- Enter the person's name (e.g., `Mohit` or `Alex`).
- Choose camera source:
  - `0`: Built-in / USB webcam.
  - `1`: IP Camera URL (e.g. `http://192.168.1.15:8080/shot.jpg`).
- The script automatically captures 100 face crops and stores them in `images/<Name>_<Index>.jpg`.
- *Recommendation*: Collect samples for at least 2 different people for multi-class facial recognition.

---

### 2. Consolidate & Preprocess Dataset (`consolidated_data.py`)
After adding or updating face images in `images/`:
```bash
python consolidated_data.py
```
- Converts images to grayscale and resizes them to `100x100`.
- Applies histogram equalization (`cv2.equalizeHist`) for consistent contrast across lighting variations.
- Serializes the dataset into `clean data/images.p` and `clean data/labels.p`.

---

### 3. Train the Model (`train.py`)
Train the CNN classifier on your consolidated dataset:
```bash
python train.py
```
- Loads the dataset and normalizes pixel intensities (`0.0 - 1.0`).
- Dynamically generates class mappings using `LabelEncoder`.
- Automatically handles both single-person datasets (with synthetic negative samples) and multi-person datasets.
- Trains a 3-layer Convolutional Neural Network with Dropout regularization.
- Saves the trained model to `final_model.h5` and class labels to `classes.npy`.

---

### 4. Real-time Face Recognition (`recognize.py`)
Run real-time inference on live camera feed:
```bash
python recognize.py
```
- Loads `haarcascade_frontalface_default.xml`, `final_model.h5`, and `classes.npy`.
- Choose your camera source (webcam `0` or IP stream URL).
- Displays bounding boxes with the predicted name and confidence percentage.
- Automatically flags unrecognized faces as **`Unknown`** if confidence is below the threshold or the identity does not match.
- Press **`q`** to exit the recognition window.
