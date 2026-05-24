# 🎭 Real-Time Emotion Recognition using Deep Learning

## 📌 Project Overview

This project presents a **Real-Time Emotion Recognition System** using **Deep Learning** and **Computer Vision** techniques to classify human facial emotions from:

- 📷 **Live webcam feed**
- 🎥 **Pre-recorded video files**

The system performs **face detection using Haar Cascade during preprocessing** and **MTCNN (Multi-task Cascaded Convolutional Networks) during real-time deployment** for improved speed and accuracy. Emotion classification is performed using multiple custom deep learning architectures built with **PyTorch**.

The project supports **9 emotion classes** and compares multiple deep learning models to determine the best-performing architecture for emotion recognition.

---

## 🎯 Objective

The main objective of this project is to:

- Detect human faces in real time
- Recognize facial emotions using deep learning
- Compare multiple CNN-based architectures
- Perform inference on both webcam and video input
- Build a GPU-accelerated real-time emotion recognition system

---

## 📊 Dataset

This project uses a **publicly available facial emotion recognition dataset collected from Kaggle**.

The dataset contains facial images categorized into the following **9 emotion classes**:

- Angry
- Content
- Disgust
- Fear
- Happy
- Neutral
- Sad
- SillyFace
- Surprise

### Dataset Source

Kaggle Dataset Link:

```text
https://www.kaggle.com/datasets/hhshamimkhan/face-emotion-datasets
```

> Note: The complete dataset is not included in this repository due to size limitations. A small sample dataset is provided for reference.

---

## 😊 Emotion Classes

The model predicts the following **9 emotion categories**:

| Emotion Classes |
|----------------|
| Angry |
| Content |
| Disgust |
| Fear |
| Happy |
| Neutral |
| Sad |
| SillyFace |
| Surprise |

---

## 🧠 Deep Learning Models Implemented

Multiple deep learning architectures were designed and trained for comparison.

### Implemented Models

- **Simple CNN**
- **CNN + Attention**
- **Custom ResNet**
- **ResNet + Attention**
- **MobileNet Custom**
- **VGG16 Custom (Best Performing Model)**

The final inference system uses the **best trained model**:

```text
final_best_model.pth
```

---

## 🔍 Face Detection Pipeline

This project uses **different face detection techniques for different stages** of the pipeline to improve efficiency and real-time performance.

### 1️⃣ Preprocessing Stage — Haar Cascade

During dataset preprocessing, face extraction is performed using:

```text
Haar Cascade Classifier (OpenCV)
```

Purpose:

- Fast face region extraction
- Lightweight preprocessing
- Efficient dataset preparation

Haar Cascade is used during:

```text
Dataset cleaning and face cropping
```

before training the deep learning models.

---

### 2️⃣ Real-Time Inference Stage — MTCNN

For real-time deployment and video inference, the system uses:

```text
MTCNN (Multi-task Cascaded Convolutional Networks)
```

via **FaceNet-PyTorch**.

Purpose:

- More accurate face localization
- Robust multi-face detection
- Better performance under pose variations
- GPU acceleration support

MTCNN is used during:

- 📷 **Webcam-based emotion recognition**
- 🎥 **Video emotion recognition**

Features:

✅ Multi-face detection

✅ High-confidence filtering

✅ Better robustness than Haar Cascade

✅ GPU-accelerated inference

---

## ⚙️ Technologies Used

- Python
- PyTorch
- TorchVision
- OpenCV
- MTCNN (FaceNet-PyTorch)
- NumPy
- Matplotlib
- Scikit-learn
- Pandas

---

## 📂 Dataset Structure

This repository includes a **small sample dataset** for demonstration purposes.

### Sample Dataset

```text
data/sample_dataset/
```

The complete dataset is excluded due to repository size limitations.

The original dataset contains:

- Train split
- Validation split
- Test split

across **9 emotion categories**.



## 📁 Project Structure

```text
Real-Time-Emotion-Recognition/
│── README.md
│── requirements.txt
│
├── data/
│   ├── sample_dataset/
│
└── src/
    ├── __init__.py
    │
    ├── data/
    │   ├── dataset.ipynb
    │   ├── preprocessing.ipynb
    │   └── split_data.ipynb
    │
    ├── detection/
    │   ├── emotion_video_inference.py
    │   ├── realtime_emotion.py
    │   ├── deploy.prototxt
    │   └── res10_300x300_ssd_iter_140000.caffemodel
    │
    ├── models/
    │   ├── all_models.py
    │   └── best_model.pth
    │
    ├── input_videos/
    ├── output_videos/
    │
    └── training/
        ├── train.ipynb
        └── models/
            └── final_best_model.pth

```

---

## 🐍 Environment Requirements

### Python Version

```text
Python 3.10.20
```

### CUDA Version

```text
CUDA 11.8
```

### GPU Support

This project uses:

```text
PyTorch GPU (cu118)
```

for faster training and real-time inference.

---

## 🚀 Installation & Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/Anindya-Midhey/Real-Time-Emotion-Recognition-DeepLearning.git
```

### Step 2: Navigate to Project Folder

```bash
cd Real-Time-Emotion-Recognition-DeepLearning
```

### Step 3: Create Environment

```bash
conda create -n emotion_gpu python=3.10.20 -y
conda activate emotion_gpu
```

### Step 4: Install PyTorch GPU

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Model Training

Run:

```bash
jupyter notebook
```

Open:

```text
src/training/train.ipynb
```

The notebook:

- Trains multiple architectures
- Saves model checkpoints
- Compares performance
- Selects best model

Saved models are stored in:

```text
src/training/models/
```

---

## 🎥 Video Emotion Recognition

To process input videos:

Run:

```bash
python src/detection/emotion_video_inference.py
```

Input videos should be placed inside:

```text
src/input_videos/
```

Processed outputs are saved in:

```text
src/output_videos/
```

---

## 📷 Real-Time Emotion Recognition

For webcam-based emotion detection:

Run:

```bash
python src/detection/realtime_emotion.py
```

The system:

- Detects faces in real time
- Predicts emotions
- Displays confidence scores
- Draws colored emotion labels

Press:

```text
Q
```

to quit.

---

## 🔥 Key Features

✅ Real-time emotion recognition

✅ GPU accelerated inference

✅ Multiple deep learning architectures

✅ Attention-based models

✅ Webcam and video support

✅ Multi-face detection

✅ Automatic dataset preprocessing

✅ Model comparison and checkpoint saving

---

## 🔮 Future Improvements

- Emotion tracking over time
- Audio emotion analysis
- Model deployment using Streamlit/Flask
- Mobile optimization
- Improved inference speed

---

## 👨‍💻 Author

**Anindya Midhey**

---

## ⭐ If you found this project useful, consider giving it a star!
