# Deep Learning for Canine Echocardiography  

### Multiclass & Binary Ultrasound Frame and Video Classification 
### ResNet50 CNN & Hybrid CNN–LSTM Models for MMVD Detection in Canine Echocardiography

This repository contains the code used in the study:

> **Deep Learning for Canine Echocardiography: A Hybrid CNN–LSTM Approach for Myxomatous Mitral Valve Disease Detection**  
> Petraki, E., Koutinas, C., Vretos, N. (2025)

The project implements two model families:

- A **ResNet50 CNN** for frame-level classification and video-level classification via max voting  
- A **hybrid ResNet50–LSTM model** for sequence-based video classification  

Both **binary** (MMVD vs Normal) and **multiclass** (Healthy / Moderate MMVD / Severe MMVD) tasks are supported.

---

The project implements two model families:

- A **ResNet50 CNN** for frame-level classification and video-level classification via max voting  
- A **hybrid ResNet50–LSTM model** for sequence-based video classification  

Both **binary** (MMVD vs Normal) and **multiclass** (Healthy / Moderate MMVD / Severe MMVD) tasks are supported.

---

## 🔗 Dataset

The training code is designed for the publicly available dataset:

> **Canine Echocardiography Dataset for MMVD Classification**  
> Frame-level and video-level versions  
> DOI: [10.5281/zenodo.17683921](https://doi.org/10.5281/zenodo.17683921)

The dataset provides two complementary versions:

- **CNN dataset (frame-level)**  
  For training ResNet50 on individual frames and performing sequence classification via max voting.

- **CNN–LSTM dataset (case-level)**  
  For training the hybrid ResNet50–LSTM model on fixed-length sequences (30 frames per case).

Details about the directory structure are included in the dataset’s own README.txt that can be found in the aforementioned webpage.

This codebase focuses on **model training and evaluation**.  
Data collection, raw video extraction, and frame generation are documented separately:

👉 **Data Collection Repo:**(https://github.com/EvangeliaPetraki/Ultrasound_Classification/tree/main/Dataset%20Collection%20and%20Preprocessing) 

--- 


## 📂 Repository Structure

### **1. CNN Training (Frame-Level)**
- `train_cnn_binary.py` — Binary CNN classifier (MMVD vs Normal) using ResNet50  
  :contentReference[oaicite:0]{index=0}  
- `train_cnn_multiclass.py` — Three-class CNN classifier  
  :contentReference[oaicite:1]{index=1}  

Each script includes:
- Albumentations-based augmentation  
- Fine-tuning of ResNet50 (selective layer unfreezing)  
- Classifier head replacement  
- Frame-level evaluation metrics  
- Video-level prediction using **max voting**

---

### **2. CNN–LSTM Training (Video-Level / Sequence-Based)**

- `train_cnn_lstm_binary.py` — Binary hybrid CNN–LSTM model  
  :contentReference[oaicite:2]{index=2}  
- `train_cnn_lstm_multiclass.py` — Multiclass hybrid CNN–LSTM model  
  :contentReference[oaicite:3]{index=3}  

The pipeline:
- Extracts ordered frame sequences from each video  
- Passes each frame through ResNet50 (feature embedding)  
- Feeds embeddings into an LSTM for temporal modeling  
- Outputs a single prediction per video

---

### **3. Dataloaders**

- `create_dataloaders.py` — Albumentations-enabled dataset class and loader generator  
  :contentReference[oaicite:4]{index=4}  

Features:
- ImageFolder-compatible loader  
- Augmentation support (Resize, Normalize, Flip, Rotate, Affine, etc.)  
- GPU-optimized pinning and batching  

---

## 🧠 Models

### **ResNet50 CNN**
- Uses HuggingFace implementation:  
  `"agent593/Thyroid-Ultrasound-Image-Classification-Resnet50Model"`
- Layer-wise fine-tuning: early encoder blocks frozen or minimally tuned; deeper blocks and classifier trained at higher LR
- Final classifier:  
  - Flatten → Linear(2048→1024) → ReLU → Dropout  
  - Linear(1024→64) → ReLU → Dropout  
  - Linear(64→N_classes)

### **Hybrid CNN–LSTM**
- Frame features extracted via ResNet50  
- LSTM processes sequences (video clips)  
- Output classification head applied to the final hidden state  

---

## 📊 Evaluation

Frame-level metrics:
- Accuracy, Precision, Recall, F1  
- Balanced Accuracy  
- Specificity (from confusion matrix)  
- Matthews Correlation Coefficient (MCC)  
- ROC-AUC  

Video-level evaluation:
- **Majority voting** across frame predictions  
- Confidence-based selection of the most informative frames  
- Neighbor frame analysis (±2 around highest-confidence frame)

---

## ▶️ Running the Models

### Train CNN (example):
```bash
python train_cnn_binary.py --batch_size 16 --lr_resnet 1e-5 --lr_classifier 1e-3 --num_epochs 25
``` 

## Requirements
- `Python 3.8+`
- `PyTorch`
- `Albumentations`
- `scikit-learn`
- `HuggingFace Transformers`

