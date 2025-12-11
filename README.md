#  Multiclass & Binary Ultrasound Frame and Video Classification  
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

👉 [**Data Collection Repo:**] (https://github.com/EvangeliaPetraki/Ultrasound_Classification/tree/main/Dataset%20Collection%20and%20Preprocessing) 


