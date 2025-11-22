# Multiclass & Binary Ultrasound Frame and Video Classification using a CNN and a hybrid CNN-LSTM model

This repository contains the code used in the study:

> **Deep Learning for Canine Echocardiography: A Hybrid CNN–LSTM Approach for Myxomatous Mitral Valve Disease Detection**

The project includes:

- A **ResNet50 CNN** for frame-level and video-level (via max voting) classification.
- A **hybrid ResNet50–LSTM model** for video-level (sequence-based) classification.
- Training scripts for **binary** and **three-class** classification.
- Dataloaders adapted to the released Zenodo dataset.

---

## 1. Dataset

The code is designed to work with the publicly available dataset:

> **Canine Echocardiography Dataset for MMVD Classification (Frame-Level and Case-Level Versions)**  
> Petraki, E., Koutinas, C., & Vretos, N. (2025).  
> DOI: [10.5281/zenodo.17683921](https://doi.org/10.5281/zenodo.17683921)

The dataset provides two complementary versions:

- **CNN dataset (frame-level)**  
  For training ResNet50 on individual frames and performing sequence classification via max voting.

- **CNN–LSTM dataset (case-level)**  
  For training the hybrid ResNet50–LSTM model on fixed-length sequences (30 frames per case).

Details about the directory structure are included in the dataset’s own README.txt that can be found in the aforementioned webpage.

