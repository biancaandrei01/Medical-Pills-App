# Automatic Pill Detection and Classification

This repository hosts the source code for an end-to-end deep learning system designed for pill detection and classification. The project covers the full development lifecycle, from data acquisition and annotation strategies to model training and evaluation, culminating in an interactive GUI for model testing. The code is publicly available here; however, the custom-built datasets and final model weights are private and not included in this repository.

---

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [The Problem](#the-problem)
3.  [Our Solution](#our-solution)
4.  [Tech Stack](#tech-stack)
5.  [The Datasets](#the-datasets)
    - [The Roboflow "Drug" Dataset](#the-roboflow-drug-dataset)
    - [The Custom "Lab" Dataset](#the-custom-lab-dataset)
6.  [Models Used](#models-used)
    - [YOLOv11](#yolov11)
    - [ResNet-50](#resnet-50)
7.  [Experimental Results](#experimental-results)
8.  [The Interactive Application](#the-interactive-application)

---

## Project Overview

The primary goal of this project is to develop a robust algorithm for the automatic detection and classification of medical pills using modern computer vision techniques. This represents a significant challenge in the pharmacological and medical fields, and a successful solution can help reduce medication errors and streamline identification processes.

### Key Features
- **Custom Dataset Creation:** A new dataset of 1,120 images across 35 pill classes was acquired in a controlled lab environment to supplement existing public data.
- **Dual-Model Approach:** Implementation and comparison of two state-of-the-art models: **YOLOv11** for simultaneous detection and classification, and **ResNet-50** for high-accuracy classification.
- **In-Depth Analysis:** Comprehensive evaluation of model performance on different datasets (uniform vs. varied conditions) and the impact of various data augmentation strategies.
- **End-to-End Workflow:** The project covers the entire ML lifecycle: data acquisition, manual annotation (`labelImg`), preprocessing, model training, evaluation, and deployment in a user-friendly application.
- **Interactive GUI:** A desktop application built with `tkinter` allows for easy, real-time model inference and visual comparison of results.

---

## The Problem

Currently, there is a lack of large, diverse, and correctly labeled public datasets for pill identification. This scarcity makes it difficult to train and evaluate models that can generalize effectively to real-world conditions where lighting, background, and pill orientation can vary significantly.

## Our Solution

To address these limitations, we undertook a multi-faceted approach:
1.  **Supplemented Existing Data:** We started with the Roboflow "Drug" dataset, which contains images taken in standardized conditions.
2.  **Created a Diverse Dataset:** We built our own "lab" dataset with controlled variations in lighting, background (white, hand, patterned templates), and pill position to introduce visual complexity and realism.
3.  **Evaluated Robustness:** We rigorously tested and compared state-of-the-art models (YOLOv11, ResNet-50) to understand their performance trade-offs in different scenarios.
4.  **Developed a Demonstrator:** We built a practical tool to showcase the models' capabilities interactively.

---

## Tech Stack

- **Languages:** Python
- **Core Libraries:** PyTorch, Pandas, NumPy, OpenCV
- **ML Models:** YOLOv11, ResNet-50
- **Data Annotation:** `labelImg`
- **Experiment Tracking:** **[Weights & Biases](https://wandb.ai/transformers_3/Medical%20Pills%20App)**
- **GUI:** `tkinter`

---

## The Datasets

Two primary datasets were used for training and evaluation.

### The Roboflow "Drug" Dataset
- A public dataset containing 932 images across 30 distinct classes.
- Captured in highly uniform, standardized conditions (Petri dish background, constant lighting).
- After cleaning and removing duplicates, our final set contained **545 images**.

### The Custom "Lab" Dataset
- **Objective:** To create a more realistic and diverse dataset.
- **Volume:** **1,120 images** across **35 pill classes**.
- **Acquisition Protocol:**
  - **Equipment**: Sony Cyber-Shot DSC-RX100, tripod, templates, ruler 
  - **Positioning**: Perpendicular shot from a fixed height (11cm extension) 
  - **Lighting**: Varied conditions: natural light, artificial light, artificial + flash 
  - **Backgrounds**: 4 types: white sheet, human hand, 4 patterned templates 
  - **Resolution**: 5472 x 3648 pixels 
  - **Metadata**: All variables (lighting, background, etc.) logged in a central `.csv` file 

#### Data Distribution
The custom "lab" dataset was designed to be perfectly balanced, while the Roboflow dataset showed some class imbalance.

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/862eee9a-09d4-46c5-8d41-e1341744503c" alt="Lab Distribution"></td>
    <td><img src="https://github.com/user-attachments/assets/6ace6899-51fc-4a3d-aaed-73fcf76f05d1" alt="Robo Distribution"></td>
  </tr>
</table>

---

## Models Used

### YOLOv11
A next-generation object detection model that provides an excellent balance of speed and accuracy, making it ideal for real-time applications. It performs detection and classification in a single pass.

<img src="https://github.com/user-attachments/assets/6623b823-ac2e-4bd1-b947-6087e50e1db4" alt="YOLOv11 Architecture" width="50%"/>

### ResNet-50
A classic and powerful deep residual network, renowned for its high accuracy in image classification tasks. By using skip connections, it can train very deep networks effectively without performance degradation.

<img src="https://github.com/user-attachments/assets/7fe6b8c5-84f8-49c8-9377-6745104f4de2" alt="ResNet Architecture" width="50%"/>

---

## Experimental Results

Our experiments revealed critical insights into model performance and the role of data augmentation:
- **On Homogeneous Data (`robo` set):** Augmentations had a minimal or even negative impact. The models performed best on the original, clean images due to the lack of visual variance.
- **On Diverse Data (`lab` set):** Augmentations (especially combined color and geometric transforms) significantly improved performance and model generalization.
- **Combined Dataset (`lab` + `robo`):** Training on the combined set yielded the most robust results, demonstrating that combining diverse, real-world data with standardized data creates a powerful training environment.
- **Model Comparison:**
    - **YOLOv11** excelled at the combined task of precisely localizing and classifying pills, making it ideal for real-time systems.
    - **ResNet-50** achieved superior scores in pure classification (Precision, Recall, F1-score) and was more stable against minor image variations.
---

## The Interactive Application

To facilitate easy testing and demonstration, we developed a GUI using `tkinter`. The application allows a user to:
1.  Select the model to use (YOLOv11 or ResNet-50).
2.  Select the dataset the model was trained on (`lab`, `robo`, or `lab+robo`).
3.  Load an image.
4.  Apply the model and view the inference results directly on the image.
<img src="https://github.com/user-attachments/assets/672055ef-d0e5-4629-a96b-bdaf3b918b95" alt="My Project GUI" width="50%"/>
