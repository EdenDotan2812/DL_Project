Final Project - Deep Learning in medical Imaging course
Classification of Cancer and Blood Cells

Overview
This project focuses on classifying cancer and blood cells using digital holographic microscopy images. The main objective is to enhance diagnostic accuracy and efficiency through the application of Convolutional Neural Networks to classify optical phase density (OPD) images of cells.

Project Description
Circulating tumor cells are crucial biomarkers for cancer diagnosis, prognosis, and treatment monitoring. Detecting these cells in peripheral blood is challenging due to their low abundance and limitations of traditional methods. This project aims to improve the detection and classification of CTCs and various blood cells using deep learning techniques applied to OPD images.

Key Features
Classification of Six Cell Types: The dataset includes 2,000 OPD maps representing two types of colorectal cancer cells (SW480 and SW620) and four types of blood cells (Granulocytes, Lymphocytes, Monocytes, and Erythrocytes).
Deep Learning Models: Implementation and comparison of different CNN architectures, MobilenetV2, resnet50, vgg16, Dinov2 and pre-trained MobilenetV2 on OPD images.

Repository Structure
dataset_invariant_sw.py: Script for loading and processing the cell image dataset.
utils_invariant.py: Contains utility functions for data processing, visualization, and performance metrics.
train.py: Main script to train and evaluate different CNN models on the dataset.
models/: Directory containing custom CNN models and pre-trained models used in the project.
results/: Directory for storing training logs, model checkpoints, and evaluation results.

Results
The best-performing model achieved over 95% accuracy in classifying cancer and blood cells based on their OPD profiles. This demonstrates the potential of using deep learning and label-free imaging for non-invasive cancer diagnostics.
