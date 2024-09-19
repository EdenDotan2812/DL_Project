#Final Project - Deep Learning in Medical Imaging Course
###Classification of Cancer and Blood Cells

##Overview
This project aims to classify cancer and blood cells using digital holographic microscopy images. By applying Convolutional Neural Networks to classify optical phase density (OPD) images, we strive to improve diagnostic accuracy and efficiency in medical imaging.

##Project Description
Circulating tumor cells (CTCs) are vital biomarkers for cancer diagnosis, prognosis, and treatment monitoring. Detecting these rare cells in peripheral blood is challenging due to their low abundance and the limitations of traditional detection methods. This project leverages deep learning techniques to enhance the detection and classification of CTCs and various blood cells using OPD images obtained through digital holographic microscopy.

##Key Features
Classification of 5 Cell Types: The dataset includes 2,000 OPD maps representing two colorectal cancer cell lines (SW480 and SW620) and three types of blood cells (Granulocytes, Monocytes, and PBMCs).
Deep Learning Models: Implementation and comparison of various CNN architectures, including MobileNetV2, ResNet50, VGG16, DinoV2, and a pre-trained MobileNetV2 optimized for OPD images.

##Repository Structure

load_data.py: Script for loading and processing the OPD maps into a structured dataset.

Final_Project.ipynb: Code for training and evaluating MobileNetV2, ResNet50, and VGG16 models on the dataset.

OPD_Model_and_DinoV2.ipynb: Code for training and evaluating DinoV2 and the pre-trained MobileNetV2 models on OPD images for different classification tasks.

saved_models/: Directory containing the trained models used for evaluation.

results/: Directory for storing evaluation plots and results.

##Results
The best-performing model achieved over 95% accuracy in classifying cancer and blood cells based on their OPD profiles. This demonstrates the effectiveness of combining deep learning with label-free imaging for non-invasive cancer diagnostics.
