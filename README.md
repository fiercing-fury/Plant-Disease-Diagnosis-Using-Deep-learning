# AI-Powered Plant Disease Diagnosis

This project uses a Convolutional Neural Network (CNN) built with TensorFlow and Keras to diagnose diseases in plant leaves from images.

## Overview

The model leverages transfer learning with the MobileNetV2 architecture to achieve high accuracy in classifying 38 different disease categories from the PlantVillage dataset.

## Features
-   Trained on the comprehensive PlantVillage dataset.
-   Achieved an accuracy of **[92.29%]** on the test set.
-   Includes a script for training (`train.py`) and a script for prediction (`predict.py`).

## How to Use
1.  Clone the repository: `git clone [https://github.com/fiercing-fury/Plant-Disease-Diagnosis-Using-Deep-learning]`
2.  Install the dependencies: `pip install tensorflow matplotlib opencv-python scikit-learn seaborn`
3.  Download the dataset (Plant Village Dataset from kaggle).
4.  Run the training script: `python train.py`
5.  Make a prediction on a new image: `python predict.py`

## Results
The model's performance was evaluated using precision, recall, F1-score, and a confusion matrix.
* **Accuracy:** [92.29]%
* **Loss/Accuracy Curves:**
    ![Training Curves](/plot_image.png) ```
