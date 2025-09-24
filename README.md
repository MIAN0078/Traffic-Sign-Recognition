# Traffic-Sign-Recognition
# Project Overview
This repository contains the implementation of a Traffic Sign Recognition system using Convolutional Neural Networks (CNNs). The model is trained on the GTSRB dataset and can classify traffic signs into their respective categories.
The notebook Traffic_Sign_Recognition.ipynb includes:
Dataset preprocessing
Model building (CNN)
Training & validation
Performance evaluation (accuracy, confusion matrix, classification report, graphs)

# Dataset

German Traffic Sign Recognition Benchmark (GTSRB)
43 traffic sign classes.
Preprocessing steps applied:
Image resizing
Normalization
One-hot encoding of labels

# Requirements
Install dependencies with:
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python


# Model Workflow

(1) Data Loading & Preprocessing
      Read images from dataset.
      Resize and normalize pixel values.
      Split into training and test sets.
(2)CNN Model Architecture
     Convolutional layers + MaxPooling layers
     Dropout layers for regularization
     Fully connected dense layers
     Softmax output for multi-class classification
(3)Training & Evaluation
      Model trained using Adam optimizer and categorical crossentropy loss.
      Metrics: accuracy, confusion matrix, and classification report.
      Plots: training/validation accuracy & loss curves.


  # How to Run

(1)Clone this repository:

    git clone https://github.com/your-username/Traffic_Sign_Recognition.git
    cd Traffic_Sign_Recognition
(2)Open the notebook:

    jupyter notebook Traffic_Sign_Recognition.ipynb
(3)Run all cells to:
      Train the CNN model
      Evaluate on test set
      View results and graphs


# Results

The trained CNN achieved an accuracy of 95.3% on the test dataset.
Evaluation included:
Accuracy score
Confusion matrix heatmap
Classification report      


# Future Work

Real-time detection using webcam input.
Transfer learning with ResNet, InceptionV3, etc.
Deploy as a web or mobile app.

