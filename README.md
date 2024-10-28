# Multiclass Classification with Pins Face Dataset Using Keras (ANN Only)
## Objective
This project aims to develop a multiclass classification model to identify individuals based on facial images using Keras and an Artificial Neural Network (ANN). The goal is to achieve an accuracy of over 6% on this task.

## Dataset Description
The dataset used in this project, "Pins Face Recognition Dataset," is available on Kaggle and contains images of various celebrities. Each folder in the dataset corresponds to a unique individual, where each image serves as a sample for classification.

**Dataset link:** Kaggle - Pins Face Dataset

**Structure:** Multiple folders (one per individual), each containing multiple images.

**Classes:** Each folder represents a unique class (i.e., a specific individual).
## Instructions for Running the Code
### 1. Data Preparation
1- Download the dataset from Kaggle and extract it to a folder.

2- Run the data splitting code to split the dataset into training, validation, and test sets with an 70-15-15% split.

### 2. Model Training
Run the provided script to train the ANN model. The model is trained on the training dataset, and validation data is used to monitor the training progress. An early stopping callback is implemented to halt training if validation loss does not improve.
### 3. Model Evaluation
1- After training, the model is evaluated on the test set.

2- A confusion matrix and classification report (precision, recall, F1-score) are generated for detailed performance analysis.
### 4. Visualizations
The script plots training/validation loss and accuracy curves to visualize the model's learning progress.
## Dependencies and Installation Instructions
### Prerequisites
Ensure Python is installed on your system. The code uses the following Python libraries:

1- **numpy**: for numerical operations on arrays.

2- **matplotlib** and **seaborn**: for data visualization.

3- **tensorflow** (Keras included): for model building and training.

4- **opencv-python**: for image processing (optional if performing face cropping).

5- **sklearn**: for model evaluation metrics.
### Installation
Run the following command to install the required packages:

pip install numpy matplotlib seaborn tensorflow opencv-python scikit-learn
## Expected Results
The model targets an accuracy of over 6% on the test dataset. During training, it uses data augmentation and regularization techniques like dropout to generalize well across different facial images of the same person. Model performance is tracked through training and validation loss and accuracy, and the final accuracy is evaluated on the test set.

The expected output includes:

1- Final test accuracy.

2- Classification report with precision, recall, and F1-scores.

3- Confusion matrix for a detailed view of classification performance across classes.

4- Visual plots of training and validation metrics (loss and accuracy) over epochs.