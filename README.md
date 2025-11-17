Breast Cancer Classification using Random Forest

This project uses machine learning to classify breast tumors as malignant or benign. It includes data loading, exploratory data analysis, visualization, and a Random Forest classifier.




Project Overview

This project analyzes the Breast Cancer Wisconsin dataset from scikit-learn. It performs the following steps:

1. Load and inspect the dataset


2. Visualize class distribution and feature relationships


3. Train a Random Forest classifier


4. Evaluate model performance


5. Display feature importance





Technologies Used

Python

Pandas

NumPy

Matplotlib

scikit-learn





Dataset Information

The dataset contains:

569 samples

30 numeric features

Target variable:

0 = Malignant

1 = Benign



Source: sklearn.datasets.load_breast_cancer




Visualizations Included

The project produces the following visualizations:

Bar chart of class distribution

Correlation heatmap

Scatter plot (mean radius vs mean texture)

Feature importance plot from the Random Forest model





Model Used

Model: RandomForestClassifier from scikit-learn.

Performance Metrics

Accuracy: 97%

Confusion matrix

Precision, recall, and F1-score for both classes


These results demonstrate that the model performs well on the dataset.




How to Run This Project

Option 1: Google Colab

1. Upload your .py file to Colab


2. Open it in a notebook cell


3. Run the script


4. Visualizations and model output will appear automatically



Option 2: Local Environment

Install required libraries:

pip install pandas numpy matplotlib scikit-learn

Run the file:

python your_script_name.py




File Structure

├── breast_cancer_project.py
└── README.md

(Adjust the filename as needed.)




Key Findings

The model identified features such as worst radius, mean concavity, and mean perimeter as strongly influential.

The dataset is slightly imbalanced but does not significantly affect model performance.

Visualizations help reveal relationships and patterns in the features.





Conclusion

This project demonstrates the complete workflow of a supervised machine learning classification task, including data exploration, visualization, model building, and performance evaluation
