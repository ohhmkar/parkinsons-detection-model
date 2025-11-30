# ***Parkinson's Disease Detection using Machine Learning***

## Project Overview

This project builds a robust diagnostic tool to detect Parkinson's Disease using acoustic voice features. It compares three different machine learning architectures:

Gradient Boosting (XGBoost/GBM)

Support Vector Machine (SVM)

Deep Neural Network (MLP)

The final model is an Ensemble of these three, achieving high sensitivity and specificity. The project also includes model explainability using SHAP values to identify key vocal biomarkers (e.g., Pitch Period Entropy, Spread).

## Key Results

Ensemble Accuracy: ~95%

Key Features Identified: PPE, spread1, MDVP:Fo(Hz)

Explainability: SHAP plots confirm that vocal instability (high entropy/jitter) is the primary predictor of the disease.

## Installation

Clone the repository:
```
git clone [https://github.com/ohhmkar/parkinsons-detection.git]
cd parkinsons-detection
```

Install dependencies:
```
pip install -r requirements.txt
```

## Usage

Run the analysis script to download data, train models, and generate reports:

python notebooks/parkinsons_analysis.py


The script will:

* Download the dataset from the UCI Machine Learning Repository.

* Perform EDA (Violin plots, Correlation heatmaps).

* Train and tune all three models.
 
* Generate evaluation metrics (Confusion Matrix, ROC-AUC, SHAP).

* Save the trained models to the models/ directory. 

* Simulate a patient diagnosis using the predict_patient_risk() function.

## Project Structure

notebooks/: Contains the main analysis code (parkinsons_analysis.py).

models/: Stores the saved .pkl and .keras models after training.

data/: Stores the dataset.
##  Model Explainability

This project uses SHAP (SHapley Additive exPlanations) to interpret model decisions. The summary plot (generated during runtime) shows how high values of features like PPE (Pitch Period Entropy) positively correlate with Parkinson's risk.

📜 License

MIT
