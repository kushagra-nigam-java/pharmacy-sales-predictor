# Pharmacy Sales Predictor
### Intelligence-Driven OTC Inventory Forecasting

**Pharmacy Sales Predictor** is a terminal-based machine learning solution designed to forecast demand for Over-the-Counter (OTC) medications. By analyzing historical sales patterns across different regions and dates, this tool helps pharmacies optimize their supply chain, minimizing both stockouts and the waste of expired inventory.

---

## Project Overview
The primary challenge in retail pharmacy is balancing stock levels. Overstocking leads to expired chemical waste, while understocking impacts patient care. This project implements a classification pipeline that identifies high-demand products using ensemble learning techniques.

### Model Evaluation and Performance
We conducted a comparative analysis between two different ensemble architectures to establish the most reliable forecasting model:

| Model | Architecture | Role | Accuracy |
| :--- | :--- | :--- | :--- |
| **Random Forest (m1)** | Parallel Ensemble | Baseline | **~21%** |
| **Gradient Boosting (m2)** | Sequential Ensemble | **Final Model** | **~24%** |

*Note: The Gradient Boosting model was selected for the final implementation. Its sequential learning approach allowed for better correction of residual errors compared to the baseline, making it more effective for the specific demand patterns in the OTC dataset.*

---

## Technical Setup

### Prerequisites
* Python: 3.8 
* Package Manager: pip

### Installation Steps
1. Clone the repository to your local machine.
2. Navigate into the project directory: pharmacy-sales-predictor.
3. Install the required libraries using the requirements.txt file.
   (Core libraries: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, and Joblib).

---

## Execution Guide

### 1. Model Training
To train the models and generate performance metrics such as Accuracy, MAE, and Confusion Matrices:

* Run the Random Forest Baseline script located in the scripts folder.
* Run the Gradient Boosting (Final) script located in the scripts folder.

### 2. Demand Prediction (CLI)
Use the command-line interface to predict high-demand medication by providing the country name and the target date as arguments to the m2 script.

Example Result: 
Predicted high-demand product: Paracetamol

---

## Data and Pipeline
The system processes data from the pharmacy_otc_sales_data.csv file, focusing on features such as Date, Product, Amount, and Country.

### Workflow Logic
1. Feature Engineering: Decomposing timestamps into Month and Day-of-Week components.
2. Data Transformation: Implementing Label Encoding for categorical variables.
3. Model Training: Comparative evaluation of ensemble methods.
4. Serialization: Saving trained models and encoders as .pkl files for seamless inference.

---

## Repository Structure
* data/ - Historical OTC sales data
* models/ - Saved .pkl model files
* scripts/ - Machine Learning and CLI scripts
* requirements.txt - List of project dependencies
* README.md - Project documentation

---

## Academic Context
* Course: CSA2001 — Fundamentals in AI and ML
* Author: Kushagra Nigam
* Registration No: 25BAI11055
* Program: B.Tech (CSE - AI & ML)
* Date: March 2026
