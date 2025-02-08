# House Price Prediction

## Overview
This project implements a machine learning pipeline to predict house prices based on the **California housing dataset**. The dataset is downloaded, preprocessed, and analyzed using various machine learning techniques, including linear regression, decision trees, random forests, and support vector machines (SVMs). The model is trained, fine-tuned, and evaluated using various statistical and machine learning techniques.

## Features and Workflow
1. **Data Acquisition:**
   - Downloads and extracts the California housing dataset.
   - Loads the data into a Pandas DataFrame.
   
2. **Data Exploration:**
   - Generates descriptive statistics and visualizes distributions.
   - Analyzes correlations between different variables.
   - Plots geographical distributions of housing prices.

3. **Data Preprocessing:**
   - Handles missing values using median imputation.
   - Encodes categorical features using one-hot encoding.
   - Feature scaling using StandardScaler and MinMaxScaler.
   - Creates new engineered features like room-per-household ratios.
   
4. **Model Training & Evaluation:**
   - Implements **Linear Regression**, **Decision Trees**, **Random Forests**, and **Support Vector Machines (SVMs)**.
   - Uses **cross-validation** for model evaluation.
   - Fine-tunes hyperparameters using **GridSearchCV** and **RandomizedSearchCV**.

5. **Feature Engineering & Selection:**
   - Uses **Cluster Similarity** to capture spatial relationships.
   - Implements **SelectFromModel** for feature selection.
   - Tests different feature transformation techniques.
   
6. **Final Model Selection & Deployment:**
   - Selects the best model based on evaluation metrics.
   - Saves the final model using **Joblib**.
   - Reloads and tests the model on unseen data.
   
## Requirements
The following Python libraries are required:
- Python 3.7+
- NumPy
- Pandas
- Scikit-learn (>=1.0.1)
- Matplotlib
- SciPy
- Joblib

## How to Run
1. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn matplotlib scipy joblib
   ```
2. Run the script:
   ```bash
   python house_price_prediction.py
   ```
3. The script will:
   - Download and process the dataset.
   - Train and evaluate models.
   - Save the best-performing model as `my_california_housing_model.pkl`.

## Results
- The best-performing model is selected based on **Root Mean Squared Error (RMSE)**.
- The final model is optimized using hyperparameter tuning.
- The system is validated on a test set and provides confidence intervals for predictions.

## Future Improvements
- Implement deep learning models such as **Neural Networks**.
- Extend the dataset to include **more geographical locations**.
- Improve hyperparameter tuning using **Bayesian Optimization**.

## Author
**Cheng-Yu Wu**

## License
This project is open-source and available under the MIT License.

