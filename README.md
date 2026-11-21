# California Housing Price Prediction 🏡

## Project Overview
This project uses Machine Learning regression algorithms to predict median house values in California based on housing and demographic features.

The goal is to explore the dataset, train multiple regression models, evaluate their performance, and select the best-performing model.

## Dataset
- Source: Scikit-learn California Housing Dataset
- Dataset contains 20,640 observations.
- 8 input features.
- Target variable: `MedHouseVal`

**Note:** The dataset is based on the 1990 California census. Target values are measured in units of $100,000 and capped at 5.

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Google Colab

## Project Workflow
1. Data Loading and Exploration
2. Exploratory Data Analysis (EDA)
3. Data Visualization
4. Train-Test Split
5. Model Training
6. Model Evaluation
7. Hyperparameter Tuning
8. Feature Importance Analysis
9. Model Saving and Loading

## Machine Learning Algorithms
- Linear Regression
- Decision Tree Regressor
- Tuned Decision Tree Regressor
- Random Forest Regressor
- Tuned Random Forest Regressor

## Model Performance

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 0.5332 | 0.7456 | 0.5758 |
| Decision Tree | 0.4547 | 0.7037 | 0.6221 |
| Tuned Decision Tree | 0.4314 | 0.6385 | 0.6889 |
| Random Forest | 0.3275 | 0.5053 | **0.8051** |

The Original Random Forest achieved the best test R² score of approximately 0.8051.

5-fold cross-validation R²: 0.8041.

## Best Model
Random Forest Regressor

- Training R²: 0.9736
- Testing R²: 0.8051
- Cross-validation mean R²: 0.8041

The model was saved using Joblib as `california_housing_model.pkl`.

## Project Files
- `California_Housing_Price_Prediction.ipynb` – Complete notebook and analysis.
- `california_housing_model.pkl` – Trained Random Forest model.
- `requirements.txt` – Required Python libraries.

## Limitations
- The dataset represents 1990 census data, not current housing prices.
- Predictions are based on district-level data, not individual houses.
- The target variable is capped at $500,000.
- The model may not accurately represent today's housing market.

## Future Improvements
- Experiment with Gradient Boosting and XGBoost.
- Perform more extensive hyperparameter tuning.
- Build an interactive prediction web application using Streamlit.

## Author
B.Tech Computer Science Engineering Student

**Learning Focus:** Machine Learning and Generative AI

## Project Visualizations

### Correlation Heatmap
![Correlation Heatmap](correlation_heatmap.png)

### Feature Importance
![Feature Importance](feature_importance.png)

### Actual vs Predicted
![Actual vs Predicted](actual_vs_predicted.png)

