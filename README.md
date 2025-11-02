# Apartment-Price-Prediction-Using-Linear-Regression
A regression project that uses Linear Regression to predict apartment prices based on various features. The project includes data cleaning, feature scaling, anomaly detection, and model evaluation using metrics such as MAE, MSE, and R².
# Apartment Price Prediction Using Linear Regression

This project demonstrates how to predict apartment prices using Linear Regression based on various features. The project includes data preprocessing, anomaly detection, feature scaling, and model evaluation.

## Project Description

In this project, the following steps are applied:

1. **Importing Libraries**:
   - Libraries like **Pandas**, **NumPy**, **Scikit-learn**, **Matplotlib**, and others are imported for data processing, model building, and visualization.

2. **Loading Data**:
   - The dataset is loaded using **Pandas** and the first few rows are displayed using `head()` to check the data.
   - Basic information about the dataset is reviewed using `info()`.

3. **Data Cleaning**:
   - Data cleaning is performed to handle missing values, if any.
   
4. **Anomaly Detection**:
   - Anomaly detection is done using **Interquartile Range (IQR)** for numeric columns.
   - Histograms are plotted for each numeric column to visualize and detect anomalies.

5. **Encoding Categorical Features**:
   - **One-Hot Encoding** is applied to convert categorical variables like addresses into numerical features for model compatibility.

6. **Data Visualization**:
   - Scatter plots are created to visualize the relationship between different price features.

7. **Data Splitting**:
   - The dataset is split into training and testing sets using **train_test_split** (80% for training, 20% for testing).

8. **Feature Scaling**:
   - Numeric features are scaled using **StandardScaler** to improve the performance of the model.

9. **Model Training**:
   - A **Linear Regression** model is trained using the training data.

10. **Model Evaluation**:
    - The model is evaluated using **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, and **R²** metrics.

11. **Model Prediction**:
    - A scatter plot is created to show the actual vs. predicted prices, visually demonstrating the quality of the model's predictions.

## Libraries Used
- Pandas
- NumPy
- Scikit-learn (LinearRegression, StandardScaler)
- Matplotlib
- 3. **Run the code**:
Open the `updated_regression(1).ipynb` file in Jupyter Notebook and follow the steps for data preprocessing, model training, and evaluation.

4. **Visualize Predictions**:
The model's predictions can be visualized with scatter plots to compare the actual vs. predicted prices.

## Contributing
If you would like to contribute to this project, please feel free to submit a pull request. All suggestions and improvements are welcome!


## How to Use
1. **Clone the repository**:
