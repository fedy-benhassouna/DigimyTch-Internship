# Wind Turbine Modeling Project

This repository contains the code and documentation for a machine learning 1 month internship project focused on the analysis and modeling of wind turbine data using various data processing, visualization, and machine learning techniques. The dataset used is sourced from Kaggle and provides SCADA (Supervisory Control and Data Acquisition) data for wind turbines.


## Project Overview
This project aims to analyze wind turbine sensor data, focusing on understanding the relationship between various factors such as wind speed, wind direction, and power production. The project involves data preprocessing, noise detection, and the use of statistical and visualization techniques to draw insights from the data.

## Installation
To run this project, you need to have Python 3 installed along with the following libraries:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

You can install these dependencies using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Data Source
The dataset used in this project is the **Wind Turbine SCADA Dataset** sourced from Kaggle. The data is automatically downloaded and extracted using a custom Python script.

## Data Preprocessing
### Noise Detection and Replacement
The dataset may contain noise, which can distort the analysis. We use Interquartile Range (IQR) to detect and replace outliers in the data:

```python
def noise_threshold(dataframe, column, q1=0.25, q3=0.75):
    Q1 = dataframe[column].quantile(q1)
    Q3 = dataframe[column].quantile(q3)
    iqr = Q3 - Q1
    up_limit = Q3 + 1.5 * iqr
    low_limit = max(Q1 - 1.5 * iqr, 0)  # Ensure the lower limit is not negative
    return low_limit, up_limit

def replace_with_thresholds(dataframe, column):
    low_limit, up_limit = noise_threshold(dataframe, column)
    dataframe.loc[(dataframe[column] < low_limit), column] = low_limit
    dataframe.loc[(dataframe[column] > up_limit), column] = up_limit
```

### Feature Extraction
We extract new features such as `month` and `hour` from the `Date/Time` field to capture temporal patterns.

## Exploratory Data Analysis (EDA)
The dataset is examined to identify the relationship between wind speed, wind direction, and power production. Correlation analysis is performed to understand how these variables are related.

## Correlation Analysis
We calculate and visualize the correlation between wind speed, wind direction, and power production:

```python
correlation_matrix = data[['Wind Speed (m/s)', 'Wind Direction (°)', 'LV ActivePower (kW)']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation between Wind Speed, Wind Direction, and Power Production')
plt.show()
```


## 2. Data Preprocessing

The dataset used includes features such as month, hour, wind speed, theoretical power curve, and wind direction. The target variable is the failure condition.

### 2.1 Feature Selection

The features used in this project include:
- `month`
- `hour`
- `Wind Speed (m/s)`
- `Theoretical_Power_Curve (KWh)`
- `Wind Direction (°)`

The target variable is:
- `Failure Condition`

### 2.2 Splitting the Data

The data is split into training and testing sets using an 80-20 split:
```python
from sklearn.model_selection import train_test_split

X = data[['month','hour', 'Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)', 'Wind Direction (°)']]
y = data['Failure Condition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
```

## 3. Model Training and Evaluation

### 3.1 Models Used
- Logistic Regression
- Random Forest
- XGBoost

The models are evaluated using accuracy, precision, recall, and F1-score due to the imbalanced nature of the dataset.

### 3.2 Model Evaluation Function
A custom function is used to evaluate the models:
```python
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
```

### 3.3 Logistic Regression
```python
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
evaluate_model(lr_model, X_test, y_test)
```

### 3.4 Random Forest
```python
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
evaluate_model(rf_model,X_test, y_test)
```

### 3.5 XGBoost
```python
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
evaluate_model(xgb_model, X_test, y_test)
```

### 3.6 Handling Imbalanced Data with SMOTE
SMOTE (Synthetic Minority Over-sampling Technique) is used to handle the imbalance in the dataset:
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
X_test_res, y_test_res = smote.fit_resample(X_test, y_test)
```

### 3.7 Model Training and Evaluation with SMOTE
The Random Forest and XGBoost models are retrained using the resampled data:
```python
rf_model.fit(X_resampled, y_resampled)
evaluate_model(rf_model, X_test_res, y_test_res)

xgb_model.fit(X_resampled, y_resampled)
evaluate_model(xgb_model, X_test_res, y_test_res)
```

### 3.8 Early Stopping and Hyperparameter Tuning
Early stopping is implemented for the XGBoost model:
```python
X_train_res, X_val_res, y_train_res, y_val_res = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
xgb_model.fit(X_train_res, y_train_res, eval_set=[(X_val_res, y_val_res)], early_stopping_rounds=10, verbose=False)
evaluate_model(xgb_model, X_test_res, y_test_res)
```

GridSearchCV is used to find the best hyperparameters:
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [1, 10, 20]
}
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='f1', verbose=False, n_jobs=-1)
grid_search.fit(X_resampled, y_resampled)
best_xgb_model = grid_search.best_estimator_
evaluate_model(best_xgb_model, X_test_res, y_test_res)
```



## Conclusion
The analysis reveals a strong positive correlation between wind speed and power production, as expected. The trained machine learning model provides a baseline for predicting power output based on wind speed and direction. Further improvements can be made by exploring more advanced models and feature engineering techniques.
The XGBoost model, even without hyperparameter tuning, provides the best balance between precision and recall, making it the preferred model for this task.

