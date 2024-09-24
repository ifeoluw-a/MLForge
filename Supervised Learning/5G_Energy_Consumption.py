# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing machine learning libraries
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols

# Configurations for visualizations
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')  # Ignore warnings for a cleaner output

# Profiling tool
from ydata_profiling import ProfileReport

# Load the dataset (Make sure the path to your file is correct)
df = pd.read_csv('/content/5G_energy_consumption_dataset.csv.crdownload')

# Generate a report of the dataset using pandas-profiling (useful for EDA)
profile = ProfileReport(df, title='5G Energy Consumption Dataset Report')
profile.to_notebook_iframe()  # View the profile report within Jupyter notebooks

# Data Overview
df.info()  # Check the structure of the dataset (data types, null values, etc.)
df.describe()  # Descriptive statistics for numerical columns

# Convert 'Time' column to datetime for easier manipulation
df['Time'] = pd.to_datetime(df['Time'])

# Visualizing box plots for potential outliers in key variables
df[['load', 'ESMODE']].plot(kind="box")
df['Energy'].plot(kind="box")

# Display the first few rows of relevant columns
df[['Energy', 'load', 'ESMODE']].head()

# Function to detect and remove outliers based on IQR
def outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df[~((df[col] < lower_bound) | (df[col] > upper_bound))]

# Remove outliers for selected columns
df = outliers(df, 'Energy')
df = outliers(df, 'load')
df = outliers(df, 'ESMODE')

# Correlation matrix to explore relationships between numerical variables
df2 = df.select_dtypes(np.number)
corr_matrix = df2.corr()['Energy']
print(corr_matrix)

# Pairplot for visualizing relationships between features
sns.pairplot(df)

# Prepare features and target variables for modeling
x = df2.drop('Energy', axis=1)
y = df2['Energy']

# Splitting data into training and test sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale the data using MinMaxScaler for normalization
scale = MinMaxScaler()
x_train_scaled = scale.fit_transform(x_train)
x_test_scaled = scale.transform(x_test)

# Linear Regression Model
lr = LinearRegression()
lr.fit(x_train_scaled, y_train)

# Make predictions and evaluate the model
predictions = lr.predict(x_test_scaled)
print("Linear Regression R^2 Score:", r2_score(y_test, predictions))
print("MSE:", mean_squared_error(y_test, predictions))
print("MAE:", mean_absolute_error(y_test, predictions))

# Polynomial Regression Model
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

# Fit Polynomial Regression
lr.fit(x_train_poly, y_train)
poly_predictions = lr.predict(x_test_poly)
poly_train_pred = lr.predict(x_train_poly)

# Evaluate Polynomial Regression
print("Polynomial Regression R^2 Score:", r2_score(y_test, poly_predictions))
print("Train MSE:", mean_squared_error(y_train, poly_train_pred))
print("Test MSE:", mean_squared_error(y_test, poly_predictions))
print("Train MAE:", mean_absolute_error(y_train, poly_train_pred))
print("Test MAE:", mean_absolute_error(y_test, poly_predictions))

# Lasso Regression Model (L1 regularization)
ls = Lasso()
ls.fit(x_train_scaled, y_train)

lasso_pred = ls.predict(x_test_scaled)

# Evaluate Lasso Regression
print("Lasso Regression R^2 Score:", r2_score(y_test, lasso_pred))
print("MSE:", mean_squared_error(y_test, lasso_pred))
print("MAE:", mean_absolute_error(y_test, lasso_pred))

# Ridge Regression Model (L2 regularization)
rd = Ridge()
rd.fit(x_train_scaled, y_train)

ridge_pred = rd.predict(x_test_scaled)

# Evaluate Ridge Regression
print("Ridge Regression R^2 Score:", r2_score(y_test, ridge_pred))
print("MSE:", mean_squared_error(y_test, ridge_pred))
print("MAE:", mean_absolute_error(y_test, ridge_pred))

# Decision Tree Regressor
dt = DecisionTreeRegressor(max_depth=2)  # Limit tree depth to prevent overfitting
dt.fit(x_train, y_train)

# Make predictions with Decision Tree
dt_pred = dt.predict(x_test)

# Evaluate Decision Tree Regressor
print("Decision Tree R^2 Score:", r2_score(y_test, dt_pred))
print("MSE:", mean_squared_error(y_test, dt_pred))
print("MAE:", mean_absolute_error(y_test, dt_pred))

# Visualize Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(dt, filled=True, feature_names=x.columns, rounded=True)
plt.show()
