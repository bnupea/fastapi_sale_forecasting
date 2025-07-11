import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, 'data', 'Advertising Budget and Sales.csv')
model_dir = os.path.join(base_dir, '..', 'models')
os.makedirs(model_dir, exist_ok=True)

scaler_path = os.path.join(model_dir, 'scaler.joblib')
poly_path   = os.path.join(model_dir, 'poly.joblib')
model_path  = os.path.join(model_dir, 'sales_model_ridge.joblib')

raw_df = pd.read_csv(data_path)
df = raw_df.rename(columns={
    'TV Ad Budget ($)': 'TV',
    'Radio Ad Budget ($)': 'Radio',
    'Newspaper Ad Budget ($)': 'Newspaper',
    'Sales ($)': 'Sales'
})

drop_cols = ['Unnamed: 0'] if 'Unnamed: 0' in df.columns else []
X = df.drop(columns=drop_cols + ['Sales'])
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('ridge', RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5))
])

pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2  = r2_score(y_test, preds)
print(f"Test MSE: {mse:.4f}")
print(f"Test R²: {r2:.4f}")

joblib.dump(pipeline.named_steps['scaler'], scaler_path)
joblib.dump(pipeline.named_steps['poly'],   poly_path)
joblib.dump(pipeline, model_path)

import matplotlib.pyplot as plt

plt.scatter(y_test, preds, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.grid(True)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()