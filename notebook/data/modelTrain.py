# =========================
#        IMPORTS
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Models
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


# =========================
#      LOAD DATA
# =========================

df = pd.read_csv("notebook/data/stud.csv")

# 🔥 Clean column names (VERY IMPORTANT)
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.replace("/", "_")

print("Columns in dataset:")
print(df.columns)
print("\nDataset Preview:")
print(df.head())


# =========================
#      SPLIT FEATURES
# =========================

X = df.drop(columns=["math_score"])
y = df["math_score"]


# =========================
#   NUMERIC & CATEGORICAL
# =========================

num_features = X.select_dtypes(exclude="object").columns
cat_features = X.select_dtypes(include="object").columns

print("\nNumeric Features:", num_features)
print("Categorical Features:", cat_features)


# =========================
#   PREPROCESSING PIPELINE
# =========================

numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", oh_transformer, cat_features),
        ("StandardScaler", numeric_transformer, num_features),
    ]
)

X = preprocessor.fit_transform(X)


# =========================
#   TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)


# =========================
#   EVALUATION FUNCTION
# =========================

def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2 = r2_score(true, predicted)
    return mae, rmse, r2


# =========================
#      MODELS
# =========================

models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "KNN": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "XGBoost": XGBRegressor(verbosity=0),
    "CatBoost": CatBoostRegressor(verbose=False),
    "AdaBoost": AdaBoostRegressor(),
}

model_list = []
r2_list = []


# =========================
#   TRAIN & EVALUATE
# =========================

for name, model in models.items():

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mae, train_rmse, train_r2 = evaluate_model(y_train, y_train_pred)
    test_mae, test_rmse, test_r2 = evaluate_model(y_test, y_test_pred)

    print(f"\n{name}")
    print("Training Performance:")
    print(f"RMSE: {train_rmse:.4f}")
    print(f"MAE: {train_mae:.4f}")
    print(f"R2 Score: {train_r2:.4f}")

    print("Test Performance:")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"R2 Score: {test_r2:.4f}")

    print("=" * 40)

    model_list.append(name)
    r2_list.append(test_r2)


# =========================
#   MODEL COMPARISON
# =========================

results = pd.DataFrame({
    "Model": model_list,
    "R2_Score": r2_list
}).sort_values(by="R2_Score", ascending=False)

print("\nModel Comparison:")
print(results)


# =========================
#   FINAL MODEL (Linear)
# =========================

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

y_pred = lin_model.predict(X_test)
score = r2_score(y_test, y_pred) * 100

print(f"\nAccuracy of Linear Regression Model: {score:.2f}%")


# =========================
#   VISUALIZATION
# =========================

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")

sns.regplot(x=y_test, y=y_pred, ci=None, color="red")

plt.show()


# =========================
#   PREDICTION DATAFRAME
# =========================

pred_df = pd.DataFrame({
    "Actual Value": y_test,
    "Predicted Value": y_pred,
    "Difference": y_test - y_pred
})

print("\nPrediction Comparison:")
print(pred_df.head())