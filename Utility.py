import os
import pickle
import pandas as pd
from pandas import read_excel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Directory management
working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

os.makedirs(f"{parent_dir}/trained_model", exist_ok=True)
os.makedirs(f"{parent_dir}/data", exist_ok=True)

# Read data
def read_data(filename):
    file_path = f"{parent_dir}/data/{filename}"
    try:
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        elif file_path.endswith((".xlsx", ".xls")):
            return read_excel(file_path)
        else:
            raise ValueError("Unsupported file format.")
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# Preprocess data
def preprocess_data(df):
    x = df.iloc[:, :-1]
    y = df['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test, scaler

# Train a single model
def train_model(x_train, y_train, model, model_name):
    model.fit(x_train, y_train)
    with open(f"{parent_dir}/trained_model/{model_name}.pkl", "wb") as file:
        pickle.dump(model, file)
    return model

# Evaluate a model
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    return round(accuracy_score(y_test, y_pred), 2)

# Train and compare multiple models
def train_all_models(x_train, x_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier()
    }
    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        results[name] = round(accuracy_score(y_test, y_pred), 2)
    return results

# Predict user values
def user_value_prediction(features, model):
    return model.predict(features)
