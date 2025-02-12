from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import logging
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score
import pickle
from flask import send_file
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

app = Flask(__name__)

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/app.log", level=logging.INFO, format="%(asctime)s - %(message)s")


def handle_missing_values(df, threshold=0.4, logs=[]):
    for col in df.columns:
        missing_ratio = df[col].isnull().mean()
        if missing_ratio > threshold:
            df.drop(columns=[col], inplace=True)
            logs.append(f"Dropped column {col} due to high missing values.")
        else:
            if df[col].dtype == "object":
                df[col].fillna(df[col].mode()[0], inplace=True)
                logs.append(f"Filled missing values in {col} with mode.")
            else:
                df[col].fillna(df[col].mean(), inplace=True)
                logs.append(f"Filled missing values in {col} with mean.")
    return df, logs

def handle_outliers(df, logs=[]):
    for col in df.select_dtypes(include=[np.number]).columns:
        mean, std = df[col].mean(), df[col].std()
        upper, lower = mean + 3 * std, mean - 3 * std
        df[col] = np.where((df[col] > upper) | (df[col] < lower), mean, df[col])
        logs.append(f"Replaced outliers in {col} with mean.")
    return df, logs

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/download_model")
def download_model():
    # Ensure the file exists before sending
    best_model_path = "models/best_model.pkl"
    return send_file(best_model_path, as_attachment=True)

@app.route("/process", methods=["POST"])
def process():
    file = request.files["file"]
    target = request.form["target"]
    task = request.form["task"]
    df = pd.read_csv(file)
    logs = []

    if target not in df.columns:
        return "Error: Target variable not found in dataset."

    logs.append(f"Processing dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

    df, logs = handle_missing_values(df, logs=logs)
    df, logs = handle_outliers(df, logs=logs)

    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
    if target in categorical_features:
        categorical_features.remove(target)

    label_enc = LabelEncoder()
    onehot_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    for col in categorical_features:
        if df[col].nunique() <= 10:
            df[col] = label_enc.fit_transform(df[col])
            logs.append(f"Applied Label Encoding to {col}.")
        else:
            onehot_encoded = pd.DataFrame(onehot_enc.fit_transform(df[[col]]))
            df = df.drop(columns=[col]).join(onehot_encoded)
            logs.append(f"Applied One-Hot Encoding to {col}.")

    X = df.drop(columns=[target])
    y = df[target]
    
        # --- FEATURE SELECTION USING SelectKBest (performed BEFORE model training) ---
    if task == "classification":
        selector = SelectKBest(score_func=f_classif, k=5)  # Adjust k as needed
    else:
        selector = SelectKBest(score_func=f_regression, k=5)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    logs.append(f"Selected top features: {list(selected_features)}")
    # Update X to only include the selected features
    X = pd.DataFrame(X_new, columns=selected_features)

    if task == "classification":
        if y.dtypes == "object":
            y = label_enc.fit_transform(y)
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "Naïve Bayes": GaussianNB(),
            "AdaBoost": AdaBoostClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "XGBoost": XGBClassifier(),
        }
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "SVM Regressor": SVR(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "Elastic Net": ElasticNet(),
            "XGBoost Regressor": XGBRegressor(),
        }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logs.append("Applied Standard Scaling to data.")

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if task == "classification":
            acc = accuracy_score(y_test, y_pred) * 100
            results[name] = {"Accuracy": round(acc, 2)}  # ✅ Changed "Score" to "Accuracy"
        else:
            mse = mean_squared_error(y_test, y_pred)
            results[name] = {"MSE": round(mse, 2)}  # ✅ Changed "Score" to "MSE"

        logs.append(f"Trained {name} model.")

    if task == "classification":
        sorted_models = sorted(results.items(), key=lambda x: x[1]["Accuracy"], reverse=True)
        top_models = sorted_models[:3]  # Ensure exactly top 3 models are selected
    else:
        sorted_models = sorted(results.items(), key=lambda x: x[1]["MSE"], reverse=False)
        top_models = sorted_models[:3]  # Ensure exactly top 3 models are selected

    tuned_models = {}
    
    param_grid = {
        "Logistic Regression": {"C": [0.1, 1, 10]},
        "Random Forest": {"n_estimators": [50, 100, 200]},
        "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
        "Decision Tree": {"max_depth": [None, 10, 20, 30]},
        "KNN": {"n_neighbors": [3, 5, 7]},
        "Naïve Bayes": {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
        "AdaBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1]},
        "Gradient Boosting": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
        "XGBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
        "Linear Regression": {"fit_intercept": [True, False], },
        "Random Forest Regressor": {"n_estimators": [50, 100, 200]},
        "SVM Regressor": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
        "Decision Tree Regressor": {"max_depth": [None, 10, 20, 30]},
        "Ridge": {"alpha": [0.1, 1, 10]},
        "Lasso": {"alpha": [0.1, 1, 10]},
        "Elastic Net": {"alpha": [0.1, 1, 10], "l1_ratio": [0.1, 0.5, 0.9]},
        "XGBoost Regressor": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
    }

    for model_name, _ in top_models[:3]:  # Ensure exactly top 3 models are tuned
        model = models[model_name]
        if model_name in param_grid and param_grid[model_name]:  # Check if hyperparameters exist
            grid_search = GridSearchCV(model, param_grid[model_name], cv=5, scoring="accuracy" if task == "classification" else "r2")
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            # **Calculate the new score after tuning**
            y_pred_tuned = best_model.predict(X_test)
            if task == "classification":
                acc_tuned = accuracy_score(y_test, y_pred_tuned) * 100
                results[model_name] = {"Accuracy (Tuned)": round(acc_tuned, 2)}  # ✅ Updated key to distinguish tuned results
            else:
                mse_tuned = mean_squared_error(y_test, y_pred_tuned)
                results[model_name] = {"MSE (Tuned)": round(mse_tuned, 2)}  # ✅ Updated key to distinguish tuned results

            logs.append(f"Updated {model_name} score after tuning: {results[model_name]}")

            best_params = grid_search.best_params_
            tuned_models[model_name] = {"model": best_model, "params": best_params}
            logs.append(f"Tuned hyperparameters for {model_name}: {best_params}")

    logging.info("\n".join(logs))
    if tuned_models:
        best_model_name = list(tuned_models.keys())[:3]  # Ensure top 3 models are displayed
        best_params = {model: tuned_models[model]["params"] for model in best_model_name}
    else:
        best_model_name = ["No model tuned"]
        best_params = {"N/A": "N/A"}
    
    # Save the best tuned model as a .pkl file for download
    os.makedirs("models", exist_ok=True)
    if tuned_models:
        # Use the best tuned model (we already computed best_tuned_model in the hyperparameter tuning block)
        if task == "classification":
            tuned_scores = {model: results[model]["Accuracy (Tuned)"] for model in tuned_models.keys() if "Accuracy (Tuned)" in results[model]}
            best_tuned_model_name = max(tuned_scores, key=tuned_scores.get)
        else:
            tuned_scores = {model: results[model]["MSE (Tuned)"] for model in tuned_models.keys() if "MSE (Tuned)" in results[model]}
            best_tuned_model_name = min(tuned_scores, key=tuned_scores.get)
        logs.append(f"Best tuned model selected: {best_tuned_model_name}")
        best_model_obj = tuned_models[best_tuned_model_name]["model"]
        best_model_filename = "models/best_model.pkl"
        with open(best_model_filename, "wb") as f:
            pickle.dump(best_model_obj, f)
    else:
        best_tuned_model_name = "No model tuned"
        best_model_filename = None

        
    if tuned_models:
        if task == "classification":
        # Build a dictionary of tuned accuracy scores for each tuned model
            tuned_scores = {model: results[model]["Accuracy (Tuned)"] for model in tuned_models.keys() if "Accuracy (Tuned)" in results[model]}
            best_tuned_model = max(tuned_scores, key=tuned_scores.get)
        else:
        # Build a dictionary of tuned MSE values for each tuned model
            tuned_scores = {model: results[model]["MSE (Tuned)"] for model in tuned_models.keys() if "MSE (Tuned)" in results[model]}
            best_tuned_model = min(tuned_scores, key=tuned_scores.get)
        logs.append(f"Best tuned model selected: {best_tuned_model}")
    else:
        best_tuned_model = "No model tuned"

        
    # Check if the task is classification or regression
    if task == "classification":
        y_pred = best_model.predict(X_test)
        final_score = accuracy_score(y_test, y_pred)
        metric_name = "Accuracy"
    else:  # Regression case
        y_pred = best_model.predict(X_test)
        final_score = mean_squared_error(y_test, y_pred)
        metric_name = "MSE"

    # Store the results
    final_results = { "Best Model": best_model_name, metric_name: final_score }

    return render_template("result.html", logs=logs, results=results, best_models=best_model_name, best_params=best_params, best_tuned_model=best_tuned_model_name, download_model=best_model_filename, selected_features=list(selected_features))

if __name__ == "__main__":
    app.run(debug=True)
