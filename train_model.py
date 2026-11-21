
import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------------
# CREATE REQUIRED FOLDERS
# --------------------------------

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# --------------------------------
# GENERATE STUDENT DATASET
# --------------------------------

np.random.seed(42)

number_of_students = 500

study_hours = np.random.uniform(1, 10, number_of_students)
attendance = np.random.uniform(40, 100, number_of_students)
previous_score = np.random.uniform(30, 100, number_of_students)

assignments_completed = np.random.randint(
    1, 11, number_of_students
)

# Generate simulated final scores
noise = np.random.normal(0, 5, number_of_students)

final_score = (
    0.30 * (study_hours / 10 * 100)
    + 0.25 * attendance
    + 0.30 * previous_score
    + 0.15 * (assignments_completed / 10 * 100)
    + noise
)

final_score = np.clip(final_score, 0, 100)

# Create DataFrame
data = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_score": previous_score,
    "assignments_completed": assignments_completed,
    "final_score": final_score
})

# Save dataset
data.to_csv(
    "data/student_performance.csv",
    index=False
)

print("Dataset created successfully!")
print("\nDataset Preview:")
print(data.head())

# --------------------------------
# FEATURES AND TARGET
# --------------------------------

X = data.drop("final_score", axis=1)
y = data["final_score"]

# --------------------------------
# TRAIN TEST SPLIT
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# --------------------------------
# HYPERPARAMETER TUNING
# --------------------------------

print("\nSearching for the best model configuration...")

model = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

parameters = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 8, None],
    "min_samples_leaf": [2, 4, 6]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=parameters,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Select best model
best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

# --------------------------------
# MODEL PREDICTIONS
# --------------------------------

train_predictions = best_model.predict(X_train)
test_predictions = best_model.predict(X_test)

# --------------------------------
# MODEL EVALUATION
# --------------------------------

train_r2 = r2_score(
    y_train,
    train_predictions
)

test_r2 = r2_score(
    y_test,
    test_predictions
)

mae = mean_absolute_error(
    y_test,
    test_predictions
)

mse = mean_squared_error(
    y_test,
    test_predictions
)

rmse = np.sqrt(mse)

cv_mae = -grid_search.best_score_

# --------------------------------
# DISPLAY RESULTS
# --------------------------------

print("\nMODEL EVALUATION RESULTS")
print("------------------------")

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("Testing R2:", round(test_r2, 4))
print("Training R2:", round(train_r2, 4))
print("Cross-validation MAE:", round(cv_mae, 2))

# --------------------------------
# SAVE EVALUATION METRICS
# --------------------------------

metrics = {
    "MAE": round(float(mae), 4),
    "RMSE": round(float(rmse), 4),
    "R2 Score": round(float(test_r2), 4),
    "Training R2": round(float(train_r2), 4),
    "Cross Validation MAE": round(float(cv_mae), 4),
    "Best Parameters": grid_search.best_params_
}

with open("models/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

# --------------------------------
# SAVE BEST MODEL
# --------------------------------

joblib.dump(
    best_model,
    "models/student_model.pkl"
)

print("\nBest model saved successfully!")
print("Evaluation metrics saved successfully!")
