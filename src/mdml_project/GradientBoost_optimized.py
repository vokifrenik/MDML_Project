import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from Coulomb import *
import joblib
from xgboost import XGBRegressor
import optuna
from optuna.samplers import TPESampler
import warnings

# Suppress FutureWarnings and Optuna verbosity
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# Assuming X and y are already defined as in your code
# X = pd.DataFrame(data = cmats, index=train.id)
# y = train['hform']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#print(X_PCA.shape)
print(X_train.shape)

# Define the Optuna objective function
def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
    }

    model = XGBRegressor(objective='reg:squarederror', random_state=42, **param)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

# Use TPE sampler for Bayesian optimization
sampler = TPESampler(seed=42)  # Ensures reproducibility
study = optuna.create_study(direction='minimize', sampler=sampler)

# Optimize the objective function with increased trials
study.optimize(objective, n_trials=100)  # Increased from 50 to 200

# Get the best parameters and RMSE
best_params = study.best_params
print(f"Best Parameters: {best_params}")
print(f"Best RMSE: {study.best_value}")

# Train the model with the best parameters
best_model = XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
best_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = best_model.predict(X_test_scaled)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Optimized XGBoost MAE: {mae}, RMSE: {rmse}")

# Save the best model
model_filename = 'xgboost_model_coulomb_new.joblib'
joblib.dump(best_model, model_filename)
print(f"Model saved as {model_filename}")


model_filename = '"C:/1uni/ML for material design/Kaggle_competition/MDML_Project/src/mdml_project/xgboost_model_coulomb_second_try_saving.joblib'
joblib.dump(best_model, model_filename)
print(f"Model saved to directory as {model_filename}")


# Feature importance
feature_importance = best_model.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(12, 6))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, sorted_idx)
plt.xlabel('Feature Importance')
plt.title('Feature Importance (MDI)')
plt.tight_layout()
plt.show()

