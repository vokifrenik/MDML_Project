import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import optuna
from tqdm import tqdm
import joblib
from soap import *
import matplotlib.pyplot as plt
import warnings

# Suppress FutureWarnings and Optuna verbosity
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Assuming X and y are already defined
# X = pd.DataFrame(data = cmats, index=train.id)
# y = train['hform']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_PCA, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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

# Define the total number of trials
n_trials = 50

# Create an Optuna study
study = optuna.create_study(direction='minimize')

# Add a progress bar with tqdm
with tqdm(total=n_trials) as pbar:
    def progress_callback(study, trial):
        pbar.update(1)  # Increment the progress bar for each trial
    
    # Optimize the objective function
    study.optimize(objective, n_trials=n_trials, callbacks=[progress_callback])

# Print the best trial
print(f"Best RMSE: {study.best_value}")
print(f"Best Parameters: {study.best_params}")

# Train the model with the best parameters
best_model = XGBRegressor(objective='reg:squarederror', random_state=42, **study.best_params)
best_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = best_model.predict(X_test_scaled)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Optimized XGBoost MAE: {mae}, RMSE: {rmse}")

# Save the best model
model_filename = 'best_xgboost_model_soap_really_soap.joblib'
joblib.dump(best_model, model_filename)
print(f"Model saved as {model_filename}")


'''
COULOMB
Best RMSE: 0.2713756565430528
Best Parameters: {'n_estimators': 853, 'learning_rate': 0.05982426082021317, 'max_depth': 6, 'subsample': 0.7574623711307333, 'colsample_bytree': 0.7118702012607611, 'min_child_weight': 2, 'gamma': 0.0010030924140809118}
Optimized XGBoost MAE: 0.19617148462442216, RMSE: 0.2713756565430528
Model saved as best_xgboost_model_soap.joblib
'''

'''
SOAP
Best RMSE: 0.252081966276994
Best Parameters: {'n_estimators': 625, 'learning_rate': 0.047199336184485595, 'max_depth': 6, 'subsample': 0.7708029699037668, 'colsample_bytree': 0.7353240292920834, 'min_child_weight': 2, 'gamma': 1.8470905270118162e-06}
Optimized XGBoost MAE: 0.17878114697355113, RMSE: 0.252081966276994
Model saved as best_xgboost_model_soap_really_soap.joblib
'''