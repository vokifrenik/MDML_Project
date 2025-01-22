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

# Assuming X and y are already defined as in your code
# X = pd.DataFrame(data = cmats, index=train.id)
# y = train['hform']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_PCA, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_PCA.shape)
print(X_train.shape)

# Define the model
gb_model = GradientBoostingRegressor(random_state=42)

# Define hyperparameters for grid search
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 4],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search
grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_absolute_error')
grid_search.fit(X_train_scaled, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test_scaled)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Best parameters: {grid_search.best_params_}")
print(f"MAE: {mae}, RMSE: {rmse}")


# Save the best model
model_filename = 'best_gradient_boosting_model_soap.joblib'
joblib.dump(best_model, model_filename)
print(f"Model saved as {model_filename}")

# To load the model later, you can use:
# loaded_model = joblib.load(model_filename)


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
