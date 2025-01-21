import numpy as np
from data import load_data
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

train, test = load_data()

X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.2, random_state=42)

y_test = y_test.values
y_train = y_train.values

kernel = RBF(length_scale=1.0) + Matern(length_scale=1.0, nu=1.5)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
gp.fit(X_train, y_train)

y_pred = gp.predict(X_test)

# print accuracy
print(rmse(y_test, y_pred))


lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)

# print accuracy
print(rmse(y_test, y_pred))


