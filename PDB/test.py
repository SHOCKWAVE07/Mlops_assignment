from sklearn.linear_model import LinearRegression
import numpy as np
import pdb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

def train_model(X, y, degree=2):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)  
    model = LinearRegression()
    model.fit(X_poly, y)
    return model, poly 

def evaluate_model(model, poly, X_test, y_test):
    X_test_poly = poly.transform(X_test)  
    predictions = model.predict(X_test_poly)
    mse = mean_squared_error(y_test, predictions)
    return mse

X_train = np.array([[1], [2], [3], [4]])
y_train = np.array([1, 4, 9, 16])

trained_model, poly = train_model(X_train, y_train)

X_test = np.array([[5], [6], [7], [8]])
y_test = np.array([25, 36, 49, 64])

mse_score = evaluate_model(trained_model, poly, X_test, y_test)
print("Mean Squared Error:", mse_score)
