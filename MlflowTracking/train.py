from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow
import pandas as pd

# Load the dataset
df = pd.read_csv("MlflowTracking/data/hour.csv")

# Feature Engineering: Adding new features and removing unnecessary columns
df['avg_temp'] = (df['atemp'] + df['temp']) / 2
df['heat_index'] = 0.5 * df['avg_temp'] + 0.5 * df['hum']
df['day_night'] = df['hr'].apply(lambda x: 1 if 6 <= x <= 18 else 0)

# Dropping unnecessary columns
df.drop(['instant', 'casual', 'registered', 'atemp', 'temp', 'hum', 'dteday'], axis=1, inplace=True)

# Splitting into features (X) and target variable (y)
X = df.drop(['cnt'], axis=1)
y = df['cnt']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")


# Function to log model and metrics with MLflow
def log_model_with_mlflow(model, model_name):
    with mlflow.start_run():
        # Train the model
        model.fit(X_train, y_train)

        # Predict on test data
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        # Log the model
        mlflow.sklearn.log_model(model, model_name)
        
        print(f"{model_name} logged with MSE: {mse} and R2: {r2}")
        
    mlflow.end_run()

# Linear Regression
linear_reg_model = LinearRegression()
log_model_with_mlflow(linear_reg_model, "Linear Regression")

# Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100)
log_model_with_mlflow(rf_model, "Random Forest Regression")
