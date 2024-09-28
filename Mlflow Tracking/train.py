from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from category_encoders import TargetEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import mlflow
import pandas as pd
import numpy as np

df = pd.read_csv("data/hour.csv")

print(df.head)

df['avg_temp'] = (df['atemp'] + df['temp'])/2
df['heat_index'] = 0.5*df['avg_temp'] + 0.5*df['hum']
df.drop(['atemp','temp','hum'],axis=1,inplace=True)

df['day_night'] = df['hr'].apply(lambda x: 'day' if 6<=x<=18 else 'night')

df.drop(['instant', 'casual', 'registered'], axis=1, inplace=True)

df['dteday'] = pd.to_datetime(df.dteday)
df['season'] = df.season.astype('category')
df['holiday'] = df.holiday.astype('category')
df['weekday'] = df.weekday.astype('category')
df['weathersit'] = df.weathersit.astype('category')
df['workingday'] = df.workingday.astype('category')
df['mnth'] = df.mnth.astype('category')
df['yr'] = df.yr.astype('category')
df['hr'] = df.hr.astype('category')

df.drop(columns=['dteday'], inplace=True)

X = df.drop(['cnt'],axis=1).copy()
y = df['cnt']

print("Train Test Split")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


numerical_features = ['avg_temp', 'heat_index', 'windspeed']
categorical_features = ['season', 'weathersit', 'day_night']


numerical_transformer = MinMaxScaler()
categorical_transformer = TargetEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


model = LinearRegression()

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

print("Starting Linear Regression")


mlflow.set_tracking_uri("http://localhost:5000")


with mlflow.start_run():

    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mlflow.log_metric("mse", mse)

    r2 = r2_score(y_test, y_pred)
    mlflow.log_metric("r2", r2)
    
    mlflow.sklearn.log_model(pipeline, "Linear Regression")
    
mlflow.end_run()

rf = RandomForestRegressor()

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf)
])


mlflow.set_tracking_uri("http://localhost:5000")


with mlflow.start_run():

    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mlflow.log_metric("mse", mse)

    r2 = r2_score(y_test, y_pred)
    mlflow.log_metric("r2", r2)
    
    mlflow.sklearn.log_model(pipeline, "Random Forest Regression")
    
mlflow.end_run()