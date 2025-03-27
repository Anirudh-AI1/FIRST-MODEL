import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


file_path = r'C:\Users\Dell\Desktop\diabetes_prediction_dataset.csv'
df = pd.read_csv(file_path)


df = df.drop_duplicates()


le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])  
df['smoking_history'] = le.fit_transform(df['smoking_history'])


print(df.isnull().sum())  
print(df.tail())


x = df.drop(columns=['diabetes', 'bmi', 'smoking_history'], axis=1)
y = df['bmi']


scaler = StandardScaler()
x = scaler.fit_transform(x)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}


rf = RandomForestRegressor(random_state=42)


random_search = RandomizedSearchCV(
    estimator=rf, 
    param_distributions=param_dist,
    n_iter=10, 
    cv=3, 
    n_jobs=1, 
    verbose=2, 
    scoring='r2',
    random_state=42
)


random_search.fit(x_train, y_train)


print("Best Parameters:", random_search.best_params_)
print("Best R² Score from CV:", random_search.best_score_)


best_model = random_search.best_estimator_
y_pred = best_model.predict(x_test)


print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
