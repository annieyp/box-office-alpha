import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

#research XGBoost hyperparameters for time series data and implement them here
def train_model(X_train, X_test, y_train, y_test, stock):
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"RMSE for {stock}: {rmse}")

    return preds 

