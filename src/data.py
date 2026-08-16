import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def prepare_data(stock):
    df = yf.download(stock, start="2022-12-01", end="2026-08-14")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.tail()

    df.dropna(inplace=True) 
    scaler = StandardScaler()
    transformed_data = scaler.fit_transform(df)
    df = pd.DataFrame(transformed_data, columns=df.columns, index=df.index)
    #add outlier detection and removal here if needed

    df['Return'] = df['Close'].pct_change()
    df['SMA_5'] = df['Close'].rolling(5).mean() #when sma5 above sma10, momentum picking up
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['Lag_1'] = df['Close'].shift(1)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    return df

def split_data(df):
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return',
                'SMA_5', 'SMA_10', 'Lag_1']
    X = df[features]
    y = df['Target']

    #shuffle false very important for time series data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test


