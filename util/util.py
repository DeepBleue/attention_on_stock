import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler


def add_all_feature(df):
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    weights_20 = np.arange(1,21)
    df['WMA_20'] = df['Close'].rolling(window=20).apply(lambda prices: np.dot(prices, weights_20)/weights_20.sum(), raw=True)

    # Price Momentum
    df['ROC_12'] = df['Close'].pct_change(periods=12)
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)

    # Bollinger Bands
    df['Middle_Band_20'] = df['Close'].rolling(window=20).mean()
    df['Upper_Band_20'] = df['Middle_Band_20'] + df['Close'].rolling(window=20).std() * 2
    df['Lower_Band_20'] = df['Middle_Band_20'] - df['Close'].rolling(window=20).std() * 2

    # Historical Volatility
    df['Historical_Volatility_20'] = df['Close'].pct_change().rolling(window=20).std()
    
    # On-Balance Volume
    df['OBV'] = np.where(df['Close'] > df['Close'].shift(1), df['Volume'], np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0)).cumsum()
    
    # Volume Weighted Average Price
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    
    # Relative Strength Index
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Stochastic Oscillator
    low_min  = df['Low'].rolling(window=14, min_periods=1).min()
    high_max = df['High'].rolling(window=14, min_periods=1).max()
    
    df['%K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['%D'] = df['%K'].rolling(window=3).mean()
    
    # Commodity Channel Index
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    moving_avg = typical_price.rolling(window=20).mean()
    mean_deviation = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    df['CCI'] = (typical_price - moving_avg) / (0.015 * mean_deviation)
    
    # ADX
    df['TR'] = df['High'].combine(df['Low'], func=lambda x1, x2: x1 - x2)  # True Range
    df['+DM'] = df['High'].diff()  # Positive Directional Movement
    df['-DM'] = df['Low'].diff()  # Negative Directional Movement
    
    df['+DM'].where(df['+DM'] > 0, 0, inplace=True)
    df['-DM'].where(df['-DM'] > 0, 0, inplace=True)

    df['TR'] = df['TR'].rolling(window=14).sum()
    df['+DM'] = df['+DM'].rolling(window=14).sum()
    df['-DM'] = df['-DM'].rolling(window=14).sum()
    
    df['+DI'] = 100 * df['+DM'] / df['TR']
    df['-DI'] = 100 * df['-DM'] / df['TR']
    df['ADX'] = 100 * abs((df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])).rolling(window=14).mean()

    # Aroon Oscillator
    df['Aroon_Up'] = df['High'].rolling(window=25).apply(lambda x: x.index.get_loc(x.idxmax()), raw=False) / 25 * 100
    df['Aroon_Down'] = df['Low'].rolling(window=25).apply(lambda x: x.index.get_loc(x.idxmin()), raw=False) / 25 * 100
    df['Aroon_Oscillator'] = df['Aroon_Up'] - df['Aroon_Down']


    # VROC
    df['VROC'] = df['Volume'].pct_change(periods=5) * 100

    # Drop temporary columns used for calculations
    df.drop(['+DM', '-DM', 'TR', 'Aroon_Up', 'Aroon_Down'], axis=1, inplace=True)

    return df 



def normalize_data_per_ticker_new(data):
    min_max_scaler = MinMaxScaler()
    z_score_scaler = StandardScaler()
    max_abs_scaler = MaxAbsScaler()

    feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'Rate of Change', 'SMA_20',
                     'SMA_50', 'SMA_100', 'SMA_200', 'EMA_12', 'EMA_26', 'WMA_20', 'ROC_12',
                     'Momentum_10', 'Middle_Band_20', 'Upper_Band_20', 'Lower_Band_20',
                     'Historical_Volatility_20', 'OBV', 'VWAP', 'RSI', 'MACD', 'Signal_Line',
                     '%K', '%D', 'CCI', '+DI', '-DI', 'ADX', 'Aroon_Oscillator', 'VROC',
                     'BPS', 'PER', 'PBR', 'EPS', 'DIV', 'DPS', 'Institutional',
                     'Other Corporations', 'Individual', 'Foreigners']
    
    features_normalize_together_min_max = ['Open', 'High', 'Low', 'Close', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200', 'EMA_12', 'EMA_26', 'WMA_20', 'Middle_Band_20', 'Upper_Band_20', 'Lower_Band_20']
    features_normalize_self_min_max  = ['VWAP','Volume','Historical_Volatility_20']
    # features_normalize_self_abs_max= ['Rate of Change','OBV','VROC','ROC_12','Momentum_10','MACD','Signal_Line','BPS', 'PER', 'PBR', 'EPS', 'DIV', 'DPS','Institutional', 'Other Corporations', 'Individual', 'Foreigners']
    features_normalize_100 = ['RSI', '%K', '%D','CCI','+DI', '-DI','ADX','Aroon_Oscillator']
    z_score_features = ['BPS', 'PER', 'PBR', 'EPS', 'DIV', 'DPS','Rate of Change','OBV','VROC','ROC_12','Momentum_10','MACD','Signal_Line','Institutional', 'Other Corporations', 'Individual', 'Foreigners']
    
    # remove_feature = ['BPS', 'PER', 'PBR', 'EPS', 'DIV', 'DPS']
    
    combined_indices = [feature_names.index(feat) for feat in features_normalize_together_min_max]
    
    
    # for i, feature in enumerate(feature_names):
    #     if feature in normalize_btw_tickers : 
    #         transformed_data = min_max_scaler.fit_transform(data[:, :, i].reshape(-1, 1))
    #         data[:, :, i] = transformed_data.reshape(data[:, :, i].shape)

    
    
    for ticker in range(data.shape[0]):  # iterating over tickers
        # Scaling combined features together
        combined_data = data[ticker, :, combined_indices]
        combined_data_reshaped = combined_data.reshape(-1, len(combined_indices))
        combined_data_scaled = min_max_scaler.fit_transform(combined_data_reshaped)
        data[ticker, :, combined_indices] = combined_data_scaled.reshape(data[ticker, :, combined_indices].shape)
        
        # Scaling other features
        for i, feature in enumerate(feature_names):
            if feature in features_normalize_self_min_max:
                data[ticker, :, i] = min_max_scaler.fit_transform(data[ticker, :, i].reshape(-1, 1)).flatten()
                
            elif feature in features_normalize_100:
                data[ticker, :, i] /= 100.0
                
            elif feature in z_score_features : 
                data[ticker, :, i] = z_score_scaler.fit_transform(data[ticker, :, i].reshape(-1, 1)).flatten()
    
    # Identifying indices of features to remove
    # remove_indices = [feature_names.index(feat) for feat in remove_feature if feat in feature_names]

    # Removing the features
    # data = np.delete(data, remove_indices, axis=2)
    
    return data
                


def normalize_data_per_ticker_one(data):
    min_max_scaler = MinMaxScaler()
    z_score_scaler = StandardScaler()
    max_abs_scaler = MaxAbsScaler()

    feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'Rate of Change', 'SMA_20',
                     'SMA_50', 'SMA_100', 'SMA_200', 'EMA_12', 'EMA_26', 'WMA_20', 'ROC_12',
                     'Momentum_10', 'Middle_Band_20', 'Upper_Band_20', 'Lower_Band_20',
                     'Historical_Volatility_20', 'OBV', 'VWAP', 'RSI', 'MACD', 'Signal_Line',
                     '%K', '%D', 'CCI', '+DI', '-DI', 'ADX', 'Aroon_Oscillator', 'VROC',
                     'BPS', 'PER', 'PBR', 'EPS', 'DIV', 'DPS', 'Institutional',
                     'Other Corporations', 'Individual', 'Foreigners']
    
    features_normalize_together_min_max = ['Open', 'High', 'Low', 'Close', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200', 'EMA_12', 'EMA_26', 'WMA_20', 'Middle_Band_20', 'Upper_Band_20', 'Lower_Band_20']
    features_normalize_self_min_max  = ['VWAP','Volume','Historical_Volatility_20']
    features_normalize_100 = ['RSI', '%K', '%D','CCI','+DI', '-DI','ADX','Aroon_Oscillator']
    z_score_features = ['BPS', 'PER', 'PBR', 'EPS', 'DIV', 'DPS','Rate of Change','OBV','VROC','ROC_12','Momentum_10','MACD','Signal_Line','Institutional', 'Other Corporations', 'Individual', 'Foreigners']
    
    combined_indices = [feature_names.index(feat) for feat in features_normalize_together_min_max]
    


    combined_data = data[:, combined_indices]
    combined_data_reshaped = combined_data.reshape(-1, len(combined_indices))
    combined_data_scaled = min_max_scaler.fit_transform(combined_data_reshaped)
    data[:, combined_indices] = combined_data_scaled.reshape(data[:, combined_indices].shape)
    
    # Scaling other features
    for i, feature in enumerate(feature_names):
        if feature in features_normalize_self_min_max:
            data[:, i] = min_max_scaler.fit_transform(data[:, i].reshape(-1, 1)).flatten()
            
        elif feature in features_normalize_100:
            data[:, i] /= 100.0
            
        elif feature in z_score_features : 
            data[:, i] = z_score_scaler.fit_transform(data[:, i].reshape(-1, 1)).flatten()

    # Identifying indices of features to remove
    # remove_indices = [feature_names.index(feat) for feat in remove_feature if feat in feature_names]

    # Removing the features
    # data = np.delete(data, remove_indices, axis=2)
    
    return data
                



import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def is_stationary(ticker_series, significance_level=0.05):
    """
    Check if a given time series is stationary using the Augmented Dickey-Fuller test.

    Parameters:
    - ticker_series: Time series data (like stock prices)
    - significance_level: Threshold p-value to determine stationarity

    Returns:
    - True if series is stationary, False otherwise
    """
    
    # Perform ADF test
    adf_result = adfuller(ticker_series)
    
    # Extract p-value from test results
    p_value = adf_result[1]
    
    if p_value < significance_level:
        # p-value is less than significance level, reject null hypothesis and consider series stationary
        return True
    else:
        # p-value is greater, cannot reject null hypothesis and consider series non-stationary
        return False

# Sample usage
if __name__ == "__main__":
    # Generate a sample random walk which is typically non-stationary
    np.random.seed(0)
    random_walk = np.random.randn(1000).cumsum()
    
    print(is_stationary(random_walk))  # Expected: False


def rename_columns(df):
    # Rename columns
    column_mapping = {
        "시가": "Open",
        "고가": "High",
        "저가": "Low",
        "종가": "Close",
        "거래량": "Volume",
        "등락률": "Rate of Change"
    }
    df.rename(columns=column_mapping, inplace=True)
    
    # Rename index
    df.index.name = 'Date'
    
    return df


def rename_to_english(df):
    # Drop the '전체' column if it exists
    if '전체' in df.columns:
        df.drop(columns=['전체'], inplace=True)

    # Rename the columns and the index name
    column_mapping = {
        '기관합계': 'Institutional',
        '기타법인': 'Other Corporations',
        '개인': 'Individual',
        '외국인합계': 'Foreigners'
    }
    df.columns = [column_mapping.get(col, col) for col in df.columns]
    df.index.name = 'Date'
    
    return df