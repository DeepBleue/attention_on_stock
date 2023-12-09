from pykrx import stock
from datetime import datetime
from util import *
from tqdm import tqdm 
import pickle
import numpy as np



init_year = 2012
final_year = 2023

market = "KOSDAQ"

folder_root = f"./data/{market}/"


################ 삼성으로 데이터 길이 동일하게 만들기 
samsung = '005930'
sample_date_len = []
for year in range(init_year,final_year):
    
    start_year = year
    end_year = year + 1
    start_date = str(start_year) + "0201"
    end_date = str(end_year) + "1231"
    df = stock.get_market_ohlcv(start_date, end_date, samsung)
    sample_date_len.append(len(df))

################# 모든 데이터에 대해서 데이터 얻기 


for idx,year in enumerate(range(init_year,final_year)):
    
    start_year = year
    end_year = year + 1
    
    start_date = str(start_year) + "0201"
    end_date = str(end_year) + "1231"
    
    tickers = stock.get_market_ticker_list(end_date, market=market)
    # tickers = tickers[:20]
    names = [stock.get_market_ticker_name(ticker) for ticker in tickers]
    
    date_len = sample_date_len[idx]

    all_data = []
    valid_ticker = {}

    
    print(f'-----  START {end_year} {market} DATA -----')

    for name , ticker in tqdm(zip(names,tickers),total=len(names)):


        ##################   GET DATA   ##################
        try : 
            df = stock.get_market_ohlcv(start_date, end_date, ticker)
        except : 
            continue
        if len(df) != date_len : 
            continue
        dates = df.index
        df = rename_columns(df)
        df = add_all_feature(df)
        df_fdm = stock.get_market_fundamental(start_date, end_date, ticker, freq="d")
        df_tra_vol_all = stock.get_market_trading_volume_by_date(start_date, end_date, ticker,on='순매수')
        df_tra_vol_all = rename_to_english(df_tra_vol_all)
        merged_df_1 = df.merge(df_fdm, left_index=True, right_index=True, how='outer')
        merged_df_2 = merged_df_1.merge(df_tra_vol_all, left_index=True, right_index=True, how='outer')
        df = merged_df_2
        
        
        if len(df.columns) != 42 : 
            print(f'ticker : {ticker} , name : {name} has no 42 features.')
            continue
        
        df = df[df.index.year == end_year]    
        
        
        # Check for NaN or infinite values in the dataframe
        if df.isnull().any().any() or np.isinf(df.values).any():
            print(f'ticker : {ticker} , name : {name} has Nan or Inf value.')
            continue


        dates = df.index    
        np_df = np.array(df)
        all_data.append(np_df)
        valid_ticker[ticker] = name


    all_data = np.array(all_data)
    print(all_data.shape)

    with open(folder_root + f'{market}_{str(end_year)}.pkl', 'wb') as f:
        pickle.dump({'data': all_data, 
                     'valid_ticker': valid_ticker,
                     'dimension' : ('tickers','time','features'),
                     'features' : df.columns,
                     'dates': dates}
                    , f)




