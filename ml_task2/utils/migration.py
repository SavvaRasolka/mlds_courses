import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

from calculate_log_return import calculate_log_return


CSV_PATH = 'archive/tech_stock_prices_2020_to_today.csv'

def update_database():

    load_dotenv()
    engine = create_engine(os.getenv("ENGINE"))

    df = pd.read_csv(CSV_PATH, parse_dates=['Date'])

    df = df.drop(['Dividends',
            'Stock Splits',
            'P/E Ratio',
            'Market Cap',
            'Price/Sales Ratio',
            'Price/Book Ratio',
            'Dividend Yield',
            'Daily Return',
            '20-Day MA',
            'index'],
            axis=1)
    df.columns = df.columns.str.lower()
    df.rename(columns={'adj close': 'adj_close'}, inplace=True)

    
    
    for ticker in df['ticker'].unique():
        print(ticker)
        temp_df = df[df['ticker']==ticker]
        # print(temp_df.head())
        temp_df.drop('ticker', axis=1, inplace=True)
        temp_df = calculate_log_return(temp_df)
        
        temp_df.to_sql(ticker,
                        engine,
                        if_exists='replace',
                        index=False,)

update_database()
