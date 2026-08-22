import numpy as np

ANOMALY_DAY_INDEX=1158

def calculate_log_return(df):
    print(0 in df.index)
    df['log_return'] = np.log(df['adj_close'] / df['adj_close'].shift(1))
    df.loc[df.index[ANOMALY_DAY_INDEX], 'log_return'] = 0
    df.loc[df.index[0], 'log_return'] = 0
    print()
    print(df.iloc[0])
    print()
    print(df.iloc[1158])
    return df