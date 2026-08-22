import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sqlalchemy import create_engine


companies = ['AAPL', 'AMZN', 'GOOGL','IBM','INTC','META','MSFT','NVDA','ORCL','TSLA']  
data_dict = {}

for company in companies:
    df = pd.read_sql(f'SELECT * FROM public."{company}" ORDER BY date', engine)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    data_dict[company] = df

fig, axes = plt.subplots(5, 2, figsize=(20, 15))
axes = axes.flatten()

for idx, (name, df) in enumerate(data_dict.items()):
    axes[idx].plot(df.index, df['close'], linewidth=1)
    axes[idx].set_title(f'{name} - Close Price')
    axes[idx].set_xlabel('date')
    axes[idx].set_ylabel('Price')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(5, 2, figsize=(20, 15))
axes = axes.flatten()

for idx, (name, df) in enumerate(data_dict.items()):
    axes[idx].plot(df.index, df['volume'], linewidth=1)
    axes[idx].set_title(f'{name} - volume')
    axes[idx].set_xlabel('date')
    axes[idx].set_ylabel('volume')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(5, 2, figsize=(20, 15))
axes = axes.flatten()

for idx, (name, df) in enumerate(data_dict.items()):
    df['Returns'] = np.log(df['close'] / df['close'].shift(1))
    axes[idx].hist(df['Returns'].dropna(), bins=50, alpha=0.7, edgecolor='black')
    axes[idx].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[idx].set_title(f'{name} - Log Returns Distribution')
    axes[idx].set_xlabel('Log Returns')
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
