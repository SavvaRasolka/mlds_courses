import matplotlib.pyplot as plt
import pandas as pd
from load_data import load_data


def plot_data(table_name, column):
    df = load_data(table_name)
    plt.plot(df["date"], df[column])
    plt.title(table_name)
    plt.xlabel("date")
    plt.ylabel(column)
    plt.show()
    # plt.savefig('images/{}_log_return.png'.format(table_name))
    # plt.grid(True)
    # plt.xticks(rotation=45)
    # plt.tight_layout()

for company in ['AAPL', 'AMZN', 'GOOGL','IBM','INTC','META','MSFT','NVDA','ORCL','TSLA']:
    plot_data(company, 'volume')