import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv


def load_data(table_name):
    load_dotenv()
    conn = psycopg2.connect(
        dbname="ml_ds2",
        user=os.getenv("DB_USER"),
        password=os.getenv("PSWD"),
        host="localhost",
        port=os.getenv("PRT"),
        )
    
    query = 'SELECT * FROM public."{}";'.format(table_name)

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df