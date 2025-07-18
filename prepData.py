## pd.df in, 
import pandas as pd
from os.path import join, exists


def clean_data(df:pd.DataFrame):
    df.dropna(subset=['text'], inplace=True,print_msg=True)
    df['text'] = df['text'].astype(str)



def clean_doc
