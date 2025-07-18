## pd.df in, 
import pandas as pd
from os.path import join, exists
import re
import nltk
from utils import get_data_path


def clean_data(df:pd.DataFrame, save_as_file:bool=False, remove_stop_words:bool=True, remove_rare_words:int=5,label_col:str='label', text_col:str='text'):    
    df.dropna(subset=[f"{text_col}"], inplace=True)
    df[f"{text_col}"] = df[f"{text_col}"].astype(str)
    df[f"{text_col}"] = df[f"{text_col}"].str.lower()
    df[f"{text_col}"] = clean_doc(df[f"{text_col}"])
    if remove_stop_words:
        df[f"{text_col}"] = stop_words_removal(df[f"{text_col}"])   
    if save_as_file:
        df.to_csv(join(get_data_path(),df.Name,  'cleaned_data.csv'), index=False)
    else:
        return df





def clean_doc(df:pd.Series):
    remove_https = re.compile(r"http[s]?\:\/\/.[a-zA-Z0-9\.\/\_?=%&#\-\+!]+")
    remove_punct = re.compile(r'[^\w\s]')

    df = df.apply(lambda x: remove_https.sub(' ', x))
    df = df.apply(lambda x: remove_punct.sub(' ', x))
    df = df.apply(lambda x: x.split())
    
    return df
def stop_words_removal(df:pd.Series)->pd.Series:
    nltk.data.path.append(get_data_path())
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))
    print("Removing stopwords...")  # ← debug line
    return df.apply(lambda x: [word for word in x if word not in stop_words])


def encode_labels(df:pd.Series)-> pd.Series:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df = le.fit_transform(df)
    return df
    pass


if __name__ == "__main__":
    mr_all = pd.read_csv(join(get_data_path(), 'mr_all.csv'))
    mr_all = clean_data(mr_all)
    print(mr_all["text"].head(5),"\n")

    ng20 = pd.read_csv(join(get_data_path(), 'ng_20_all.csv'))
    ng20 = clean_data(ng20)
    ng = ng20.copy()
    ng["label"] = encode_labels(ng20["label"])
    print(ng20["label"].head(5),"\n",ng["label"].head(5),"\n")