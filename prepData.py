## pd.df in, 
import pandas as pd
from os.path import join, exists
import re
import nltk
from utils import get_data_path
import torch
from collections import Counter
import torch.nn.functional as F

def clean_data(df:pd.DataFrame, label_col:str='label', text_col:str='text',remove_stop_words:bool=True,remove_rare_words:int=True)-> list[pd.DataFrame,dict]:    
    df.dropna(subset=[f"{text_col}"], inplace=True)
    df.dropna(subset=[f"{label_col}"], inplace=True)
    df[f"{text_col}"] = df[f"{text_col}"].astype(str)
    df[f"{text_col}"] = df[f"{text_col}"].str.lower()
    df[f"{text_col}"] = clean_doc(df[f"{text_col}"])
    vocab = create_vocab(df[f"{text_col}"])
    if remove_stop_words:
        df[f"{text_col}"],vocab = stop_words_removal(df[f"{text_col}"], vocab)
    if remove_rare_words:
        df[f"{text_col}"],vocab = rare_words_removal(df[f"{text_col}"], vocab, min_freq=remove_rare_words)
    #df[f"{label_col}"] = encode_labels(df[f"{label_col}"])  

    return df, vocab

def create_vocab(df:pd.Series)->dict:
    from collections import Counter
    vocab = Counter()
    
    #Creatte a frequency vocab
    for text in df:
        vocab.update(text)
    
    # Add special tokens
    vocab['<PAD>'] = 0  # Padding token
    vocab['<UNK>'] = 1  # Unknown token

    return vocab


def encode_dataset(df:pd.DataFrame, encode_token_type:str,model_type:str,vocab:dict=None):
    if  model_type == "lstm":
        return encode_tokens(encode_token_type,df =df["text"],vocab=vocab),encode_labels(df["label"])

    pass
def encode_tokens(encode_token_type,vocab,df:pd.Series, max_len:int=None)->torch.Tensor:
    if encode_token_type == "one-hot":
        vocab_size = len(vocab)
        index_seq = df.apply(lambda x: [vocab.get(word, vocab['<UNK>']) for word in x])
        index_seq = [torch.tensor(seq, dtype=torch.long) for seq in index_seq]

        # Padding
        from torch.nn.utils.rnn import pad_sequence
        padded = pad_sequence(index_seq, batch_first=True, padding_value=vocab['<PAD>'])
        # One-hot encode
        tensor = F.one_hot(padded, num_classes=vocab_size)
    elif encode_token_type == "word2vec":
        pass
    elif encode_token_type == "glove":
        pass
    elif encode_token_type == "index":
        from torch.nn.utils.rnn import pad_sequence
        tensor = df.apply(lambda x: [vocab.get(word, vocab['<UNK>']) for word in x])
        tensor = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in tensor], batch_first=True, padding_value=vocab['<PAD>'])
    else:
        raise ValueError(f"Unknown encoding type: {encode_token_type}, check the config file.")
    
    return tensor
def encode_labels(df:pd.Series)-> torch.Tensor:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    tensor  = torch.tensor(le.fit_transform(df)) #dtype=torch.long?
    return tensor



def clean_doc(df:pd.Series):
    remove_https = re.compile(r"http[s]?\:\/\/.[a-zA-Z0-9\.\/\_?=%&#\-\+!]+")
    remove_punct = re.compile(r'[^\w\s]')

    df = df.apply(lambda x: remove_https.sub(' ', x))
    df = df.apply(lambda x: remove_punct.sub(' ', x))
    df = df.apply(lambda x: x.split())
    
    return df
def stop_words_removal(df:pd.Series,vocab:dict)->list[pd.Series, dict]:

    nltk.data.path.append(get_data_path())
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))
    print("Removing stopwords...")  # ← debug line
    #remove stopwords from vocab
    vocab = Counter({word: idx for word, idx in vocab.items() if word not in stop_words})
    return df.apply(lambda x: [word for word in x if word not in stop_words]), vocab

def rare_words_removal(df:pd.Series, vocab:dict, min_freq:int=2)->list[pd.Series, dict]:
    # remove words that appear less than min_freq times in the vocab
    counter = vocab
    print("Removing rare words...")  # ← debug line
    vocab = Counter({word: idx for word, idx in vocab.items() if counter[word] >= min_freq or word in  ['<PAD>', '<UNK>']})
    df = df.apply(lambda x: [word for word in x if word in vocab])
    msg = f"removed {len(counter) - len(vocab)} rare words from the vocabulary, thats total {counter.total() - vocab.total()} words removed from the cropus."
   

    #not_in_vocab = set(counter.keys()) - set(vocab.keys())
    #print(not_in_vocab, vocab["<PAD>"], vocab["<UNK>"] )

    print(msg)
    return df, vocab


if __name__ == "__main__":
    mr_all = pd.read_csv(join(get_data_path(), 'mr.csv'))
    mr_all,vocab = clean_data(mr_all,remove_stop_words=True, remove_rare_words=0)
    print(type(mr_all), mr_all.shape, "\n")
    text_tensor, label_tensor = encode_dataset(df=mr_all, encode_token_type="one-hot", model_type="lstm",vocab=vocab)
    print("Text tensor shape:", text_tensor.size())
    print("Label tensor shape:", label_tensor.size())
    print(text_tensor[5:10], "\n", label_tensor[:5])

    #print(mr_all)

    # ng20 = pd.read_csv(join(get_data_path(), '20ng.csv'))
    # ng20 = clean_data(ng20)
    # ng = ng20.copy()
    # ng["label"] = encode_labels(ng20["label"])
    # print(ng20["label"].head(5),"\n",ng["label"].head(5),"\n")