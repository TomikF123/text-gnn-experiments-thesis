## pd.df in,
import pandas as pd
from os.path import join, exists
import re
import nltk
from utils import get_data_path
import torch
from collections import Counter
import torch.nn.functional as F


def clean_data(
    df: pd.DataFrame,
    label_col: str = "label",
    text_col: str = "text",
    remove_stop_words: bool = True,
    remove_rare_words: int = True,
    vocab_size: int = None,
) -> list[pd.DataFrame, dict]:
    # orig_len = len(df) #debug
    df.dropna(subset=[f"{text_col}"], inplace=True)
    df.dropna(subset=[f"{label_col}"], inplace=True)
    # print(f"Cleaned data from {orig_len} to {len(df)} samples.") #debug
    df[f"{text_col}"] = df[f"{text_col}"].astype(str)
    df[f"{text_col}"] = df[f"{text_col}"].str.lower()
    df[f"{text_col}"] = clean_doc(df[f"{text_col}"])
    word_freq = build_word_freq(df[f"{text_col}"], vocab_size)
    if remove_stop_words:
        df[f"{text_col}"], word_freq = stop_words_removal(df[f"{text_col}"], word_freq)
    if remove_rare_words:
        df[f"{text_col}"], word_freq = rare_words_removal(
            df[f"{text_col}"], word_freq, min_freq=remove_rare_words
        )
    vocab = build_word_to_index(word_freq)
    # df[f"{label_col}"] = encode_labels(df[f"{label_col}"])
    df.dropna(subset=[f"{text_col}"], inplace=True)
    print(f"number of NAs in {text_col}: {df[f'{text_col}'].isna().sum()}")
    df = df[df[text_col].apply(lambda x: len(x) > 0)].copy()
    print(
        "Number of rows that have empty text:",
        df[f"{text_col}"].apply(lambda x: len(x) == 0).sum(),
    )
    # final_len = len(df)#debug
    # print(f"Cleaned data from {orig_len} to {final_len} samples.") #debug
    return df, vocab


def build_word_freq(df: pd.Series, vocab_size: int = None) -> Counter:
    vocab_counter = Counter()
    for text in df:
        vocab_counter.update(text)

    if vocab_size:
        vocab_counter = Counter(
            dict(vocab_counter.most_common(vocab_size - 2))
        )  # Reserve 2 slots for PAD and UNK tokens

    return vocab_counter


def build_word_to_index(vocab_counter: Counter) -> dict:
    word_to_index = {"<PAD>": 0, "<UNK>": 1}
    for idx, (word, _) in enumerate(vocab_counter.items(), start=2):
        word_to_index[word] = idx
    return word_to_index


def encode_labels(df: pd.Series) -> torch.Tensor:
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    tensor = torch.tensor(le.fit_transform(df))  # dtype=torch.long?
    return tensor


def clean_doc(df: pd.Series):
    remove_https = re.compile(r"http[s]?\:\/\/.[a-zA-Z0-9\.\/\_?=%&#\-\+!]+")
    remove_punct = re.compile(r"[^\w\s]")
    remove_encoded_block = re.compile(  # Remove long UUencode/Base64-like lines
        r"(?:[-=]{5,} Part \d+ of \d+ [-=]{5,}\s+" r"(?:[^\n]{15,}\s*){3,})",
        flags=re.IGNORECASE,
    )
    remove_uuencode_block = re.compile(  # Encoded files in 20ng
        r"begin \d{3} .+?\n(?:.*\n)*?end", flags=re.IGNORECASE
    )

    # These should follow some logical order
    df = df.apply(lambda x: remove_https.sub(" ", x))
    df = df.apply(lambda x: remove_encoded_block.sub(" ", x))
    df = df.apply(lambda x: remove_uuencode_block.sub(" ", x))
    df = df.apply(lambda x: remove_punct.sub(" ", x))
    df = df.apply(lambda x: x.split())

    return df


def stop_words_removal(df: pd.Series, vocab: dict) -> list[pd.Series, dict]:

    nltk.data.path.append(get_data_path())
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words("english"))
    print("Removing stopwords...")  #  debug
    # remove stopwords from vocab
    vocab = Counter(
        {word: idx for word, idx in vocab.items() if word not in stop_words}
    )
    return df.apply(lambda x: [word for word in x if word not in stop_words]), vocab


def rare_words_removal(
    df: pd.Series, vocab: dict, min_freq: int = 2
) -> list[pd.Series, dict]:
    # remove words that appear less than min_freq times in the vocab
    counter = vocab
    print("Removing rare words...")  # ← debug line
    vocab = Counter(
        {
            word: idx
            for word, idx in vocab.items()
            if counter[word] >= min_freq or word in ["<PAD>", "<UNK>"]
        }
    )
    df = df.apply(lambda x: [word for word in x if word in vocab])
    msg = f"removed {len(counter) - len(vocab)} rare words from the vocabulary, thats total {counter.total() - vocab.total()} words removed from the cropus."

    # not_in_vocab = set(counter.keys()) - set(vocab.keys())
    # print(not_in_vocab, vocab["<PAD>"], vocab["<UNK>"] )

    print(msg)
    return df, vocab


if __name__ == "__main__":
    # mr_all = pd.read_csv(join(get_data_path(), 'mr.csv'))
    # mr_all,vocab = clean_data(mr_all,remove_stop_words=False, remove_rare_words=0,vocab_size=None)
    # print(type(mr_all), mr_all.shape, "\n")
    # text_tensor, label_tensor = encode_dataset(df=mr_all, encode_token_type="ti-idf", model_type="lstm",vocab=vocab)
    # print("Text tensor shape:", text_tensor.size())
    # #print("Label tensor shape:", label_tensor.size())
    # print(text_tensor[0], "\n")
    # print(build_word_to_index(build_word_freq(mr_all["text"], vocab_size=1000)))
    pass
    # print(mr_all)

    # ng20 = pd.read_csv(join(get_data_path(), '20ng.csv'))
    # ng20 = clean_data(ng20)
    # ng = ng20.copy()
    # ng["label"] = encode_labels(ng20["label"])
    # print(ng20["label"].head(5),"\n",ng["label"].head(5),"\n")
