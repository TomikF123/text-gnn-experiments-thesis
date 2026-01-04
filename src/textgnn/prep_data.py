import pandas as pd
from os.path import join, exists
import re
import nltk
from .utils import get_data_path
import torch
from collections import Counter
from .logger import setup_logger

logger = setup_logger(__name__)


def clean_data(
    df: pd.DataFrame,
    label_col: str = "label",
    text_col: str = "text",
    remove_stop_words: bool = True,
    remove_rare_words: int = True,
    vocab_size: int = None,
) -> tuple[pd.DataFrame, dict]:
    """
    DEPRECATED: This function processes all data together and causes data leakage.
    Use clean_text_pipeline(), build_vocabulary(), and apply_vocabulary() instead.

    This function is kept for backward compatibility but should not be used.
    """
    import warnings
    warnings.warn(
        "clean_data() is deprecated due to data leakage. "
        "Use clean_text_pipeline(), build_vocabulary(), and apply_vocabulary() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # Keep existing implementation for backward compatibility
    df.dropna(subset=[f"{text_col}"], inplace=True)
    df.dropna(subset=[f"{label_col}"], inplace=True)
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
    df.dropna(subset=[f"{text_col}"], inplace=True)
    logger.info(f"Number of NAs in {text_col}: {df[f'{text_col}'].isna().sum()}")
    df = df[df[text_col].apply(lambda x: len(x) > 0)].copy()
    logger.info(
        f"Number of rows with empty text: {df[f'{text_col}'].apply(lambda x: len(x) == 0).sum()}"
    )
    return df, vocab


def build_word_freq(df: pd.Series, vocab_size: int = None) -> Counter:
    vocab_counter = Counter()
    
    for text in df:
        if text is None:
            print(f"WARNING: Found None in text_series")
            continue
        if not isinstance(text, (list, str)):
            print(f"WARNING: Found non-string/list type: {type(text)}, value: {text}")
            continue
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


def clean_doc(df: pd.Series) -> pd.Series:
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


def clean_text_pipeline(
    text_series: pd.Series,
    text_col: str = "text"
) -> pd.Series:
    """
    Clean and tokenize text without building vocabulary.

    Args:
        text_series: Series of raw text strings
        text_col: Column name (for logging)

    Returns:
        Series of tokenized text (list[str])
    """
    # Fill NAs with empty string to preserve indices
    text_series = text_series.fillna("")

    # Convert to string and lowercase
    text_series = text_series.astype(str).str.lower()

    # Apply cleaning (reuse existing clean_doc logic)
    text_series = clean_doc(text_series)

    # Convert empty strings to empty lists (from .split() on "")
    # This ensures all documents are lists, even if empty
    text_series = text_series.apply(lambda x: x if isinstance(x, list) else [])

    logger.info(f"Cleaned {len(text_series)} documents")

    return text_series


def build_vocabulary(
    text_series: pd.Series,
    remove_stop_words: bool = True,
    remove_rare_words: int | None = None,
    vocab_size: int | None = None
) -> dict[str, int]:
    """
    Build vocabulary from cleaned, tokenized text.

    Args:
        text_series: Series of tokenized documents (list[str])
        remove_stop_words: Whether to exclude stopwords from vocab
        remove_rare_words: Minimum word frequency (None to keep all)
        vocab_size: Maximum vocabulary size (None for unlimited)

    Returns:
        word-to-index mapping with <PAD> and <UNK> tokens
    """
    # Build word frequency counter
    word_freq = build_word_freq(text_series, vocab_size)

    # Remove stopwords from vocabulary
    if remove_stop_words:
        from nltk.corpus import stopwords
        import nltk
        nltk.data.path.append(get_data_path())
        stop_words = set(stopwords.words("english"))

        logger.info("Removing stopwords from vocabulary...")
        word_freq = Counter({
            word: count for word, count in word_freq.items()
            if word not in stop_words
        })

    # Remove rare words from vocabulary
    if remove_rare_words:
        logger.info(f"Removing words with frequency < {remove_rare_words}...")
        original_count = len(word_freq)
        word_freq = Counter({
            word: count for word, count in word_freq.items()
            if count >= remove_rare_words
        })
        logger.info(f"Removed {original_count - len(word_freq)} rare words from vocabulary")

    # Build word-to-index mapping
    vocab = build_word_to_index(word_freq)
    logger.info(f"Final vocabulary size: {len(vocab)}")

    return vocab


def apply_vocabulary(
    text_series: pd.Series,
    vocab: dict[str, int],
    remove_stop_words: bool = False
) -> pd.Series:
    """
    Filter tokenized text to only include words in vocabulary.

    Args:
        text_series: Series of tokenized documents (list[str])
        vocab: word-to-index mapping
        remove_stop_words: Whether to also remove stopwords from documents

    Returns:
        Series of filtered tokenized documents
    """
    def filter_tokens(tokens):
        """Helper to safely filter tokens, handling NaN/invalid values."""
        # Handle NaN or non-list values
        if not isinstance(tokens, list):
            return []

        if remove_stop_words:
            from nltk.corpus import stopwords
            import nltk
            nltk.data.path.append(get_data_path())
            stop_words = set(stopwords.words("english"))
            return [word for word in tokens if word not in stop_words and word in vocab]
        else:
            return [word for word in tokens if word in vocab]

    if remove_stop_words:
        from nltk.corpus import stopwords
        import nltk
        nltk.data.path.append(get_data_path())
        stop_words = set(stopwords.words("english"))

        # Remove stopwords AND filter to vocab
        text_series = text_series.apply(
            lambda tokens: [] if not isinstance(tokens, list) else [
                word for word in tokens
                if word not in stop_words and word in vocab
            ]
        )
    else:
        # Only filter to vocab
        text_series = text_series.apply(
            lambda tokens: [] if not isinstance(tokens, list) else [word for word in tokens if word in vocab]
        )

    logger.info(f"Applied vocabulary to {len(text_series)} documents")

    return text_series


if __name__ == "__main__":
    pass
