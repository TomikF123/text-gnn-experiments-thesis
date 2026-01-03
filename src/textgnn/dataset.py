from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, df, vocab: dict = None,encode_token_type: str = "index"):
        self.df = df
        self.vocab = vocab
        self.encode_token_type = encode_token_type
        self.collate_fn = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        raise NotImplementedError("This method should be overridden by subclasses")

    def __repr__(self):
        base = super().__repr__()
        return f"{base[:-1]} \n vocab len={len(self.vocab) if self.vocab else 0}\n encode_token_type={self.encode_token_type} \n num classes={len(set(self.labels))}\n num samples={len(self.df)}>"
