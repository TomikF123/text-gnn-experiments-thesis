from dataset import TextDataset

class LSTMDataset(TextDataset):
    def __init__(self, data, labels):
        super().__init__(data, labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
