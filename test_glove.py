import pickle
from textgnn.utils import load_glove_embeddings

vocab = pickle.load(open("saved/mr-train-80-val-10-test-10-stop-words-remove-false-rare-words-remove-0-vocab-size-none-v2/vocab.pkl", "rb"))
print(f"Vocab size: {len(vocab)}")
emb = load_glove_embeddings(vocab, 300, tokens_trained_on=6)
print(f"Shape: {emb.shape}")
