import torch
import torch.nn as nn
from textgnn.models.base_text_classifier import BaseTextClassifier
from textgnn.config_class import ModelConfig, DatasetConfig


def create_fasttext_model(
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    dataset = None
):
    """
    Create FastText model from configuration.

    Args:
        model_config: Pydantic ModelConfig model
        dataset_config: Pydantic DatasetConfig model
        dataset: Optional dataset instance (not used for FastText)

    Returns:
        FastTextClassifier instance
    """
    model_specific_params = model_config.model_specific_params

    embedding_dim = model_specific_params.get("embedding_dim", 100)
    output_dim = model_specific_params.get("output_size", 2)
    min_n = model_specific_params.get("min_n", 3)
    max_n = model_specific_params.get("max_n", 6)
    vocab_size = model_specific_params.get("ngram_vocab_size", 100000)

    return FastTextClassifier(
        embedding_dim=embedding_dim,
        num_classes=output_dim,
        min_n=min_n,
        max_n=max_n,
        vocab_size=vocab_size
    )


class FastTextClassifier(BaseTextClassifier):
    """
    FastText model for text classification using character n-grams.

    This is the true FastText implementation that uses subword information
    via character n-grams. Unlike word-based models, it can handle
    out-of-vocabulary words by composing them from character n-grams.

    Architecture:
        1. Character n-gram embeddings (min_n to max_n characters)
        2. Average pooling across all n-grams in document
        3. Linear classification layer

    Args:
        embedding_dim: Dimension of n-gram embeddings
        num_classes: Number of output classes
        min_n: Minimum n-gram length (default 3)
        max_n: Maximum n-gram length (default 6)
        vocab_size: Maximum n-gram vocabulary size
    """
    def __init__(
        self,
        embedding_dim: int = 100,
        num_classes: int = 2,
        min_n: int = 3,
        max_n: int = 6,
        vocab_size: int = 100000
    ):
        # Note: BaseTextClassifier expects vocab_size, embedding_dim, output_size
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            output_size=num_classes
        )

        self.min_n = min_n
        self.max_n = max_n
        self.num_classes = num_classes

        # Character n-gram embeddings (override base embedding)
        self.ngram_embeddings = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )
        # Remove base embedding to avoid confusion
        self.embedding = None

        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # N-gram to index mapping (built dynamically during training)
        self.ngram2idx = {}
        self.idx_counter = 1  # Start from 1 (0 reserved for padding)

        # Required: pointer to training function
        from textgnn.models.fastText.train import train as train_fasttext
        self.train_func = train_fasttext

    def get_ngrams(self, word: str) -> list[str]:
        """
        Generate character n-grams for a word.

        Args:
            word: Input word

        Returns:
            List of n-gram strings
        """
        # Add word boundaries
        word = f"<{word}>"
        ngrams = []

        # Generate n-grams from min_n to max_n length
        for n in range(self.min_n, min(self.max_n + 1, len(word) + 1)):
            for i in range(len(word) - n + 1):
                ngrams.append(word[i:i+n])

        return ngrams

    def ngram_to_indices(self, ngrams: list[str]) -> list[int]:
        """
        Convert n-gram strings to indices, building vocabulary on the fly.

        Args:
            ngrams: List of n-gram strings

        Returns:
            List of integer indices
        """
        indices = []
        for ngram in ngrams:
            if ngram not in self.ngram2idx:
                # Add new n-gram to vocabulary if space available
                if self.idx_counter < self.ngram_embeddings.num_embeddings:
                    self.ngram2idx[ngram] = self.idx_counter
                    self.idx_counter += 1
                else:
                    # If vocab is full, use padding index (0)
                    indices.append(0)
                    continue
            indices.append(self.ngram2idx[ngram])
        return indices

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode a single text document into embedding vector.

        Args:
            text: Space-separated tokenized text string

        Returns:
            [embedding_dim] tensor
        """
        # Split into words (already tokenized)
        words = text.lower().split()

        # Get n-grams for all words
        all_ngrams = []
        for word in words:
            ngrams = self.get_ngrams(word)
            indices = self.ngram_to_indices(ngrams)
            all_ngrams.extend(indices)

        if not all_ngrams:
            # Return zero vector if no n-grams
            return torch.zeros(
                self.embedding_dim,
                device=next(self.parameters()).device
            )

        # Convert to tensor and get embeddings
        indices_tensor = torch.tensor(
            all_ngrams,
            device=next(self.parameters()).device
        )
        embeddings = self.ngram_embeddings(indices_tensor)

        # Average all n-gram embeddings
        text_embedding = embeddings.mean(dim=0)
        return text_embedding

    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Forward pass for a batch of texts.

        Args:
            texts: List of text strings (batch)

        Returns:
            [batch_size, num_classes] logits
        """
        embeddings = []
        for text in texts:
            embedding = self.encode_text(text)
            embeddings.append(embedding)

        # Stack embeddings into batch
        batch_embeddings = torch.stack(embeddings)

        # Classification
        logits = self.classifier(batch_embeddings)
        return logits
