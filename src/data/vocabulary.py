import torchtext.vocab as vocab
from collections import Counter
from typing import List, Union

class Vocabulary:
    def __init__(self, tokens: Union[List[str], None] = None, min_freq: int = 1, special_tokens: Union[List[str], None] = None):
        """Builds a vocabulary from a list of tokens.

        Args:
            tokens: A list of tokens to build the vocabulary from.
                   If None, the vocabulary will be empty.
            min_freq: The minimum frequency a token must have to be included in the vocabulary.
            special_tokens: A list of special tokens to add to the vocabulary (e.g., <UNK>, <PAD>).
        """
        self.min_freq = min_freq
        self.special_tokens = special_tokens if special_tokens else []
        self.token_to_index = {}
        self.index_to_token = []
        self.build_vocabulary(tokens if tokens else [])
        

    def build_vocabulary(self, tokens: List[str]):
        """Builds the vocabulary from a list of tokens.

        Args:
            tokens: A list of tokens to build the vocabulary from.
        """
        counter = Counter(tokens)
        # Filter tokens by minimum frequency
        filtered_tokens = [token for token, count in counter.items() if count >= self.min_freq]

        # Add special tokens first
        for token in self.special_tokens:
            self.add_token(token)

        # Add filtered tokens to the vocabulary
        for token in filtered_tokens:
            self.add_token(token)

    def add_token(self, token: str):
        """Adds a token to the vocabulary.

        Args:
            token: The token to add.
        """
        if token not in self.token_to_index:
            self.token_to_index[token] = len(self.index_to_token)
            self.index_to_token.append(token)

    def __len__(self) -> int:
        """Returns the size of the vocabulary.
        """
        return len(self.index_to_token)

    def __getitem__(self, token: str) -> int:
        """Returns the index of a token.

        Args:
            token: The token to get the index of.

        Returns:
            The index of the token, or -1 if the token is not in the vocabulary.
        """
        return self.token_to_index.get(token, -1)

    def get_token(self, index: int) -> str:
        """Returns the token at a given index.

        Args:
            index: The index of the token to get.

        Returns:
            The token at the given index, or None if the index is out of range.
        """
        if 0 <= index < len(self.index_to_token):
            return self.index_to_token[index]
        return None

    def contains(self, token: str) -> bool:
        """Checks if the vocabulary contains a given token.

        Args:
            token: The token to check.

        Returns:
            True if the vocabulary contains the token, False otherwise.
        """
        return token in self.token_to_index

    def get_vocab(self) -> vocab.Vocab:
        """Returns the torchtext Vocab object.
        """
        # Currently always rebuilds, could be cached if needed
        counter = Counter(self.index_to_token)
        return vocab.Vocab(counter, min_freq=0, specials=self.special_tokens)

