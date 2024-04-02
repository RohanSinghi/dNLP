'''Module for managing vocabulary.'''

class Vocabulary:
    """A class representing a vocabulary.

    Attributes:
        word_to_index (dict): A dictionary mapping words to indices.
        index_to_word (list): A list mapping indices to words.
    """
    def __init__(self):
        """Initializes a new Vocabulary instance.
        """
        self.word_to_index = {}
        self.index_to_word = []

    def add_word(self, word):
        """Adds a word to the vocabulary.

        :param word: The word to add.
        :type word: str
        :return: The index of the word in the vocabulary.
        :rtype: int
        """
        if word not in self.word_to_index:
            self.word_to_index[word] = len(self.index_to_word)
            self.index_to_word.append(word)
        return self.word_to_index[word]

    def __len__(self):
        """Returns the size of the vocabulary.

        :return: The size of the vocabulary.
        :rtype: int
        """
        return len(self.index_to_word)