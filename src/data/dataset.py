'''Module for dataset related classes and methods.'''

class Dataset:
    """A class representing a dataset.

    Attributes:
        name (str): The name of the dataset.
        data (list): The data associated with the dataset.
    """
    def __init__(self, name, data):
        """Initializes a new Dataset instance.

        :param name: The name of the dataset.
        :type name: str
        :param data: The data for the dataset.
        :type data: list
        """
        self.name = name
        self.data = data

    def __len__(self):
        """Returns the length of the dataset.

        :return: The length of the dataset.
        :rtype: int
        """
        return len(self.data)