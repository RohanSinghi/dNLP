'''Module for defining data fields.'''

class Field:
    """A class representing a data field.

    Attributes:
        name (str): The name of the field.
        dtype (str): The data type of the field.
    """
    def __init__(self, name, dtype):
        """Initializes a new Field instance.

        :param name: The name of the field.
        :type name: str
        :param dtype: The data type of the field.
        :type dtype: str
        """
        self.name = name
        self.dtype = dtype

    def __repr__(self):
        """Returns a string representation of the Field.

        :return: A string representation of the Field.
        :rtype: str
        """
        return f"Field(name='{self.name}', dtype='{self.dtype}')"