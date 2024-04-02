'''Module for text preprocessing functions.'''

import re

def clean_text(text):
    """Cleans the input text by removing special characters and converting to lowercase.

    :param text: The input text.
    :type text: str
    :return: The cleaned text.
    :rtype: str
    """
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()