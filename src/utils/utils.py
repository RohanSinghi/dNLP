'''Module containing utility functions.'''

import json

def load_json(file_path):
    """Loads a JSON file.

    :param file_path: The path to the JSON file.
    :type file_path: str
    :return: The loaded JSON data.
    :rtype: dict
    """
    with open(file_path, 'r') as f:
        return json.load(f)