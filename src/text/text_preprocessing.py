```python
# src/text/text_preprocessing.py
# This module is now deprecated, and its functionality has been moved to src/utils/utils.py
# Consider removing this file in a future version.

import re


def remove_special_characters(text):
    # Remove special characters, punctuation, etc.
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


def convert_to_lowercase(text):
    return text.lower()


def remove_extra_whitespace(text):
  # Remove multiple spaces to single space
  text = re.sub(' +', ' ', text)
  return text.strip()



def preprocess_text(text):
  text = remove_special_characters(text)
  text = convert_to_lowercase(text)
  text = remove_extra_whitespace(text)
  return text
```