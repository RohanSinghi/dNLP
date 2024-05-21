```python
# src/utils/utils.py
import yaml
import re

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config, config_path):
    with open(config_path, 'w') as f:
        yaml.dump(config, f)


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


def normalize_text(text):
    text = remove_special_characters(text)
    text = convert_to_lowercase(text)
    text = remove_extra_whitespace(text)
    return text
```