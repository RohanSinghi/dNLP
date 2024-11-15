import nltk
import re

# Download nltk resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def tokenize(text):
    '''Tokenizes the input text using nltk.word_tokenize.'''
    return nltk.word_tokenize(text)

def lowercase(text):
    '''Converts the input text to lowercase.'''
    return text.lower()

def clean_text(text):
    '''Performs basic text cleaning by removing special characters and extra spaces.'''
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = re.sub(r'\s+', ' ', text) # Remove extra spaces
    return text.strip()

if __name__ == '__main__':
    text = "This is a test string with some punctuation!  And extra spaces."
    print(f"Original text: {text}")
    tokenized_text = tokenize(text)
    print(f"Tokenized text: {tokenized_text}")
    lowercased_text = lowercase(text)
    print(f"Lowercased text: {lowercased_text}")
    cleaned_text = clean_text(text)
    print(f"Cleaned text: {cleaned_text}")