import unittest
from src.text.text_preprocessing import tokenize, lowercase, clean_text

class TestTextPreprocessing(unittest.TestCase):

    def test_tokenize(self):
        text = "This is a test."
        expected_tokens = ['This', 'is', 'a', 'test', '.']
        self.assertEqual(tokenize(text), expected_tokens)

    def test_lowercase(self):
        text = "This IS a Test."
        expected_text = "this is a test."
        self.assertEqual(lowercase(text), expected_text)

    def test_clean_text(self):
        text = "This is a test string with some punctuation!  And extra spaces."
        expected_text = "This is a test string with some punctuation  And extra spaces"
        self.assertEqual(clean_text(text), "This is a test string with some punctuation And extra spaces")

    def test_clean_text_empty(self):
        text = ""
        expected_text = ""
        self.assertEqual(clean_text(text), expected_text)

if __name__ == '__main__':
    unittest.main()