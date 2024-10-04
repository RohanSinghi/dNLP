import unittest
import os
from src.data.dataset import TextDataset

class TestTextDataset(unittest.TestCase):

    def setUp(self):
        # Create a dummy dataset file for testing.
        self.dummy_data = ["This is a positive example.\t1", "This is a negative example.\t0", "Another positive example.\t1"]
        self.dummy_file_path = 'test_dataset.txt'
        with open(self.dummy_file_path, 'w') as f:
            f.write('\n'.join(self.dummy_data))

    def tearDown(self):
        # Remove the dummy dataset file.
        os.remove(self.dummy_file_path)

    def test_dataset_loading(self):
        dataset = TextDataset(self.dummy_file_path)
        self.assertEqual(len(dataset), 3)
        text, label = dataset[0]
        self.assertEqual(label, 1)
        self.assertEqual(text, "This is a positive example.")


if __name__ == '__main__':
    unittest.main()