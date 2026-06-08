import unittest
from src.data_loader import load_data

class TestDataLoader(unittest.TestCase):

    def test_load_data(self):
        # Test loading the dataset
        data = load_data('data/raw/breast_cancer_data.csv')
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)

    def test_load_data_invalid_path(self):
        # Test loading data from an invalid path
        with self.assertRaises(FileNotFoundError):
            load_data('data/raw/invalid_path.csv')

if __name__ == '__main__':
    unittest.main()