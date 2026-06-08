import unittest
from src.data_loader import load_data  # Adjust the import based on your actual function name

class TestDataLoader(unittest.TestCase):

    def test_load_data(self):
        # Test loading data from the raw directory
        data = load_data('data/raw')  # Adjust the path as necessary
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)  # Ensure data is loaded and not empty

    def test_data_format(self):
        # Test the format of the loaded data
        data = load_data('data/raw')  # Adjust the path as necessary
        self.assertIsInstance(data, list)  # Assuming the data is loaded as a list
        # Add more assertions based on the expected structure of the data

if __name__ == '__main__':
    unittest.main()