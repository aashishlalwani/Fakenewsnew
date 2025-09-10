"""
Unit tests for the preprocessing module.
"""

import unittest
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.preprocessing import TextPreprocessor, split_data, get_text_statistics


class TestTextPreprocessor(unittest.TestCase):
    """Test cases for TextPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.preprocessor = TextPreprocessor()
        self.sample_text = "This is a SAMPLE text with URLs http://example.com and some punctuation!!!"
        
    def test_clean_text(self):
        """Test text cleaning functionality."""
        cleaned = self.preprocessor.clean_text(self.sample_text)
        
        # Should remove URLs
        self.assertNotIn("http://example.com", cleaned)
        
        # Should remove punctuation
        self.assertNotIn("!!!", cleaned)
        
        # Should convert to lowercase
        self.assertNotIn("SAMPLE", cleaned)
        self.assertIn("sample", cleaned)
    
    def test_clean_text_with_none(self):
        """Test cleaning with None input."""
        result = self.preprocessor.clean_text(None)
        self.assertEqual(result, "")
    
    def test_tokenize_and_process(self):
        """Test tokenization and processing."""
        tokens = self.preprocessor.tokenize_and_process("This is a test sentence")
        
        # Should return list of strings
        self.assertIsInstance(tokens, list)
        self.assertTrue(all(isinstance(token, str) for token in tokens))
        
        # Should remove single characters if configured
        self.assertNotIn("a", tokens)
    
    def test_preprocess_text(self):
        """Test complete preprocessing pipeline."""
        processed = self.preprocessor.preprocess_text(self.sample_text)
        
        # Should return a string
        self.assertIsInstance(processed, str)
        
        # Should be different from original
        self.assertNotEqual(processed, self.sample_text)
        
        # Should not be empty
        self.assertTrue(len(processed) > 0)


class TestDataUtils(unittest.TestCase):
    """Test cases for data utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_df = pd.DataFrame({
            'text': ['This is real news', 'This is fake news', 'Another real article'],
            'label': ['REAL', 'FAKE', 'REAL']
        })
    
    def test_split_data(self):
        """Test data splitting functionality."""
        # Add processed text column
        self.sample_df['text_processed'] = self.sample_df['text']
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            self.sample_df, test_size=0.33, val_size=0.33
        )
        
        # Check that all splits have data
        self.assertTrue(len(X_train) > 0)
        self.assertTrue(len(X_val) > 0)
        self.assertTrue(len(X_test) > 0)
        
        # Check that labels match features
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_val), len(y_val))
        self.assertEqual(len(X_test), len(y_test))
    
    def test_get_text_statistics(self):
        """Test text statistics calculation."""
        stats = get_text_statistics(self.sample_df)
        
        # Check that all expected keys are present
        expected_keys = [
            'avg_char_count', 'median_char_count', 'max_char_count', 'min_char_count',
            'avg_word_count', 'median_word_count', 'max_word_count', 'min_word_count'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
            self.assertIsInstance(stats[key], (int, float))


if __name__ == '__main__':
    unittest.main()