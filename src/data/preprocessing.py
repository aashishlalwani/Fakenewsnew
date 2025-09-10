"""
Data preprocessing utilities for fake news detection.

This module contains functions for cleaning, preprocessing, and preparing
text data for machine learning models.
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.model_selection import train_test_split


class TextPreprocessor:
    """
    A comprehensive text preprocessing class for fake news detection.
    """
    
    def __init__(self, 
                 remove_stopwords: bool = True,
                 use_lemmatization: bool = True,
                 use_stemming: bool = False):
        """
        Initialize the text preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            use_lemmatization: Whether to use lemmatization
            use_stemming: Whether to use stemming
        """
        self.remove_stopwords = remove_stopwords
        self.use_lemmatization = use_lemmatization
        self.use_stemming = use_stemming
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize NLTK tools
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        
        if self.use_lemmatization:
            self.lemmatizer = WordNetLemmatizer()
            
        if self.use_stemming:
            self.stemmer = PorterStemmer()
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            
        try:
            nltk.data.find('corpora/omw-1.4')
        except LookupError:
            nltk.download('omw-1.4')
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning operations.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove extra whitespaces and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers (optional - you might want to keep them)
        text = re.sub(r'\d+', '', text)
        
        return text.strip()
    
    def tokenize_and_process(self, text: str) -> List[str]:
        """
        Tokenize text and apply preprocessing steps.
        
        Args:
            text: Input text string
            
        Returns:
            List of processed tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Remove single characters and empty strings
        tokens = [token for token in tokens if len(token) > 1]
        
        # Apply lemmatization
        if self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Apply stemming
        if self.use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        return tokens
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete text preprocessing pipeline.
        
        Args:
            text: Input text string
            
        Returns:
            Processed text string
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize and process
        tokens = self.tokenize_and_process(cleaned_text)
        
        # Join tokens back to string
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df: pd.DataFrame, 
                           text_column: str = 'text',
                           target_column: str = 'label') -> pd.DataFrame:
        """
        Preprocess an entire dataframe.
        
        Args:
            df: Input dataframe
            text_column: Name of the text column
            target_column: Name of the target column
            
        Returns:
            Processed dataframe
        """
        df_processed = df.copy()
        
        # Apply preprocessing to text column
        print("Preprocessing text data...")
        df_processed[f'{text_column}_processed'] = df_processed[text_column].apply(
            self.preprocess_text
        )
        
        # Remove rows with empty processed text
        df_processed = df_processed[df_processed[f'{text_column}_processed'] != '']
        
        print(f"Original dataset size: {len(df)}")
        print(f"Processed dataset size: {len(df_processed)}")
        
        return df_processed


def split_data(df: pd.DataFrame, 
               text_column: str = 'text_processed',
               target_column: str = 'label',
               test_size: float = 0.2,
               val_size: float = 0.1,
               random_state: int = 42) -> Tuple[pd.DataFrame, ...]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: Input dataframe
        text_column: Name of the text column
        target_column: Name of the target column
        test_size: Proportion of test data
        val_size: Proportion of validation data
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X = df[text_column]
    y = df[target_column]
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate train and validation
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_text_statistics(df: pd.DataFrame, text_column: str = 'text') -> dict:
    """
    Get basic statistics about text data.
    
    Args:
        df: Input dataframe
        text_column: Name of the text column
        
    Returns:
        Dictionary with text statistics
    """
    stats = {}
    
    # Character count statistics
    char_counts = df[text_column].str.len()
    stats['avg_char_count'] = char_counts.mean()
    stats['median_char_count'] = char_counts.median()
    stats['max_char_count'] = char_counts.max()
    stats['min_char_count'] = char_counts.min()
    
    # Word count statistics
    word_counts = df[text_column].str.split().str.len()
    stats['avg_word_count'] = word_counts.mean()
    stats['median_word_count'] = word_counts.median()
    stats['max_word_count'] = word_counts.max()
    stats['min_word_count'] = word_counts.min()
    
    return stats


if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor()
    
    # Example text
    sample_text = "This is a SAMPLE text with URLs http://example.com and some punctuation!!!"
    processed = preprocessor.preprocess_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Processed: {processed}")