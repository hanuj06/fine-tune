"""
Data Preprocessing Module
==========================
This module handles tokenization, cleaning, and train-validation split
for the collected text data.
"""

import os
import json
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    A class to preprocess text data for language model fine-tuning.
    Handles cleaning, tokenization preparation, and train-val split.
    """
    
    def __init__(self, output_dir: str = "data/processed"):
        """
        Initialize the DataPreprocessor.
        
        Args:
            output_dir: Directory to save processed data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_texts: List[str] = []
        
    def clean_text(self, text: str, 
                   remove_urls: bool = True,
                   remove_emails: bool = True,
                   remove_extra_spaces: bool = True) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
            remove_urls: Remove URLs from text
            remove_emails: Remove email addresses
            remove_extra_spaces: Remove extra whitespace
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove URLs
        if remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        if remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        if remove_extra_spaces:
            text = ' '.join(text.split())
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def remove_noise(self, texts: List[str],
                    min_length: int = 5,
                    max_length: Optional[int] = None,
                    remove_duplicates: bool = True) -> List[str]:
        """
        Remove noisy or invalid text entries.
        
        Args:
            texts: List of text entries
            min_length: Minimum text length (in characters)
            max_length: Maximum text length (None for no limit)
            remove_duplicates: Remove duplicate entries
            
        Returns:
            Filtered list of texts
        """
        cleaned_texts = []
        seen = set()
        
        for text in texts:
            # Skip empty or very short texts
            if not text or len(text) < min_length:
                continue
            
            # Skip texts that are too long
            if max_length and len(text) > max_length:
                continue
            
            # Remove duplicates
            if remove_duplicates:
                if text in seen:
                    continue
                seen.add(text)
            
            cleaned_texts.append(text)
        
        print(f"Filtered {len(texts)} -> {len(cleaned_texts)} texts")
        return cleaned_texts
    
    def preprocess_batch(self, texts: List[str],
                        clean: bool = True,
                        remove_noise_flag: bool = True,
                        min_length: int = 5,
                        max_length: Optional[int] = None) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of text entries
            clean: Apply text cleaning
            remove_noise_flag: Remove noisy entries
            min_length: Minimum text length
            max_length: Maximum text length
            
        Returns:
            Preprocessed texts
        """
        processed = texts.copy()
        
        # Clean texts
        if clean:
            processed = [self.clean_text(text) for text in processed]
        
        # Remove noise
        if remove_noise_flag:
            processed = self.remove_noise(processed, min_length, max_length)
        
        self.processed_texts = processed
        print(f"Preprocessed {len(processed)} texts")
        return processed
    
    def create_train_val_split(self, texts: List[str],
                               val_size: float = 0.1,
                               test_size: float = 0.1,
                               random_state: int = 42) -> Dict[str, List[str]]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            texts: List of text entries
            val_size: Validation set proportion
            test_size: Test set proportion
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with 'train', 'val', and 'test' splits
        """
        # First split: separate test set
        train_val, test = train_test_split(
            texts, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Second split: separate validation from training
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            random_state=random_state
        )
        
        splits = {
            'train': train,
            'val': val,
            'test': test
        }
        
        print(f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        return splits
    
    def save_splits(self, splits: Dict[str, List[str]], 
                   prefix: str = "data") -> Dict[str, str]:
        """
        Save train/val/test splits to files.
        
        Args:
            splits: Dictionary with train/val/test splits
            prefix: Prefix for output filenames
            
        Returns:
            Dictionary with paths to saved files
        """
        saved_paths = {}
        
        for split_name, texts in splits.items():
            # Save as JSON
            json_path = self.output_dir / f"{prefix}_{split_name}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({"texts": texts}, f, ensure_ascii=False, indent=2)
            
            # Also save as plain text (one entry per line)
            txt_path = self.output_dir / f"{prefix}_{split_name}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                for text in texts:
                    f.write(text + '\n')
            
            saved_paths[split_name] = str(json_path)
            print(f"Saved {split_name} split to {json_path}")
        
        return saved_paths
    
    def prepare_for_training(self, texts: List[str],
                            max_length: int = 512) -> List[str]:
        """
        Prepare texts for model training by ensuring appropriate length.
        
        Args:
            texts: List of text entries
            max_length: Maximum sequence length (in characters, approximate)
            
        Returns:
            Prepared texts
        """
        prepared = []
        
        for text in texts:
            # If text is too long, we could split it or truncate
            # For simplicity, we'll truncate here
            if len(text) > max_length:
                text = text[:max_length]
            
            prepared.append(text)
        
        return prepared
    
    def get_statistics(self, texts: List[str]) -> Dict:
        """
        Get statistics about the text data.
        
        Args:
            texts: List of text entries
            
        Returns:
            Dictionary with statistics
        """
        if not texts:
            return {"total_entries": 0}
        
        lengths = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]
        
        # Count language distribution (simple heuristic)
        hindi_count = sum(1 for text in texts if any('\u0900' <= c <= '\u097F' for c in text))
        english_count = sum(1 for text in texts if any('a' <= c.lower() <= 'z' for c in text))
        
        return {
            "total_entries": len(texts),
            "total_characters": sum(lengths),
            "avg_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "total_words": sum(word_counts),
            "avg_words": np.mean(word_counts),
            "std_words": np.std(word_counts),
            "texts_with_hindi": hindi_count,
            "texts_with_english": english_count
        }
    
    def create_tokenizer_training_data(self, texts: List[str],
                                      output_file: str = "tokenizer_training.txt") -> str:
        """
        Create a single file for training a custom tokenizer.
        
        Args:
            texts: List of text entries
            output_file: Output filename
            
        Returns:
            Path to the created file
        """
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        print(f"Created tokenizer training file: {output_path}")
        return str(output_path)


def preprocess_pipeline(input_file: str,
                       output_dir: str = "data/processed",
                       val_size: float = 0.1,
                       test_size: float = 0.1,
                       min_length: int = 5,
                       max_length: Optional[int] = None) -> Dict[str, str]:
    """
    Complete preprocessing pipeline.
    
    Args:
        input_file: Path to input data file (JSON)
        output_dir: Directory for processed data
        val_size: Validation set proportion
        test_size: Test set proportion
        min_length: Minimum text length
        max_length: Maximum text length
        
    Returns:
        Dictionary with paths to saved splits
    """
    print("=" * 50)
    print("Starting Preprocessing Pipeline")
    print("=" * 50)
    
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'texts' in data:
        texts = data['texts']
    elif isinstance(data, list):
        texts = data
    else:
        raise ValueError("Invalid data format")
    
    print(f"Loaded {len(texts)} texts from {input_file}")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(output_dir)
    
    # Preprocess
    processed_texts = preprocessor.preprocess_batch(
        texts,
        clean=True,
        remove_noise_flag=True,
        min_length=min_length,
        max_length=max_length
    )
    
    # Show statistics
    stats = preprocessor.get_statistics(processed_texts)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Create splits
    splits = preprocessor.create_train_val_split(
        processed_texts,
        val_size=val_size,
        test_size=test_size
    )
    
    # Save splits
    saved_paths = preprocessor.save_splits(splits)
    
    print("\nPreprocessing complete!")
    return saved_paths


if __name__ == "__main__":
    # Demo usage
    print("=" * 50)
    print("Data Preprocessing Module Demo")
    print("=" * 50)
    
    # Example: preprocess the sample dataset
    input_file = "data/raw/sample_hindi_hinglish.json"
    
    if os.path.exists(input_file):
        saved_paths = preprocess_pipeline(
            input_file=input_file,
            output_dir="data/processed",
            val_size=0.15,
            test_size=0.15,
            min_length=5
        )
        
        print("\nSaved files:")
        for split, path in saved_paths.items():
            print(f"  {split}: {path}")
    else:
        print(f"Input file not found: {input_file}")
        print("Please run data_collection.py first to create sample data.")
