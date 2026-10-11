"""
Data Collection Module
======================
This module handles collection and loading of regional or domain-specific text data.
Supported formats: .txt, .csv, .json
"""

import os
import json
import pandas as pd
from typing import List, Dict, Optional, Union
from pathlib import Path


class DataCollector:
    """
    A class to collect and manage text data from various sources.
    Supports .txt, .csv, and .json file formats.
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the DataCollector.
        
        Args:
            data_dir: Directory to store/load raw data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.collected_data: List[str] = []
        
    def load_txt(self, file_path: str, encoding: str = "utf-8") -> List[str]:
        """
        Load text data from a .txt file.
        
        Args:
            file_path: Path to the .txt file
            encoding: File encoding (default: utf-8)
            
        Returns:
            List of text lines
        """
        texts = []
        with open(file_path, 'r', encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    texts.append(line)
        print(f"Loaded {len(texts)} lines from {file_path}")
        return texts
    
    def load_csv(self, file_path: str, text_column: str = "text", 
                 encoding: str = "utf-8") -> List[str]:
        """
        Load text data from a .csv file.
        
        Args:
            file_path: Path to the .csv file
            text_column: Name of the column containing text
            encoding: File encoding (default: utf-8)
            
        Returns:
            List of text entries
        """
        df = pd.read_csv(file_path, encoding=encoding)
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV. "
                           f"Available columns: {list(df.columns)}")
        texts = df[text_column].dropna().tolist()
        print(f"Loaded {len(texts)} entries from {file_path}")
        return texts
    
    def load_json(self, file_path: str, text_key: str = "text") -> List[str]:
        """
        Load text data from a .json file.
        
        Args:
            file_path: Path to the .json file
            text_key: Key containing text in JSON objects
            
        Returns:
            List of text entries
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and text_key in item:
                    texts.append(item[text_key])
                elif isinstance(item, str):
                    texts.append(item)
        elif isinstance(data, dict):
            if text_key in data:
                if isinstance(data[text_key], list):
                    texts = data[text_key]
                else:
                    texts = [data[text_key]]
        
        print(f"Loaded {len(texts)} entries from {file_path}")
        return texts
    
    def load_from_file(self, file_path: str, **kwargs) -> List[str]:
        """
        Automatically detect file format and load data.
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments for specific loaders
            
        Returns:
            List of text entries
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension == '.txt':
            return self.load_txt(str(file_path), **kwargs)
        elif extension == '.csv':
            return self.load_csv(str(file_path), **kwargs)
        elif extension == '.json':
            return self.load_json(str(file_path), **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {extension}. "
                           f"Supported formats: .txt, .csv, .json")
    
    def load_from_directory(self, directory: str = None, 
                           extensions: List[str] = None) -> List[str]:
        """
        Load all text data from a directory.
        
        Args:
            directory: Directory path (default: self.data_dir)
            extensions: List of file extensions to load
            
        Returns:
            Combined list of all text entries
        """
        if directory is None:
            directory = self.data_dir
        else:
            directory = Path(directory)
            
        if extensions is None:
            extensions = ['.txt', '.csv', '.json']
        
        all_texts = []
        for ext in extensions:
            for file_path in directory.glob(f"*{ext}"):
                texts = self.load_from_file(str(file_path))
                all_texts.extend(texts)
        
        print(f"Total loaded: {len(all_texts)} text entries from {directory}")
        return all_texts
    
    def add_text(self, text: Union[str, List[str]]) -> None:
        """
        Add text data to the collection.
        
        Args:
            text: Single text string or list of texts
        """
        if isinstance(text, str):
            self.collected_data.append(text)
        else:
            self.collected_data.extend(text)
        print(f"Added {len(text) if isinstance(text, list) else 1} entries. "
              f"Total: {len(self.collected_data)}")
    
    def save_collected_data(self, filename: str = "collected_data.json") -> str:
        """
        Save collected data to a JSON file.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.data_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({"texts": self.collected_data}, f, 
                     ensure_ascii=False, indent=2)
        print(f"Saved {len(self.collected_data)} entries to {output_path}")
        return str(output_path)
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning and normalization.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about collected data.
        
        Returns:
            Dictionary with data statistics
        """
        if not self.collected_data:
            return {"total_entries": 0}
        
        lengths = [len(text) for text in self.collected_data]
        word_counts = [len(text.split()) for text in self.collected_data]
        
        return {
            "total_entries": len(self.collected_data),
            "total_characters": sum(lengths),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "total_words": sum(word_counts),
            "avg_words": sum(word_counts) / len(word_counts)
        }


def create_sample_dataset(output_dir: str = "data/raw") -> str:
    """
    Create a sample Hindi/Hinglish dataset for demonstration.
    
    Args:
        output_dir: Directory to save the sample dataset
        
    Returns:
        Path to the created dataset
    """
    # Sample Hindi/Hinglish sentences for fine-tuning
    sample_texts = [
        # Hindi sentences
        "भारत एक विशाल और विविधताओं से भरा देश है।",
        "हिमालय की चोटियाँ बहुत सुंदर हैं।",
        "मुझे हिंदी भाषा बहुत पसंद है।",
        "आज मौसम बहुत अच्छा है।",
        "प्रौद्योगिकी का विकास बहुत जरूरी है।",
        "शिक्षा सबका अधिकार है।",
        "स्वास्थ्य ही सबसे बड़ा धन है।",
        "परिवार सबसे महत्वपूर्ण है।",
        "समय का सदुपयोग करना चाहिए।",
        "मेहनत का कोई विकल्प नहीं है।",
        
        # Hinglish sentences
        "Aaj ka din bahut accha hai.",
        "Mujhe coding karna bahut pasand hai.",
        "Machine learning ek interesting field hai.",
        "Python programming seekhna easy hai.",
        "Artificial intelligence future hai.",
        "Data science mein career bahut bright hai.",
        "Technology se life easy ho gayi hai.",
        "Internet ne duniya ko connect kar diya.",
        "Mobile phones bahut useful hain.",
        "Social media ka impact bahut bada hai.",
        
        # More Hindi sentences
        "कंप्यूटर विज्ञान बहुत रोचक है।",
        "कृत्रिम बुद्धिमत्ता का युग आ गया है।",
        "भाषा मॉडल बहुत उपयोगी हैं।",
        "डीप लर्निंग से बहुत कुछ संभव है।",
        "न्यूरल नेटवर्क की शक्ति अद्भुत है।",
        
        # More Hinglish sentences
        "AI models ko train karna interesting hai.",
        "Fine-tuning se model better ho jata hai.",
        "Dataset preparation important step hai.",
        "Transformers architecture revolutionary hai.",
        "NLP mein bahut scope hai.",
        "Language models text generate kar sakte hain.",
        "Pre-trained models time bachate hain.",
        "GPU se training fast hoti hai.",
        "Tokenization pehla step hai.",
        "Loss function ko minimize karna goal hai."
    ]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    json_path = output_path / "sample_hindi_hinglish.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({"texts": sample_texts}, f, ensure_ascii=False, indent=2)
    
    # Also save as TXT
    txt_path = output_path / "sample_hindi_hinglish.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        for text in sample_texts:
            f.write(text + '\n')
    
    print(f"Created sample dataset with {len(sample_texts)} entries")
    print(f"Saved to: {json_path} and {txt_path}")
    
    return str(json_path)


if __name__ == "__main__":
    # Demo usage
    print("=" * 50)
    print("Data Collection Module Demo")
    print("=" * 50)
    
    # Create sample dataset
    sample_path = create_sample_dataset()
    
    # Initialize collector
    collector = DataCollector()
    
    # Load the sample data
    texts = collector.load_from_file(sample_path)
    collector.add_text(texts)
    
    # Show statistics
    stats = collector.get_statistics()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
