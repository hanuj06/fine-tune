"""
Model Selection Module
======================
This module handles loading and configuring pre-trained language models
and their tokenizers for fine-tuning.
"""

import os
import torch
from typing import Optional, Tuple, Dict
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    PreTrainedTokenizer,
    PreTrainedModel
)


class ModelSelector:
    """
    A class to select and load pre-trained language models.
    Supports various small language models suitable for fine-tuning.
    """
    
    # Available models with their configurations
    AVAILABLE_MODELS = {
        'distilgpt2': {
            'name': 'distilgpt2',
            'description': 'Distilled GPT-2 (82M parameters) - Fast and efficient',
            'size': '82M',
            'recommended': True
        },
        'gpt2': {
            'name': 'gpt2',
            'description': 'GPT-2 Small (124M parameters) - Good balance',
            'size': '124M',
            'recommended': True
        },
        'gpt2-medium': {
            'name': 'gpt2-medium',
            'description': 'GPT-2 Medium (355M parameters) - Better quality',
            'size': '355M',
            'recommended': False
        },
        'gpt2-large': {
            'name': 'gpt2-large',
            'description': 'GPT-2 Large (774M parameters) - High quality, slower',
            'size': '774M',
            'recommended': False
        }
    }
    
    def __init__(self, model_name: str = 'distilgpt2', cache_dir: Optional[str] = None):
        """
        Initialize the ModelSelector.
        
        Args:
            model_name: Name of the pre-trained model to use
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model: Optional[PreTrainedModel] = None
        self.device = self._get_device()
        
        print(f"Initialized ModelSelector with model: {model_name}")
        print(f"Device: {self.device}")
    
    def _get_device(self) -> str:
        """
        Determine the best available device (GPU/CPU).
        
        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'  # Apple Silicon GPU
        else:
            return 'cpu'
    
    @classmethod
    def list_available_models(cls) -> None:
        """
        Print information about available models.
        """
        print("\nAvailable Models:")
        print("=" * 70)
        for key, info in cls.AVAILABLE_MODELS.items():
            recommended = " [RECOMMENDED]" if info['recommended'] else ""
            print(f"\n{key}{recommended}")
            print(f"  Description: {info['description']}")
            print(f"  Size: {info['size']}")
        print("=" * 70)
    
    def load_tokenizer(self, add_special_tokens: bool = True) -> PreTrainedTokenizer:
        """
        Load the tokenizer for the selected model.
        
        Args:
            add_special_tokens: Whether to add special tokens
            
        Returns:
            Loaded tokenizer
        """
        print(f"Loading tokenizer: {self.model_name}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Set padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"Tokenizer loaded successfully")
            print(f"  Vocabulary size: {len(self.tokenizer)}")
            print(f"  Special tokens: {self.tokenizer.special_tokens_map}")
            
            return self.tokenizer
            
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise
    
    def load_model(self, load_in_8bit: bool = False) -> PreTrainedModel:
        """
        Load the pre-trained language model.
        
        Args:
            load_in_8bit: Load model in 8-bit precision (saves memory)
            
        Returns:
            Loaded model
        """
        print(f"Loading model: {self.model_name}...")
        
        try:
            # Load model (removed load_in_8bit as it's not supported by GPT2)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            # Set model to training mode
            self.model.train()
            
            print(f"Model loaded successfully")
            print(f"  Parameters: {self.count_parameters():,}")
            print(f"  Device: {self.device}")
            
            return self.model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_model_and_tokenizer(self, 
                                load_in_8bit: bool = False) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load both model and tokenizer.
        
        Args:
            load_in_8bit: Load model in 8-bit precision
            
        Returns:
            Tuple of (model, tokenizer)
        """
        tokenizer = self.load_tokenizer()
        model = self.load_model(load_in_8bit=False)  # 8-bit loading not supported by GPT2
        return model, tokenizer
    
    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters in the model.
        
        Returns:
            Number of parameters
        """
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None or self.tokenizer is None:
            return {"status": "Model not loaded"}
        
        return {
            "model_name": self.model_name,
            "device": self.device,
            "parameters": self.count_parameters(),
            "vocab_size": len(self.tokenizer),
            "max_length": self.tokenizer.model_max_length,
            "pad_token": self.tokenizer.pad_token,
            "eos_token": self.tokenizer.eos_token,
            "bos_token": self.tokenizer.bos_token
        }
    
    def save_model(self, output_dir: str) -> None:
        """
        Save the model and tokenizer to a directory.
        
        Args:
            output_dir: Directory to save the model
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving model to {output_path}...")
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print("Model saved successfully")
    
    def test_generation(self, prompt: str, max_length: int = 50) -> str:
        """
        Test text generation with the loaded model.
        
        Args:
            prompt: Input prompt text
            max_length: Maximum length of generated text
            
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
        
        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text


def compare_models(prompts: list, model_names: list = ['distilgpt2', 'gpt2']) -> None:
    """
    Compare text generation across different models.
    
    Args:
        prompts: List of prompts to test
        model_names: List of model names to compare
    """
    print("\n" + "=" * 70)
    print("Model Comparison")
    print("=" * 70)
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 70)
        
        for model_name in model_names:
            try:
                selector = ModelSelector(model_name)
                selector.load_model_and_tokenizer()
                
                generated = selector.test_generation(prompt, max_length=50)
                print(f"\n{model_name}:")
                print(f"  {generated}")
                
            except Exception as e:
                print(f"\n{model_name}: Error - {e}")


if __name__ == "__main__":
    # Demo usage
    print("=" * 70)
    print("Model Selection Module Demo")
    print("=" * 70)
    
    # List available models
    ModelSelector.list_available_models()
    
    # Load a model
    print("\nLoading distilgpt2...")
    selector = ModelSelector('distilgpt2', cache_dir='models/base')
    
    try:
        model, tokenizer = selector.load_model_and_tokenizer()
        
        # Show model info
        info = selector.get_model_info()
        print("\nModel Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test generation (before fine-tuning)
        print("\nTesting generation (before fine-tuning):")
        test_prompts = [
            "Aaj ka din",
            "Machine learning",
            "भारत"
        ]
        
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            generated = selector.test_generation(prompt, max_length=30)
            print(f"Generated: {generated}")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Note: This demo requires internet connection to download models.")
