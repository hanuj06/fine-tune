"""
Inference & Demonstration Module
=================================
This module provides text generation capabilities and a CLI interface
for interacting with fine-tuned language models.
"""

import os
import sys
import torch
import argparse
from typing import Optional, List
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


class TextGenerator:
    """
    A class for text generation using fine-tuned language models.
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the TextGenerator.
        
        Args:
            model_path: Path to the fine-tuned model
            device: Device to use (cuda/mps/cpu)
        """
        self.model_path = Path(model_path)
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        print(f"Loading model from {self.model_path}...")
        self.model, self.tokenizer = self._load_model()
        print(f"Model loaded successfully on {self.device}!")
    
    def _load_model(self) -> tuple:
        """
        Load the model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            model = model.to(self.device)
            model.eval()
            
            return model, tokenizer
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate(self,
                prompt: str,
                max_length: int = 100,
                temperature: float = 0.7,
                top_k: int = 50,
                top_p: float = 0.9,
                num_return_sequences: int = 1,
                do_sample: bool = True) -> List[str]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt text
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            num_return_sequences: Number of sequences to generate
            do_sample: Whether to use sampling (vs greedy decoding)
            
        Returns:
            List of generated texts
        """
        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature if do_sample else 1.0,
                top_k=top_k if do_sample else 0,
                top_p=top_p if do_sample else 1.0,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
        
        # Decode
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def interactive_mode(self):
        """
        Start an interactive text generation session.
        """
        print("\n" + "=" * 70)
        print("Interactive Text Generation Mode")
        print("=" * 70)
        print("\nCommands:")
        print("  - Type a prompt to generate text")
        print("  - Type 'settings' to adjust generation parameters")
        print("  - Type 'quit' or 'exit' to exit")
        print("\n" + "=" * 70 + "\n")
        
        # Default settings
        settings = {
            'max_length': 100,
            'temperature': 0.7,
            'top_k': 50,
            'top_p': 0.9,
            'num_sequences': 1
        }
        
        while True:
            try:
                prompt = input("\nPrompt: ").strip()
                
                if not prompt:
                    continue
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("\nExiting interactive mode. Goodbye!")
                    break
                
                if prompt.lower() == 'settings':
                    self._adjust_settings(settings)
                    continue
                
                # Generate text
                print("\nGenerating...")
                generated_texts = self.generate(
                    prompt,
                    max_length=settings['max_length'],
                    temperature=settings['temperature'],
                    top_k=settings['top_k'],
                    top_p=settings['top_p'],
                    num_return_sequences=settings['num_sequences']
                )
                
                # Display results
                print("\n" + "-" * 70)
                for i, text in enumerate(generated_texts, 1):
                    if settings['num_sequences'] > 1:
                        print(f"\nGeneration {i}:")
                    print(text)
                print("-" * 70)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Exiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")
    
    def _adjust_settings(self, settings: dict):
        """
        Adjust generation settings interactively.
        
        Args:
            settings: Settings dictionary to modify
        """
        print("\n" + "=" * 70)
        print("Current Settings:")
        print("=" * 70)
        for key, value in settings.items():
            print(f"  {key}: {value}")
        print("=" * 70)
        
        print("\nEnter new values (press Enter to keep current):")
        
        try:
            # Max length
            val = input(f"max_length [{settings['max_length']}]: ").strip()
            if val:
                settings['max_length'] = int(val)
            
            # Temperature
            val = input(f"temperature [{settings['temperature']}]: ").strip()
            if val:
                settings['temperature'] = float(val)
            
            # Top-k
            val = input(f"top_k [{settings['top_k']}]: ").strip()
            if val:
                settings['top_k'] = int(val)
            
            # Top-p
            val = input(f"top_p [{settings['top_p']}]: ").strip()
            if val:
                settings['top_p'] = float(val)
            
            # Number of sequences
            val = input(f"num_sequences [{settings['num_sequences']}]: ").strip()
            if val:
                settings['num_sequences'] = int(val)
            
            print("\nSettings updated!")
            
        except ValueError as e:
            print(f"\nInvalid input: {e}")
            print("Settings not changed.")
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[List[str]]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of prompts
            **kwargs: Generation parameters
            
        Returns:
            List of lists of generated texts
        """
        results = []
        for prompt in prompts:
            generated = self.generate(prompt, **kwargs)
            results.append(generated)
        return results


def cli_interface():
    """
    Command-line interface for text generation.
    """
    parser = argparse.ArgumentParser(
        description="Text Generation with Fine-Tuned Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python inference.py --model models/fine_tuned/final_model --interactive
  
  # Single prompt
  python inference.py --model models/fine_tuned/final_model --prompt "Aaj ka din"
  
  # Multiple generations
  python inference.py --model models/fine_tuned/final_model --prompt "Machine learning" --num-sequences 3
  
  # Adjust parameters
  python inference.py --model models/fine_tuned/final_model --prompt "भारत" --temperature 0.9 --max-length 150
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the fine-tuned model directory'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        help='Input prompt for text generation'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Start interactive mode'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=100,
        help='Maximum length of generated text (default: 100)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (default: 0.7)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='Top-k sampling parameter (default: 50)'
    )
    
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.9,
        help='Top-p sampling parameter (default: 0.9)'
    )
    
    parser.add_argument(
        '--num-sequences',
        type=int,
        default=1,
        help='Number of sequences to generate (default: 1)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'mps', 'cpu'],
        help='Device to use for inference'
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    try:
        generator = TextGenerator(args.model, device=args.device)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Interactive mode
    if args.interactive:
        generator.interactive_mode()
    
    # Single prompt mode
    elif args.prompt:
        print(f"\nPrompt: {args.prompt}")
        print("\nGenerating...\n")
        
        generated_texts = generator.generate(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=args.num_sequences
        )
        
        print("=" * 70)
        for i, text in enumerate(generated_texts, 1):
            if args.num_sequences > 1:
                print(f"\nGeneration {i}:")
            print(text)
            if i < len(generated_texts):
                print("\n" + "-" * 70)
        print("=" * 70)
    
    else:
        print("Error: Please provide either --prompt or --interactive flag")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    # Check if running as CLI
    if len(sys.argv) > 1:
        cli_interface()
    else:
        # Demo usage
        print("=" * 70)
        print("Inference & Demonstration Module")
        print("=" * 70)
        print("\nUsage:")
        print("\n1. Command-line interface:")
        print("   python inference.py --model <model_path> --interactive")
        print("   python inference.py --model <model_path> --prompt \"Your prompt\"")
        print("\n2. Python API:")
        print("""
    from inference import TextGenerator
    
    # Initialize generator
    generator = TextGenerator('models/fine_tuned/final_model')
    
    # Generate text
    texts = generator.generate(
        prompt="Aaj ka din",
        max_length=100,
        temperature=0.7
    )
    
    print(texts[0])
    
    # Interactive mode
    generator.interactive_mode()
        """)
        print("\n" + "=" * 70)
        print("\nFor help, run: python inference.py --help")
