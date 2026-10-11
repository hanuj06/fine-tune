"""
Evaluation Module
=================
This module handles evaluation and comparison of model performance
before and after fine-tuning.
"""

import os
import json
import torch
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from transformers import PreTrainedModel, PreTrainedTokenizer
import time


class ModelEvaluator:
    """
    A class to evaluate and compare language models.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the ModelEvaluator.
        
        Args:
            device: Device to use (cuda/mps/cpu)
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        print(f"ModelEvaluator initialized on device: {self.device}")
    
    def generate_text(self,
                     model: PreTrainedModel,
                     tokenizer: PreTrainedTokenizer,
                     prompt: str,
                     max_length: int = 50,
                     temperature: float = 0.7,
                     top_k: int = 50,
                     top_p: float = 0.9,
                     num_return_sequences: int = 1) -> List[str]:
        """
        Generate text using the model.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            num_return_sequences: Number of sequences to generate
            
        Returns:
            List of generated texts
        """
        model.eval()
        model = model.to(self.device)
        
        # Encode prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_texts = []
        for output in outputs:
            text = tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def compare_models(self,
                      base_model: PreTrainedModel,
                      base_tokenizer: PreTrainedTokenizer,
                      finetuned_model: PreTrainedModel,
                      finetuned_tokenizer: PreTrainedTokenizer,
                      prompts: List[str],
                      max_length: int = 50,
                      temperature: float = 0.7) -> Dict:
        """
        Compare base model and fine-tuned model outputs.
        
        Args:
            base_model: Base pre-trained model
            base_tokenizer: Base model tokenizer
            finetuned_model: Fine-tuned model
            finetuned_tokenizer: Fine-tuned model tokenizer
            prompts: List of prompts to test
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Dictionary with comparison results
        """
        print("\n" + "=" * 70)
        print("Model Comparison: Base vs Fine-Tuned")
        print("=" * 70)
        
        results = {
            'prompts': [],
            'base_outputs': [],
            'finetuned_outputs': [],
            'comparison': []
        }
        
        for i, prompt in enumerate(prompts):
            print(f"\n[{i+1}/{len(prompts)}] Prompt: {prompt}")
            print("-" * 70)
            
            # Generate with base model
            base_output = self.generate_text(
                base_model, base_tokenizer, prompt,
                max_length=max_length, temperature=temperature
            )[0]
            
            # Generate with fine-tuned model
            finetuned_output = self.generate_text(
                finetuned_model, finetuned_tokenizer, prompt,
                max_length=max_length, temperature=temperature
            )[0]
            
            print(f"\nBase Model Output:")
            print(f"  {base_output}")
            print(f"\nFine-Tuned Model Output:")
            print(f"  {finetuned_output}")
            
            # Store results
            results['prompts'].append(prompt)
            results['base_outputs'].append(base_output)
            results['finetuned_outputs'].append(finetuned_output)
            results['comparison'].append({
                'prompt': prompt,
                'base': base_output,
                'finetuned': finetuned_output
            })
        
        return results
    
    def calculate_perplexity(self,
                           model: PreTrainedModel,
                           tokenizer: PreTrainedTokenizer,
                           texts: List[str],
                           max_length: int = 128) -> float:
        """
        Calculate perplexity on a set of texts.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            texts: List of texts to evaluate
            max_length: Maximum sequence length
            
        Returns:
            Average perplexity
        """
        model.eval()
        model = model.to(self.device)
        
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Forward pass
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
                # Accumulate
                total_loss += loss.item() * inputs['input_ids'].size(1)
                total_tokens += inputs['input_ids'].size(1)
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def evaluate_on_test_set(self,
                           model: PreTrainedModel,
                           tokenizer: PreTrainedTokenizer,
                           test_texts: List[str],
                           max_length: int = 128) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            test_texts: Test texts
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating on {len(test_texts)} test samples...")
        
        start_time = time.time()
        perplexity = self.calculate_perplexity(model, tokenizer, test_texts, max_length)
        elapsed_time = time.time() - start_time
        
        results = {
            'perplexity': perplexity,
            'num_samples': len(test_texts),
            'evaluation_time': elapsed_time
        }
        
        print(f"Perplexity: {perplexity:.2f}")
        print(f"Evaluation time: {elapsed_time:.2f}s")
        
        return results
    
    def qualitative_evaluation(self,
                             model: PreTrainedModel,
                             tokenizer: PreTrainedTokenizer,
                             prompts: List[str],
                             num_samples: int = 3,
                             max_length: int = 50) -> Dict:
        """
        Perform qualitative evaluation with multiple samples per prompt.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            prompts: List of prompts
            num_samples: Number of samples to generate per prompt
            max_length: Maximum generation length
            
        Returns:
            Dictionary with generated samples
        """
        print("\n" + "=" * 70)
        print("Qualitative Evaluation")
        print("=" * 70)
        
        results = {}
        
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            print("-" * 70)
            
            generated_texts = self.generate_text(
                model, tokenizer, prompt,
                max_length=max_length,
                num_return_sequences=num_samples
            )
            
            results[prompt] = generated_texts
            
            for i, text in enumerate(generated_texts, 1):
                print(f"\nSample {i}:")
                print(f"  {text}")
        
        return results
    
    def save_comparison_results(self,
                              results: Dict,
                              output_file: str = "outputs/comparison_results.json") -> None:
        """
        Save comparison results to a file.
        
        Args:
            results: Comparison results dictionary
            output_file: Output file path
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nComparison results saved to {output_path}")
    
    def generate_evaluation_report(self,
                                 base_metrics: Dict,
                                 finetuned_metrics: Dict,
                                 comparison_results: Dict,
                                 output_file: str = "outputs/evaluation_report.txt") -> None:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            base_metrics: Metrics for base model
            finetuned_metrics: Metrics for fine-tuned model
            comparison_results: Comparison results
            output_file: Output file path
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            # Quantitative metrics
            f.write("QUANTITATIVE METRICS\n")
            f.write("-" * 70 + "\n\n")
            
            f.write("Base Model:\n")
            for key, value in base_metrics.items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\nFine-Tuned Model:\n")
            for key, value in finetuned_metrics.items():
                f.write(f"  {key}: {value}\n")
            
            # Improvement
            if 'perplexity' in base_metrics and 'perplexity' in finetuned_metrics:
                improvement = ((base_metrics['perplexity'] - finetuned_metrics['perplexity']) 
                             / base_metrics['perplexity'] * 100)
                f.write(f"\nPerplexity Improvement: {improvement:.2f}%\n")
            
            # Qualitative comparison
            f.write("\n\n" + "=" * 70 + "\n")
            f.write("QUALITATIVE COMPARISON\n")
            f.write("=" * 70 + "\n\n")
            
            for i, comp in enumerate(comparison_results.get('comparison', []), 1):
                f.write(f"Example {i}:\n")
                f.write(f"Prompt: {comp['prompt']}\n\n")
                f.write(f"Base Model Output:\n{comp['base']}\n\n")
                f.write(f"Fine-Tuned Model Output:\n{comp['finetuned']}\n\n")
                f.write("-" * 70 + "\n\n")
        
        print(f"\nEvaluation report saved to {output_path}")


def load_model_from_path(model_path: str, device: Optional[str] = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a model and tokenizer from a path.
    
    Args:
        model_path: Path to model directory
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    if device:
        model = model.to(device)
    
    print("Model loaded successfully!")
    return model, tokenizer


if __name__ == "__main__":
    # Demo usage
    print("=" * 70)
    print("Evaluation Module Demo")
    print("=" * 70)
    
    print("\nThis module requires:")
    print("  1. A base pre-trained model")
    print("  2. A fine-tuned model")
    print("  3. Test prompts for comparison")
    print("\nExample usage:")
    print("""
    from evaluation import ModelEvaluator, load_model_from_path
    import json
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load models
    base_model, base_tokenizer = load_model_from_path('models/base/distilgpt2')
    finetuned_model, finetuned_tokenizer = load_model_from_path('models/fine_tuned/final_model')
    
    # Load test data
    with open('data/processed/data_test.json', 'r') as f:
        test_texts = json.load(f)['texts']
    
    # Evaluate on test set
    base_metrics = evaluator.evaluate_on_test_set(base_model, base_tokenizer, test_texts)
    finetuned_metrics = evaluator.evaluate_on_test_set(finetuned_model, finetuned_tokenizer, test_texts)
    
    # Compare on prompts
    prompts = [
        "Aaj ka din",
        "Machine learning",
        "भारत"
    ]
    
    comparison = evaluator.compare_models(
        base_model, base_tokenizer,
        finetuned_model, finetuned_tokenizer,
        prompts
    )
    
    # Generate report
    evaluator.generate_evaluation_report(base_metrics, finetuned_metrics, comparison)
    """)
