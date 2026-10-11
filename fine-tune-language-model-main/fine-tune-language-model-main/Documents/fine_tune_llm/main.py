#!/usr/bin/env python3
"""
Main Application
================
Orchestrates all modules for the fine-tuning pipeline.
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_collection import DataCollector, create_sample_dataset
from data_preprocessing import DataPreprocessor, preprocess_pipeline
from model_selection import ModelSelector
from fine_tuning import FineTuner
from evaluation import ModelEvaluator, load_model_from_path
from inference import TextGenerator


class FineTuningPipeline:
    """
    Main pipeline for fine-tuning language models.
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialize the pipeline.
        
        Args:
            config_file: Path to configuration file (optional)
        """
        self.config = self._load_config(config_file) if config_file else self._default_config()
        self.base_dir = Path.cwd()
        
    def _default_config(self) -> dict:
        """
        Get default configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            'model_name': 'distilgpt2',
            'data': {
                'raw_dir': 'data/raw',
                'processed_dir': 'data/processed',
                'val_size': 0.15,
                'test_size': 0.15,
                'min_length': 5,
                'max_length': 512
            },
            'training': {
                'epochs': 3,
                'batch_size': 4,
                'learning_rate': 5e-5,
                'max_seq_length': 128,
                'warmup_steps': 100
            },
            'output': {
                'models_dir': 'models',
                'outputs_dir': 'outputs'
            }
        }
    
    def _load_config(self, config_file: str) -> dict:
        """
        Load configuration from file.
        
        Args:
            config_file: Path to config file
            
        Returns:
            Configuration dictionary
        """
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def run_data_collection(self, create_sample: bool = True) -> str:
        """
        Run data collection phase.
        
        Args:
            create_sample: Whether to create sample dataset
            
        Returns:
            Path to collected data
        """
        print("\n" + "=" * 70)
        print("PHASE 1: Data Collection")
        print("=" * 70)
        
        if create_sample:
            data_path = create_sample_dataset(self.config['data']['raw_dir'])
            print(f"\nSample dataset created at: {data_path}")
            return data_path
        else:
            # Load existing data
            collector = DataCollector(self.config['data']['raw_dir'])
            texts = collector.load_from_directory()
            data_path = collector.save_collected_data()
            return data_path
    
    def run_preprocessing(self, input_file: str) -> dict:
        """
        Run data preprocessing phase.
        
        Args:
            input_file: Path to input data file
            
        Returns:
            Dictionary with paths to processed splits
        """
        print("\n" + "=" * 70)
        print("PHASE 2: Data Preprocessing")
        print("=" * 70)
        
        saved_paths = preprocess_pipeline(
            input_file=input_file,
            output_dir=self.config['data']['processed_dir'],
            val_size=self.config['data']['val_size'],
            test_size=self.config['data']['test_size'],
            min_length=self.config['data']['min_length'],
            max_length=self.config['data']['max_length']
        )
        
        return saved_paths
    
    def run_model_selection(self) -> tuple:
        """
        Run model selection phase.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        print("\n" + "=" * 70)
        print("PHASE 3: Model Selection")
        print("=" * 70)
        
        selector = ModelSelector(
            model_name=self.config['model_name'],
            cache_dir=str(self.base_dir / self.config['output']['models_dir'] / 'base')
        )
        
        model, tokenizer = selector.load_model_and_tokenizer()
        
        # Show model info
        info = selector.get_model_info()
        print("\nModel Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test generation before fine-tuning
        print("\nTesting generation (before fine-tuning):")
        test_prompts = ["Aaj ka din", "Machine learning"]
        for prompt in test_prompts:
            generated = selector.test_generation(prompt, max_length=30)
            print(f"\nPrompt: {prompt}")
            print(f"Output: {generated}")
        
        # Save base model for later evaluation
        base_model_save_path = self.base_dir / self.config['output']['models_dir'] / 'base' / self.config['model_name'].replace('/', '_')
        print(f"\nSaving base model to {base_model_save_path}...")
        selector.save_model(str(base_model_save_path))
        
        return model, tokenizer
    
    def run_fine_tuning(self, model, tokenizer, data_splits: dict) -> str:
        """
        Run fine-tuning phase.
        
        Args:
            model: Pre-trained model
            tokenizer: Tokenizer
            data_splits: Dictionary with paths to data splits
            
        Returns:
            Path to fine-tuned model
        """
        print("\n" + "=" * 70)
        print("PHASE 4: Fine-Tuning")
        print("=" * 70)
        
        # Load training data
        with open(data_splits['train'], 'r', encoding='utf-8') as f:
            train_data = json.load(f)['texts']
        
        with open(data_splits['val'], 'r', encoding='utf-8') as f:
            val_data = json.load(f)['texts']
        
        # Initialize fine-tuner
        output_dir = self.base_dir / self.config['output']['models_dir'] / 'fine_tuned'
        tuner = FineTuner(model, tokenizer, output_dir=str(output_dir))
        
        # Train
        history = tuner.train(
            train_texts=train_data,
            val_texts=val_data,
            epochs=self.config['training']['epochs'],
            batch_size=self.config['training']['batch_size'],
            learning_rate=self.config['training']['learning_rate'],
            max_length=self.config['training']['max_seq_length'],
            warmup_steps=self.config['training']['warmup_steps']
        )
        
        model_path = output_dir / 'final_model'
        return str(model_path)
    
    def run_evaluation(self, base_model_path: str, finetuned_model_path: str, 
                      data_splits: dict) -> dict:
        """
        Run evaluation phase.
        
        Args:
            base_model_path: Path to base model
            finetuned_model_path: Path to fine-tuned model
            data_splits: Dictionary with paths to data splits
            
        Returns:
            Evaluation results
        """
        print("\n" + "=" * 70)
        print("PHASE 5: Evaluation")
        print("=" * 70)
        
        # Initialize evaluator
        evaluator = ModelEvaluator()
        
        # Load models
        print("\nLoading base model...")
        base_model, base_tokenizer = load_model_from_path(base_model_path)
        
        print("\nLoading fine-tuned model...")
        finetuned_model, finetuned_tokenizer = load_model_from_path(finetuned_model_path)
        
        # Load test data
        with open(data_splits['test'], 'r', encoding='utf-8') as f:
            test_data = json.load(f)['texts']
        
        # Evaluate on test set
        print("\nEvaluating base model...")
        base_metrics = evaluator.evaluate_on_test_set(
            base_model, base_tokenizer, test_data
        )
        
        print("\nEvaluating fine-tuned model...")
        finetuned_metrics = evaluator.evaluate_on_test_set(
            finetuned_model, finetuned_tokenizer, test_data
        )
        
        # Compare on prompts
        test_prompts = [
            "Aaj ka din",
            "Machine learning",
            "भारत",
            "Technology se",
            "हिंदी भाषा"
        ]
        
        comparison = evaluator.compare_models(
            base_model, base_tokenizer,
            finetuned_model, finetuned_tokenizer,
            test_prompts,
            max_length=50
        )
        
        # Save results
        output_dir = self.base_dir / self.config['output']['outputs_dir']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        evaluator.save_comparison_results(
            comparison,
            str(output_dir / 'comparison_results.json')
        )
        
        evaluator.generate_evaluation_report(
            base_metrics,
            finetuned_metrics,
            comparison,
            str(output_dir / 'evaluation_report.txt')
        )
        
        return {
            'base_metrics': base_metrics,
            'finetuned_metrics': finetuned_metrics,
            'comparison': comparison
        }
    
    def run_inference_demo(self, model_path: str):
        """
        Run inference demonstration.
        
        Args:
            model_path: Path to fine-tuned model
        """
        print("\n" + "=" * 70)
        print("PHASE 6: Inference Demo")
        print("=" * 70)
        
        generator = TextGenerator(model_path)
        
        demo_prompts = [
            "Aaj ka din bahut",
            "Machine learning ek",
            "भारत एक",
            "Technology se life",
            "Python programming"
        ]
        
        print("\nGenerating text with fine-tuned model:\n")
        for prompt in demo_prompts:
            print(f"Prompt: {prompt}")
            generated = generator.generate(prompt, max_length=50, num_return_sequences=1)
            print(f"Output: {generated[0]}")
            print("-" * 70)
    
    def run_full_pipeline(self, create_sample: bool = True, 
                         run_inference: bool = True) -> dict:
        """
        Run the complete fine-tuning pipeline.
        
        Args:
            create_sample: Whether to create sample dataset
            run_inference: Whether to run inference demo
            
        Returns:
            Dictionary with pipeline results
        """
        print("\n" + "#" * 70)
        print("#" + "" * 68 + "#")
        print("#" + " " * 15 + "FINE-TUNING PIPELINE STARTED" + " " * 25 + "#")
        print("#" + " " * 68 + "#")
        print("#" * 70)
        
        results = {}
        
        try:
            # Phase 1: Data Collection
            data_path = self.run_data_collection(create_sample=create_sample)
            results['data_path'] = data_path
            
            # Phase 2: Data Preprocessing
            data_splits = self.run_preprocessing(data_path)
            results['data_splits'] = data_splits
            
            # Phase 3: Model Selection
            model, tokenizer = self.run_model_selection()
            base_model_path = str(self.base_dir / self.config['output']['models_dir'] / 'base' / self.config['model_name'])
            results['base_model_path'] = base_model_path
            
            # Phase 4: Fine-Tuning
            finetuned_model_path = self.run_fine_tuning(model, tokenizer, data_splits)
            results['finetuned_model_path'] = finetuned_model_path
            
            # Phase 5: Evaluation
            eval_results = self.run_evaluation(
                base_model_path,
                finetuned_model_path,
                data_splits
            )
            results['evaluation'] = eval_results
            
            # Phase 6: Inference Demo (optional)
            if run_inference:
                self.run_inference_demo(finetuned_model_path)
            
            print("\n" + "#" * 70)
            print("#" + " " * 68 + "#")
            print("#" + " " * 15 + "PIPELINE COMPLETED SUCCESSFULLY!" + " " * 20 + "#")
            print("#" + " " * 68 + "#")
            print("#" * 70)
            
            print("\nResults Summary:")
            print(f"  Fine-tuned model: {finetuned_model_path}")
            print(f"  Evaluation report: {self.config['output']['outputs_dir']}/evaluation_report.txt")
            print(f"  Comparison results: {self.config['output']['outputs_dir']}/comparison_results.json")
            
            return results
            
        except Exception as e:
            print(f"\n\nERROR: Pipeline failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return results


def main():
    """
    Main entry point for the application.
    """
    parser = argparse.ArgumentParser(
        description="Fine-Tune a Small Language Model on Regional/Custom Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with sample data
  python main.py --mode full
  
  # Run specific phases
  python main.py --mode collect
  python main.py --mode preprocess
  python main.py --mode train
  python main.py --mode evaluate
  python main.py --mode inference
  
  # Use custom configuration
  python main.py --mode full --config config.json
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'collect', 'preprocess', 'train', 'evaluate', 'inference'],
        default='full',
        help='Pipeline mode to run'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (JSON)'
    )
    
    parser.add_argument(
        '--no-sample',
        action='store_true',
        help='Do not create sample dataset (use existing data)'
    )
    
    parser.add_argument(
        '--skip-inference',
        action='store_true',
        help='Skip inference demo at the end'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = FineTuningPipeline(config_file=args.config)
    
    # Run based on mode
    if args.mode == 'full':
        pipeline.run_full_pipeline(
            create_sample=not args.no_sample,
            run_inference=not args.skip_inference
        )
    
    elif args.mode == 'collect':
        pipeline.run_data_collection(create_sample=not args.no_sample)
    
    elif args.mode == 'preprocess':
        data_path = 'data/raw/sample_hindi_hinglish.json'
        pipeline.run_preprocessing(data_path)
    
    elif args.mode == 'train':
        print("Training mode requires preprocessed data and base model.")
        print("Please run 'full' mode or run phases sequentially.")
    
    elif args.mode == 'evaluate':
        print("Evaluation mode requires both base and fine-tuned models.")
        print("Please run 'full' mode first.")
    
    elif args.mode == 'inference':
        model_path = 'models/fine_tuned/final_model'
        if Path(model_path).exists():
            pipeline.run_inference_demo(model_path)
        else:
            print(f"Model not found at {model_path}")
            print("Please run fine-tuning first.")


if __name__ == "__main__":
    main()
