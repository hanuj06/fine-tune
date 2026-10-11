"""
Configuration Settings
======================
Centralized configuration for the fine-tuning project.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    # Model selection
    'model_name': 'distilgpt2',  # Options: 'distilgpt2', 'gpt2', 'gpt2-medium'
    
    # Model paths
    'base_model_dir': BASE_DIR / 'models' / 'base',
    'finetuned_model_dir': BASE_DIR / 'models' / 'fine_tuned',
    
    # Model parameters
    'load_in_8bit': False,  # Use 8-bit quantization (saves memory)
}

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

DATA_CONFIG = {
    # Data directories
    'raw_data_dir': BASE_DIR / 'data' / 'raw',
    'processed_data_dir': BASE_DIR / 'data' / 'processed',
    
    # Data split ratios
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    
    # Data preprocessing
    'min_text_length': 5,  # Minimum characters
    'max_text_length': 512,  # Maximum characters
    'remove_duplicates': True,
    'remove_urls': True,
    'remove_emails': True,
    
    # Supported file formats
    'supported_formats': ['.txt', '.csv', '.json'],
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    # Training hyperparameters
    'epochs': 3,
    'batch_size': 4,  # Reduce if running out of memory
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    
    # Sequence length
    'max_seq_length': 128,  # Maximum tokens per sequence
    
    # Optimization
    'warmup_steps': 100,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,  # Gradient clipping
    
    # Scheduler
    'scheduler_type': 'linear',  # Options: 'linear', 'cosine', 'constant'
    
    # Checkpointing
    'save_steps': 500,
    'save_total_limit': 3,  # Keep only last 3 checkpoints
    'eval_steps': 500,
    
    # Early stopping
    'early_stopping_patience': 3,
    'early_stopping_threshold': 0.001,
}

# ============================================================================
# GENERATION CONFIGURATION
# ============================================================================

GENERATION_CONFIG = {
    # Generation parameters
    'max_length': 100,
    'min_length': 10,
    'temperature': 0.7,  # Higher = more random, Lower = more deterministic
    'top_k': 50,  # Top-k sampling
    'top_p': 0.9,  # Nucleus sampling
    'repetition_penalty': 1.2,  # Penalize repetition
    'num_return_sequences': 1,
    'do_sample': True,  # Use sampling vs greedy decoding
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

EVALUATION_CONFIG = {
    # Evaluation metrics
    'calculate_perplexity': True,
    'calculate_bleu': False,  # Requires reference texts
    
    # Test prompts for qualitative evaluation
    'test_prompts': [
        # Hindi prompts
        "भारत एक",
        "हिंदी भाषा",
        "शिक्षा का",
        "प्रौद्योगिकी से",
        
        # Hinglish prompts
        "Aaj ka din",
        "Machine learning",
        "Python programming",
        "Technology se life",
        "Data science mein",
        
        # Mixed prompts
        "कंप्यूटर विज्ञान",
        "Artificial intelligence",
    ],
    
    # Number of samples per prompt
    'num_samples_per_prompt': 3,
}

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

OUTPUT_CONFIG = {
    # Output directories
    'outputs_dir': BASE_DIR / 'outputs',
    'logs_dir': BASE_DIR / 'outputs' / 'logs',
    'results_dir': BASE_DIR / 'outputs' / 'results',
    
    # Output files
    'training_history_file': 'training_history.json',
    'evaluation_report_file': 'evaluation_report.txt',
    'comparison_results_file': 'comparison_results.json',
    
    # Logging
    'log_level': 'INFO',  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'log_to_file': True,
    'log_to_console': True,
}

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

DEVICE_CONFIG = {
    # Device selection (auto-detect by default)
    'device': None,  # Options: None (auto), 'cuda', 'mps', 'cpu'
    
    # Mixed precision training
    'use_fp16': False,  # Use 16-bit floating point (faster, less memory)
    
    # Multi-GPU
    'use_multi_gpu': False,
}

# ============================================================================
# LANGUAGE CONFIGURATION
# ============================================================================

LANGUAGE_CONFIG = {
    # Target languages/domains
    'languages': ['Hindi', 'Hinglish', 'English'],
    
    # Language-specific settings
    'hindi_unicode_range': ('\u0900', '\u097F'),
    'devanagari_support': True,
    
    # Domain
    'domain': 'general',  # Options: 'general', 'technical', 'conversational'
}

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

EXPERIMENT_CONFIG = {
    # Experiment tracking
    'experiment_name': 'hindi_hinglish_finetuning',
    'run_name': None,  # Auto-generated if None
    
    # Reproducibility
    'random_seed': 42,
    
    # Weights & Biases (optional)
    'use_wandb': False,
    'wandb_project': 'language-model-finetuning',
    'wandb_entity': None,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config(config_name: str = None) -> dict:
    """
    Get configuration dictionary.
    
    Args:
        config_name: Name of config to get (e.g., 'MODEL', 'TRAINING')
                    If None, returns all configs
    
    Returns:
        Configuration dictionary
    """
    all_configs = {
        'MODEL': MODEL_CONFIG,
        'DATA': DATA_CONFIG,
        'TRAINING': TRAINING_CONFIG,
        'GENERATION': GENERATION_CONFIG,
        'EVALUATION': EVALUATION_CONFIG,
        'OUTPUT': OUTPUT_CONFIG,
        'DEVICE': DEVICE_CONFIG,
        'LANGUAGE': LANGUAGE_CONFIG,
        'EXPERIMENT': EXPERIMENT_CONFIG,
    }
    
    if config_name is None:
        return all_configs
    
    return all_configs.get(config_name.upper(), {})


def create_directories():
    """
    Create all necessary directories for the project.
    """
    directories = [
        DATA_CONFIG['raw_data_dir'],
        DATA_CONFIG['processed_data_dir'],
        MODEL_CONFIG['base_model_dir'],
        MODEL_CONFIG['finetuned_model_dir'],
        OUTPUT_CONFIG['outputs_dir'],
        OUTPUT_CONFIG['logs_dir'],
        OUTPUT_CONFIG['results_dir'],
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("All directories created successfully!")


def print_config():
    """
    Print all configuration settings.
    """
    print("=" * 70)
    print("CONFIGURATION SETTINGS")
    print("=" * 70)
    
    configs = get_config()
    
    for config_name, config_dict in configs.items():
        print(f"\n{config_name} CONFIG:")
        print("-" * 70)
        for key, value in config_dict.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)


def save_config_to_file(output_file: str = 'config_snapshot.json'):
    """
    Save current configuration to a JSON file.
    
    Args:
        output_file: Output filename
    """
    import json
    
    configs = get_config()
    
    # Convert Path objects to strings for JSON serialization
    serializable_configs = {}
    for config_name, config_dict in configs.items():
        serializable_configs[config_name] = {}
        for key, value in config_dict.items():
            if isinstance(value, Path):
                serializable_configs[config_name][key] = str(value)
            else:
                serializable_configs[config_name][key] = value
    
    output_path = OUTPUT_CONFIG['outputs_dir'] / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_configs, f, indent=2)
    
    print(f"Configuration saved to {output_path}")


if __name__ == "__main__":
    # Demo: Print configuration
    print_config()
    
    # Create directories
    print("\nCreating project directories...")
    create_directories()
    
    # Save config snapshot
    print("\nSaving configuration snapshot...")
    save_config_to_file()
