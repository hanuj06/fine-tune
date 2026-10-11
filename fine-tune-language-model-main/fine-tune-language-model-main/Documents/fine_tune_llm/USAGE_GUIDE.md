# Usage Guide - Fine-Tuning Small Language Model

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Detailed Usage](#detailed-usage)
4. [Module-by-Module Guide](#module-by-module-guide)
5. [Configuration](#configuration)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Setup Steps

```bash
# Navigate to project directory
cd ~/Documents/fine_tune_llm

# Create virtual environment (recommended)
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Run Complete Pipeline (Easiest)

```bash
# This will run everything: data collection, preprocessing, training, evaluation, and inference
python main.py --mode full
```

This single command will:
1. ✅ Create sample Hindi/Hinglish dataset
2. ✅ Preprocess and split data
3. ✅ Download and load DistilGPT-2 model
4. ✅ Fine-tune the model (3 epochs)
5. ✅ Evaluate performance
6. ✅ Generate sample outputs

**Expected Time**: 15-30 minutes (depending on hardware)

---

## Detailed Usage

### Option 1: Full Pipeline

```bash
# Run with default settings
python main.py --mode full

# Skip inference demo at the end
python main.py --mode full --skip-inference

# Use custom configuration file
python main.py --mode full --config my_config.json
```

### Option 2: Step-by-Step Execution

```bash
# Step 1: Collect/Create Dataset
python main.py --mode collect

# Step 2: Preprocess Data
python main.py --mode preprocess

# Step 3: Train Model
python main.py --mode train

# Step 4: Evaluate Model
python main.py --mode evaluate

# Step 5: Run Inference
python main.py --mode inference
```

### Option 3: Interactive Inference

```bash
# Start interactive mode (after training)
python src/inference.py --model models/fine_tuned/final_model --interactive

# In interactive mode, you can:
# - Type prompts to generate text
# - Type 'settings' to adjust parameters
# - Type 'quit' to exit
```

### Option 4: Single Prompt Generation

```bash
# Generate from a single prompt
python src/inference.py --model models/fine_tuned/final_model --prompt "Aaj ka din"

# Generate multiple sequences
python src/inference.py --model models/fine_tuned/final_model --prompt "Machine learning" --num-sequences 3

# Adjust generation parameters
python src/inference.py --model models/fine_tuned/final_model \n  --prompt "भारत" \n  --temperature 0.9 \n  --max-length 150 \n  --top-k 50 \n  --top-p 0.95
```

---

## Module-by-Module Guide

### 1. Data Collection

```python
from src.data_collection import DataCollector, create_sample_dataset

# Create sample dataset
data_path = create_sample_dataset('data/raw')

# Or load your own data
collector = DataCollector('data/raw')
texts = collector.load_from_file('your_data.json')

# Get statistics
stats = collector.get_statistics()
print(stats)
```

### 2. Data Preprocessing

```python
from src.data_preprocessing import preprocess_pipeline

# Run preprocessing
saved_paths = preprocess_pipeline(
    input_file='data/raw/sample_hindi_hinglish.json',
    output_dir='data/processed',
    val_size=0.15,
    test_size=0.15
)

print(f"Train: {saved_paths['train']}")
print(f"Val: {saved_paths['val']}")
print(f"Test: {saved_paths['test']}")
```

### 3. Model Selection

```python
from src.model_selection import ModelSelector

# List available models
ModelSelector.list_available_models()

# Load a model
selector = ModelSelector('distilgpt2')
model, tokenizer = selector.load_model_and_tokenizer()

# Test generation (before fine-tuning)
text = selector.test_generation("Hello", max_length=50)
print(text)
```

### 4. Fine-Tuning

```python
from src.fine_tuning import FineTuner
import json

# Load training data
with open('data/processed/data_train.json', 'r') as f:
    train_data = json.load(f)['texts']

with open('data/processed/data_val.json', 'r') as f:
    val_data = json.load(f)['texts']

# Initialize fine-tuner
tuner = FineTuner(model, tokenizer, output_dir='models/fine_tuned')

# Train
history = tuner.train(
    train_texts=train_data,
    val_texts=val_data,
    epochs=3,
    batch_size=4,
    learning_rate=5e-5
)
```

### 5. Evaluation

```python
from src.evaluation import ModelEvaluator, load_model_from_path

# Initialize evaluator
evaluator = ModelEvaluator()

# Load models
base_model, base_tokenizer = load_model_from_path('models/base/distilgpt2')
finetuned_model, finetuned_tokenizer = load_model_from_path('models/fine_tuned/final_model')

# Compare models
prompts = ["Aaj ka din", "Machine learning", "भारत"]
comparison = evaluator.compare_models(
    base_model, base_tokenizer,
    finetuned_model, finetuned_tokenizer,
    prompts
)
```

### 6. Inference

```python
from src.inference import TextGenerator

# Initialize generator
generator = TextGenerator('models/fine_tuned/final_model')

# Generate text
texts = generator.generate(
    prompt="Aaj ka din",
    max_length=100,
    temperature=0.7,
    num_return_sequences=1
)

print(texts[0])

# Interactive mode
generator.interactive_mode()
```

---

## Configuration

### Editing config.py

The `config.py` file contains all configurable parameters:

```python
# Model Configuration
MODEL_CONFIG = {
    'model_name': 'distilgpt2',  # Change to 'gpt2' for larger model
    'load_in_8bit': False,  # Set True to save memory
}

# Training Configuration
TRAINING_CONFIG = {
    'epochs': 3,  # Increase for better quality
    'batch_size': 4,  # Reduce if out of memory
    'learning_rate': 5e-5,
    'max_seq_length': 128,
}

# Generation Configuration
GENERATION_CONFIG = {
    'temperature': 0.7,  # Higher = more creative
    'top_k': 50,
    'top_p': 0.9,
}
```

### Creating Custom Configuration

```json
// custom_config.json
{
  "model_name": "gpt2",
  "data": {
    "raw_dir": "data/raw",
    "processed_dir": "data/processed",
    "val_size": 0.2,
    "test_size": 0.1
  },
  "training": {
    "epochs": 5,
    "batch_size": 8,
    "learning_rate": 3e-5
  }
}
```

Use it with:
```bash
python main.py --mode full --config custom_config.json
```

---

## Examples

### Example 1: Quick Test with Sample Data

```bash
# Run full pipeline with sample data
python main.py --mode full

# After completion, test the model
python src/inference.py --model models/fine_tuned/final_model --prompt "Technology se"
```

### Example 2: Train on Custom Dataset

```bash
# 1. Place your data in data/raw/my_data.json
# Format: {"texts": ["text1", "text2", ...]}

# 2. Preprocess
python main.py --mode preprocess

# 3. Train
python main.py --mode train

# 4. Evaluate
python main.py --mode evaluate
```

### Example 3: Interactive Generation Session

```bash
python src/inference.py --model models/fine_tuned/final_model --interactive

# Then in interactive mode:
# Prompt: Aaj ka din bahut
# Prompt: Machine learning ek
# Prompt: settings  (to adjust parameters)
# Prompt: quit  (to exit)
```

### Example 4: Batch Generation

```python
from src.inference import TextGenerator

generator = TextGenerator('models/fine_tuned/final_model')

prompts = [
    "Aaj ka din",
    "Machine learning",
    "भारत एक",
    "Technology se"
]

results = generator.batch_generate(prompts, max_length=50)

for prompt, generated in zip(prompts, results):
    print(f"Prompt: {prompt}")
    print(f"Output: {generated[0]}")
    print("-" * 50)
```

---

## Troubleshooting

### Issue: Out of Memory Error

**Solution 1**: Reduce batch size
```python
# In config.py
TRAINING_CONFIG = {
    'batch_size': 2,  # or even 1
}
```

**Solution 2**: Reduce sequence length
```python
TRAINING_CONFIG = {
    'max_seq_length': 64,  # instead of 128
}
```

**Solution 3**: Use smaller model
```python
MODEL_CONFIG = {
    'model_name': 'distilgpt2',  # smallest option
}
```

### Issue: Training is Very Slow

**Check GPU availability**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
```

**Solution**: Ensure you're using GPU if available, or reduce dataset size for testing.

### Issue: Poor Generation Quality

**Solution 1**: Train for more epochs
```python
TRAINING_CONFIG = {
    'epochs': 5,  # or more
}
```

**Solution 2**: Add more training data
- Collect more Hindi/Hinglish text
- Add to `data/raw/` directory

**Solution 3**: Adjust generation parameters
```bash
python src/inference.py --model models/fine_tuned/final_model \n  --prompt "Your prompt" \n  --temperature 0.8 \n  --top-p 0.95
```

### Issue: Model Not Found

**Error**: `Model not found at models/fine_tuned/final_model`

**Solution**: Run training first
```bash
python main.py --mode full
```

### Issue: Import Errors

**Error**: `ModuleNotFoundError: No module named 'transformers'`

**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

---

## Tips for Best Results

1. **Start Small**: Use sample data first to test the pipeline
2. **Monitor Training**: Watch the loss values - they should decrease
3. **Experiment**: Try different temperatures and top_p values
4. **More Data**: Collect more Hindi/Hinglish text for better results
5. **Longer Training**: More epochs = better quality (but slower)
6. **Save Checkpoints**: The system saves checkpoints automatically
7. **Compare Models**: Always compare base vs fine-tuned outputs

---

## Command Reference

### Main Application
```bash
python main.py --mode [full|collect|preprocess|train|evaluate|inference]
python main.py --mode full --config CONFIG_FILE
python main.py --mode full --skip-inference
python main.py --mode full --no-sample
```

### Inference
```bash
python src/inference.py --model MODEL_PATH --interactive
python src/inference.py --model MODEL_PATH --prompt "TEXT"
python src/inference.py --model MODEL_PATH --prompt "TEXT" --num-sequences N
python src/inference.py --model MODEL_PATH --prompt "TEXT" --temperature T --max-length L
```

### Configuration
```bash
python config.py  # Print and save configuration
```

---

## Getting Help

```bash
# Main application help
python main.py --help

# Inference help
python src/inference.py --help

# View README
cat README.md

# View project summary
cat PROJECT_SUMMARY.md
```

---

**Happy Fine-Tuning! 🚀**