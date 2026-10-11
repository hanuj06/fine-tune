# Fine-Tuning Small Language Model - Project Summary

## Project Overview

This project implements a complete pipeline for fine-tuning small language models on regional or custom language datasets (Hindi, Punjabi, Tamil, Hinglish, or domain-specific text).

## Project Structure

```
fine_tune_llm/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── config.py                    # Configuration settings
├── main.py                      # Main application entry point
├── .gitignore                   # Git ignore file
├── PROJECT_SUMMARY.md           # This file
│
├── src/                         # Source code modules
│   ├── data_collection.py       # Dataset collection module
│   ├── data_preprocessing.py    # Data preprocessing module
│   ├── model_selection.py       # Model selection module
│   ├── fine_tuning.py          # Fine-tuning module
│   ├── evaluation.py           # Evaluation module
│   └── inference.py            # Inference & demonstration module
│
├── data/                        # Dataset storage
│   ├── raw/                    # Raw collected data
│   │   └── sample_hindi_hinglish.json
│   └── processed/              # Preprocessed data (generated)
│
├── models/                      # Model storage
│   ├── base/                   # Base pre-trained models (downloaded)
│   └── fine_tuned/             # Fine-tuned model checkpoints (generated)
│
└── outputs/                     # Training outputs and logs
    ├── logs/                   # Training logs
    └── results/                # Evaluation results
```

## Modules Description

### 1. Data Collection Module (`data_collection.py`)
- Collects regional or domain-specific text data
- Supports .txt, .csv, .json formats
- Includes sample Hindi/Hinglish dataset
- Features:
  - Load data from multiple file formats
  - Basic text cleaning and normalization
  - Dataset statistics

### 2. Data Preprocessing Module (`data_preprocessing.py`)
- Tokenization and text cleaning
- Noise removal (URLs, emails, duplicates)
- Train-validation-test split
- Features:
  - Configurable preprocessing pipeline
  - Statistical analysis
  - Multiple output formats

### 3. Model Selection Module (`model_selection.py`)
- Load lightweight pre-trained models
- Supported models:
  - DistilGPT-2 (82M parameters) - Recommended
  - GPT-2 Small (124M parameters)
  - GPT-2 Medium (355M parameters)
  - GPT-2 Large (774M parameters)
- Features:
  - Automatic device detection (CUDA/MPS/CPU)
  - Model information and statistics
  - Pre-training generation testing

### 4. Fine-Tuning Module (`fine_tuning.py`)
- Complete training pipeline
- Features:
  - Custom dataset handling
  - Training loop with progress tracking
  - Loss monitoring
  - Checkpoint management
  - Learning rate scheduling
  - Gradient clipping
  - Validation during training

### 5. Evaluation Module (`evaluation.py`)
- Compare base vs fine-tuned models
- Features:
  - Perplexity calculation
  - Qualitative comparison
  - Side-by-side generation comparison
  - Comprehensive evaluation reports

### 6. Inference Module (`inference.py`)
- Text generation with fine-tuned models
- Features:
  - Interactive CLI mode
  - Batch generation
  - Configurable generation parameters
  - Command-line interface

### 7. Main Application (`main.py`)
- Orchestrates all modules
- Complete end-to-end pipeline
- Features:
  - Full pipeline execution
  - Individual phase execution
  - Configuration management

### 8. Configuration (`config.py`)
- Centralized settings
- Configurations for:
  - Model selection
  - Data processing
  - Training hyperparameters
  - Generation parameters
  - Evaluation settings
  - Output paths

## Quick Start Guide

### Installation

```bash
cd fine_tune_llm

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Option 1: Run Full Pipeline

```bash
# Run complete pipeline with sample data
python main.py --mode full
```

This will:
1. Create/load sample Hindi/Hinglish dataset
2. Preprocess the data
3. Load base model (DistilGPT-2)
4. Fine-tune the model
5. Evaluate performance
6. Run inference demo

#### Option 2: Run Individual Phases

```bash
# Data collection
python main.py --mode collect

# Data preprocessing
python main.py --mode preprocess

# Training (requires preprocessed data)
python main.py --mode train

# Evaluation (requires trained model)
python main.py --mode evaluate

# Inference (requires trained model)
python main.py --mode inference
```

#### Option 3: Interactive Inference

```bash
# After training, use the fine-tuned model interactively
python src/inference.py --model models/fine_tuned/final_model --interactive

# Or generate from a single prompt
python src/inference.py --model models/fine_tuned/final_model --prompt "Aaj ka din"
```

### Using Individual Modules

```python
# Example: Data Collection
from src.data_collection import DataCollector

collector = DataCollector('data/raw')
texts = collector.load_from_file('data/raw/sample_hindi_hinglish.json')
stats = collector.get_statistics()

# Example: Model Selection
from src.model_selection import ModelSelector

selector = ModelSelector('distilgpt2')
model, tokenizer = selector.load_model_and_tokenizer()

# Example: Inference
from src.inference import TextGenerator

generator = TextGenerator('models/fine_tuned/final_model')
text = generator.generate("Machine learning", max_length=100)
print(text[0])
```

## Configuration

Edit `config.py` to customize:

- **Model**: Change model type, paths
- **Data**: Adjust split ratios, preprocessing settings
- **Training**: Modify epochs, batch size, learning rate
- **Generation**: Tune temperature, top-k, top-p
- **Evaluation**: Set test prompts, metrics

## Expected Outcomes

✅ Successfully fine-tuned language model  
✅ Improved text generation quality for Hindi/Hinglish  
✅ Clear understanding of NLP fine-tuning workflow  
✅ Hands-on experience with transformer-based models  
✅ Comprehensive evaluation reports  
✅ Interactive inference capabilities  

## Technology Stack

- **Language**: Python 3.8+
- **ML Framework**: PyTorch
- **NLP Library**: HuggingFace Transformers
- **Data Processing**: Pandas, NumPy, scikit-learn
- **Training**: Accelerate, tqdm
- **Optional**: Weights & Biases, TensorBoard

## Sample Dataset

The project includes a sample Hindi/Hinglish dataset with 45+ sentences covering:
- Pure Hindi sentences (Devanagari script)
- Hinglish sentences (Roman script)
- Technology and AI-related terms
- General conversational phrases

## Performance Tips

1. **Memory Issues**: Reduce `batch_size` in config.py
2. **Faster Training**: Use GPU (CUDA/MPS) if available
3. **Better Quality**: Increase `epochs` and dataset size
4. **Smaller Model**: Use 'distilgpt2' instead of 'gpt2'
5. **8-bit Loading**: Set `load_in_8bit=True` for memory savings

## Troubleshooting

### Issue: Out of Memory
- Reduce batch_size to 2 or 1
- Reduce max_seq_length to 64
- Use distilgpt2 instead of larger models
- Enable 8-bit quantization

### Issue: Slow Training
- Ensure GPU is being used (check device output)
- Reduce max_seq_length
- Use smaller model

### Issue: Poor Generation Quality
- Increase training epochs
- Add more training data
- Adjust generation parameters (temperature, top_p)
- Try different prompts

## Next Steps

1. **Expand Dataset**: Add more Hindi/Hinglish text data
2. **Experiment**: Try different models and hyperparameters
3. **Evaluate**: Test on diverse prompts
4. **Deploy**: Create web interface using Gradio/Flask
5. **Optimize**: Implement model quantization for deployment

## License

MIT License

## Author

Aryaman Nain

## Acknowledgments

- HuggingFace Transformers library
- OpenAI GPT-2 model
- PyTorch framework

---

**Note**: This is a complete, production-ready implementation of a language model fine-tuning pipeline. All modules are fully functional and documented.
