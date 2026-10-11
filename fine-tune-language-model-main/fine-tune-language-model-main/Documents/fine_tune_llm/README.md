# Fine-Tuning a Small Language Model on Regional/Custom Language Dataset

This project demonstrates how to fine-tune a small pre-trained language model on a custom or regional language dataset (Hindi, Punjabi, Tamil, Hinglish, or domain-specific text).

## Project Structure

```
fine_tune_llm/
├── data/                    # Dataset storage
│   ├── raw/                 # Raw collected data
│   └── processed/           # Preprocessed data
├── models/                  # Saved models
│   ├── base/                # Base pre-trained model
│   └── fine_tuned/          # Fine-tuned model checkpoints
├── outputs/                 # Training outputs and logs
├── src/                     # Source code modules
│   ├── data_collection.py   # Dataset collection module
│   ├── data_preprocessing.py # Data preprocessing module
│   ├── model_selection.py   # Model selection module
│   ├── fine_tuning.py       # Fine-tuning module
│   ├── evaluation.py        # Evaluation module
│   └── inference.py         # Inference module
├── config.py                # Configuration settings
├── main.py                  # Main application entry point
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Features

### 1. Dataset Collection Module
- Collect regional or domain-specific text
- Support for .txt, .csv, .json formats
- Data cleaning and normalization

### 2. Data Preprocessing Module
- Tokenization using HuggingFace tokenizers
- Noise removal and text cleaning
- Train-validation split

### 3. Model Selection Module
- Load lightweight pre-trained models (GPT-2 small, DistilGPT-2)
- Configure tokenizer and model architecture

### 4. Fine-Tuning Module
- Configure training hyperparameters
- Train model on custom dataset
- Monitor training loss and metrics

### 5. Evaluation Module
- Compare outputs before and after fine-tuning
- Generate sample prompts
- Analyze qualitative improvements

### 6. Inference & Demonstration Module
- Generate text using fine-tuned model
- CLI interface for interaction
- Save and reload trained models

## Installation

```bash
# Clone the repository
cd fine_tune_llm

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
# Run the complete pipeline
python main.py

# Or run individual modules
python main.py --mode collect      # Collect data
python main.py --mode preprocess   # Preprocess data
python main.py --mode train        # Fine-tune model
python main.py --mode evaluate     # Evaluate model
python main.py --mode inference    # Run inference
```

### Configuration

Edit `config.py` to customize:
- Model selection (default: distilgpt2)
- Training parameters (epochs, batch size, learning rate)
- Dataset paths and formats
- Output directories

## Supported Languages/Domains

- Hindi (हिंदी)
- Punjabi (ਪੰਜਾਬੀ)
- Tamil (தமிழ்)
- Hinglish (Hindi + English)
- Custom domain-specific text

## Technology Stack

- **Programming Language**: Python 3.8+
- **ML Framework**: PyTorch
- **NLP Library**: HuggingFace Transformers
- **Training**: CPU/GPU support via Accelerate

## Expected Outcomes

✔ Successfully fine-tuned language model  
✔ Improved text generation quality for target language/domain  
✔ Clear understanding of NLP fine-tuning workflow  
✔ Hands-on experience with transformer-based models  

## License

MIT License

## Author

Aryaman Nain
