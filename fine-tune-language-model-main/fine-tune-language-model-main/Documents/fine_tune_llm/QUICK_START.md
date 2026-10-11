# Quick Start Guide - Fine-Tuning Language Model

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

```bash
cd ~/Documents/fine_tune_llm

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Step 2: Run the Complete Pipeline

```bash
# Run full pipeline with sample Hindi/Hinglish data
python main.py --mode full
```

This will:
- ✅ Load sample Hindi/Hinglish dataset (45+ sentences)
- ✅ Preprocess and split data (train/val/test)
- ✅ Download DistilGPT-2 model (82M parameters)
- ✅ Fine-tune on your data (3 epochs)
- ✅ Evaluate before/after performance
- ✅ Generate sample outputs

**Expected Time**: 10-30 minutes (depending on hardware)

### Step 3: Try Interactive Generation

```bash
# Start interactive mode
python src/inference.py --model models/fine_tuned/final_model --interactive
```

Then type prompts like:
- `Aaj ka din`
- `Machine learning`
- `भारत एक`
- `Technology se`

---

## 🛠️ Advanced Usage

### Run Individual Phases

```bash
# Just collect/preprocess data
python main.py --mode collect
python main.py --mode preprocess

# Just run inference (if model already trained)
python main.py --mode inference
```

### Generate Text from Command Line

```bash
# Single prompt
python src/inference.py --model models/fine_tuned/final_model --prompt "Aaj ka din"

# Multiple generations
python src/inference.py --model models/fine_tuned/final_model --prompt "Machine learning" --num-sequences 3

# Adjust creativity
python src/inference.py --model models/fine_tuned/final_model --prompt "भारत" --temperature 0.9 --max-length 150
```

### Use Python API

```python
# Load and use the fine-tuned model
from src.inference import TextGenerator

generator = TextGenerator('models/fine_tuned/final_model')
texts = generator.generate(
    prompt="Aaj ka din",
    max_length=100,
    temperature=0.7
)
print(texts[0])
```

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Change model
MODEL_CONFIG['model_name'] = 'gpt2'  # or 'distilgpt2', 'gpt2-medium'

# Adjust training
TRAINING_CONFIG['epochs'] = 5
TRAINING_CONFIG['batch_size'] = 8
TRAINING_CONFIG['learning_rate'] = 3e-5

# Tune generation
GENERATION_CONFIG['temperature'] = 0.8
GENERATION_CONFIG['max_length'] = 150
```

---

## 📊 Check Results

After training, check:

1. **Evaluation Report**: `outputs/evaluation_report.txt`
2. **Comparison Results**: `outputs/comparison_results.json`
3. **Training History**: `models/fine_tuned/training_history.json`
4. **Fine-tuned Model**: `models/fine_tuned/final_model/`

---

## 🐛 Troubleshooting

### Out of Memory?
```python
# In config.py, reduce:
TRAINING_CONFIG['batch_size'] = 2
TRAINING_CONFIG['max_seq_length'] = 64
```

### Slow Training?
- Check if GPU is detected (look for "Device: cuda" or "Device: mps")
- Use smaller model: `MODEL_CONFIG['model_name'] = 'distilgpt2'`

### Poor Quality?
- Add more training data to `data/raw/`
- Increase epochs: `TRAINING_CONFIG['epochs'] = 5`
- Adjust generation temperature

---

## 📚 Learn More

- **Full Documentation**: See `README.md`
- **Project Structure**: See `PROJECT_SUMMARY.md`
- **Module Details**: Check docstrings in `src/*.py`

---

## ✨ Example Outputs

**Before Fine-tuning** (Base GPT-2):
```
Prompt: "Aaj ka din"
Output: "Aaj ka din in the United States, and the United States..."
```

**After Fine-tuning** (on Hindi/Hinglish):
```
Prompt: "Aaj ka din"
Output: "Aaj ka din bahut accha hai. Mausam bhi bahut sundar hai..."
```

---

**Happy Fine-tuning! 🎉**
