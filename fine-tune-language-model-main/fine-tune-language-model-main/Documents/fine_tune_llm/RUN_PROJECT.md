# How to Run and Test This Project

## Quick Test (No Installation Required)

First, verify the project structure is complete:

```bash
cd ~/Documents/fine_tune_llm
python3 simple_test.py
```

This will check that all files and directories are in place.

## Full Installation and Testing

### Step 1: Install Dependencies

Install all required Python packages:

```bash
pip3 install -r requirements.txt
```

**Note**: This may take 5-10 minutes as it downloads PyTorch, Transformers, and other ML libraries.

### Step 2: Test Module Imports

After installation, test that all modules can be imported:

```bash
python3 test_project.py
```

This should show:
- ✅ data_collection imported successfully
- ✅ data_preprocessing imported successfully
- ✅ model_selection imported successfully
- ✅ fine_tuning imported successfully
- ✅ evaluation imported successfully
- ✅ inference imported successfully

### Step 3: Run the Main Program

Run the complete fine-tuning pipeline:

```bash
python3 main.py
```

This will:
1. Load the sample Hindi/Hinglish dataset
2. Preprocess the data
3. Load a pre-trained model (GPT-2 or DistilGPT-2)
4. Fine-tune the model on your dataset
5. Evaluate the results
6. Generate sample text

**Expected Runtime**: 10-30 minutes depending on your hardware (CPU/GPU)

## Troubleshooting

### If pip3 install fails:

Try installing packages individually:
```bash
pip3 install torch
pip3 install transformers
pip3 install datasets
pip3 install pandas numpy scikit-learn tqdm
```

### If you get "No module named 'transformers'":

Make sure you're using the correct Python version:
```bash
python3 --version
which python3
```

### If training is too slow:

Edit `config.py` and reduce:
- `num_train_epochs` (try 1 instead of 3)
- `max_length` (try 64 instead of 128)

## Project Status

✅ All 6 modules implemented (~3,100 lines of code)
✅ Sample dataset included
✅ Comprehensive documentation
✅ Ready to run and test

## What This Project Does

1. **Data Collection**: Loads regional language text (Hindi/Hinglish/Tamil/etc.)
2. **Preprocessing**: Cleans and tokenizes the text
3. **Model Selection**: Loads a small pre-trained language model
4. **Fine-Tuning**: Trains the model on your custom dataset
5. **Evaluation**: Compares model performance before/after training
6. **Inference**: Generates new text using the fine-tuned model

## Next Steps After Testing

1. Add your own dataset to the `data/` folder
2. Modify `config.py` to customize training parameters
3. Run the pipeline with your data
4. Push to GitHub (see PUSH_TO_GITHUB.md)
