"""
Fine-Tuning Module
==================
This module handles the fine-tuning process for language models,
including training loop, loss monitoring, and checkpoint management.
"""

import os
import json
import torch
import time
from typing import Optional, Dict, List
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup
)


class TextDataset(Dataset):
    """
    Custom Dataset for text data.
    """
    
    def __init__(self, texts: List[str], tokenizer: PreTrainedTokenizer, max_length: int = 128):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text entries
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # For causal language modeling, labels are the same as input_ids
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }


class FineTuner:
    """
    A class to handle fine-tuning of language models.
    """
    
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 output_dir: str = "models/fine_tuned",
                 device: Optional[str] = None):
        """
        Initialize the FineTuner.
        
        Args:
            model: Pre-trained model to fine-tune
            tokenizer: Tokenizer for the model
            output_dir: Directory to save fine-tuned models
            device: Device to use (cuda/mps/cpu)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        self.model = self.model.to(self.device)
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'epochs': []
        }
        
        print(f"FineTuner initialized on device: {self.device}")
    
    def create_dataloader(self,
                         texts: List[str],
                         batch_size: int = 8,
                         max_length: int = 128,
                         shuffle: bool = True) -> DataLoader:
        """
        Create a DataLoader from text data.
        
        Args:
            texts: List of text entries
            batch_size: Batch size for training
            max_length: Maximum sequence length
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader object
        """
        dataset = TextDataset(texts, self.tokenizer, max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        return dataloader
    
    def train_epoch(self,
                   train_loader: DataLoader,
                   optimizer: torch.optim.Optimizer,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Evaluate the model on validation data.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Evaluating")
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(self,
             train_texts: List[str],
             val_texts: Optional[List[str]] = None,
             epochs: int = 3,
             batch_size: int = 8,
             learning_rate: float = 5e-5,
             max_length: int = 128,
             warmup_steps: int = 100,
             save_steps: int = 500,
             eval_steps: int = 500) -> Dict:
        """
        Fine-tune the model.
        
        Args:
            train_texts: Training texts
            val_texts: Validation texts (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            max_length: Maximum sequence length
            warmup_steps: Number of warmup steps
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            
        Returns:
            Training history dictionary
        """
        print("=" * 70)
        print("Starting Fine-Tuning")
        print("=" * 70)
        print(f"Training samples: {len(train_texts)}")
        if val_texts:
            print(f"Validation samples: {len(val_texts)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Max length: {max_length}")
        print(f"Device: {self.device}")
        print("=" * 70)
        
        # Create data loaders
        train_loader = self.create_dataloader(
            train_texts, batch_size, max_length, shuffle=True
        )
        
        val_loader = None
        if val_texts:
            val_loader = self.create_dataloader(
                val_texts, batch_size, max_length, shuffle=False
            )
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 70)
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['epochs'].append(epoch + 1)
            
            print(f"Train Loss: {train_loss:.4f}")
            
            # Evaluate
            if val_loader:
                val_loss = self.evaluate(val_loader)
                self.training_history['val_loss'].append(val_loss)
                print(f"Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(f"best_model_epoch_{epoch+1}")
                    print(f"Saved best model (val_loss: {val_loss:.4f})")
            
            # Save checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch+1}")
        
        # Training complete
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("Fine-Tuning Complete!")
        print(f"Total time: {elapsed_time/60:.2f} minutes")
        print("=" * 70)
        
        # Save final model
        self.save_model("final_model")
        self.save_training_history()
        
        return self.training_history
    
    def save_checkpoint(self, checkpoint_name: str) -> None:
        """
        Save a training checkpoint.
        
        Args:
            checkpoint_name: Name for the checkpoint
        """
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
    
    def save_model(self, model_name: str = "final_model") -> str:
        """
        Save the fine-tuned model.
        
        Args:
            model_name: Name for the saved model
            
        Returns:
            Path to saved model
        """
        model_path = self.output_dir / model_name
        model_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving model to {model_path}...")
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        print("Model saved successfully!")
        
        return str(model_path)
    
    def save_training_history(self, filename: str = "training_history.json") -> None:
        """
        Save training history to a JSON file.
        
        Args:
            filename: Output filename
        """
        history_path = self.output_dir / filename
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"Training history saved to {history_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading checkpoint from {checkpoint_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = self.model.to(self.device)
        print("Checkpoint loaded successfully!")


if __name__ == "__main__":
    # Demo usage
    print("=" * 70)
    print("Fine-Tuning Module Demo")
    print("=" * 70)
    
    print("\nThis module requires:")
    print("  1. A pre-trained model (from model_selection.py)")
    print("  2. Preprocessed training data (from data_preprocessing.py)")
    print("\nExample usage:")
    print("""
    from model_selection import ModelSelector
    from fine_tuning import FineTuner
    import json
    
    # Load model
    selector = ModelSelector('distilgpt2')
    model, tokenizer = selector.load_model_and_tokenizer()
    
    # Load training data
    with open('data/processed/data_train.json', 'r') as f:
        train_data = json.load(f)['texts']
    
    with open('data/processed/data_val.json', 'r') as f:
        val_data = json.load(f)['texts']
    
    # Fine-tune
    tuner = FineTuner(model, tokenizer)
    history = tuner.train(
        train_texts=train_data,
        val_texts=val_data,
        epochs=3,
        batch_size=4,
        learning_rate=5e-5
    )
    """)
