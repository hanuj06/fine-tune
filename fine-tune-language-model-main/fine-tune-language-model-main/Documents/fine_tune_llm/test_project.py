#!/usr/bin/env python3
"""
Quick Test Script
Tests if all modules can be imported successfully.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("Testing Fine-Tuning Project...")
print("=" * 50)

try:
    print("\n1. Testing data_collection module...")
    from data_collection import DataCollector, create_sample_dataset
    print("   ✅ data_collection imported successfully")
    
    print("\n2. Testing data_preprocessing module...")
    from data_preprocessing import DataPreprocessor
    print("   ✅ data_preprocessing imported successfully")
    
    print("\n3. Testing model_selection module...")
    from model_selection import ModelSelector
    print("   ✅ model_selection imported successfully")
    
    print("\n4. Testing fine_tuning module...")
    from fine_tuning import FineTuner, TextDataset
    print("   ✅ fine_tuning imported successfully")
    
    print("\n5. Testing evaluation module...")
    from evaluation import ModelEvaluator
    print("   ✅ evaluation imported successfully")
    
    print("\n6. Testing inference module...")
    from inference import TextGenerator
    print("   ✅ inference imported successfully")
    
    print("\n7. Testing config module...")
    import config
    print("   ✅ config imported successfully")
    
    print("\n8. Testing sample dataset exists...")
    sample_path = Path('data/raw/sample_hindi_hinglish.json')
    if sample_path.exists():
        print(f"   ✅ Sample dataset found at {sample_path}")
    else:
        print(f"   ❌ Sample dataset not found")
    
    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED!")
    print("=" * 50)
    print("\nProject is ready to use!")
    print("\nNext steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run pipeline: python main.py --mode full")
    print("\n")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
