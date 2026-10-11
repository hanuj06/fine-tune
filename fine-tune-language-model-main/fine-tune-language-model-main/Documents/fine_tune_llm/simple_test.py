#!/usr/bin/env python3
"""
Simple test to verify project structure without requiring dependencies
"""
import os
import sys

def test_project_structure():
    """Test if all required files and directories exist"""
    print("=" * 60)
    print("TESTING PROJECT STRUCTURE")
    print("=" * 60)
    
    # Required directories
    required_dirs = ['src', 'data', 'models', 'outputs']
    
    # Required files
    required_files = [
        'requirements.txt',
        'README.md',
        'config.py',
        'main.py',
        'src/data_collection.py',
        'src/data_preprocessing.py',
        'src/model_selection.py',
        'src/fine_tuning.py',
        'src/evaluation.py',
        'src/inference.py'
    ]
    
    all_passed = True
    
    # Check directories
    print("\n1. Checking directories...")
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"   ✅ {dir_name}/ exists")
        else:
            print(f"   ❌ {dir_name}/ NOT FOUND")
            all_passed = False
    
    # Check files
    print("\n2. Checking files...")
    for file_path in required_files:
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"   ✅ {file_path} ({size} bytes)")
        else:
            print(f"   ❌ {file_path} NOT FOUND")
            all_passed = False
    
    # Check if sample data exists
    print("\n3. Checking sample data...")
    data_files = os.listdir('data')
    if data_files:
        print(f"   ✅ Found {len(data_files)} file(s) in data/")
        for f in data_files:
            print(f"      - {f}")
    else:
        print("   ⚠️  No data files found (you'll need to add your dataset)")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ PROJECT STRUCTURE TEST PASSED!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Install dependencies: pip3 install -r requirements.txt")
        print("2. Run the full test: python3 test_project.py")
        print("3. Run the main program: python3 main.py")
        return 0
    else:
        print("❌ PROJECT STRUCTURE TEST FAILED!")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(test_project_structure())
