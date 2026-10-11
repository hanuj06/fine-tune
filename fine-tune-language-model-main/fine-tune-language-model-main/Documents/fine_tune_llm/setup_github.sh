#!/bin/bash
# Script to initialize git and push to GitHub

echo "Setting up Git repository..."
echo "======================================"

cd ~/Documents/fine_tune_llm

# Initialize git if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    echo "✅ Git initialized"
else
    echo "✅ Git already initialized"
fi

# Add all files
echo "\nAdding files to git..."
git add .
echo "✅ Files added"

# Create initial commit
echo "\nCreating commit..."
git commit -m "Initial commit: Complete fine-tuning language model project

- Implemented 6 core modules (data collection, preprocessing, model selection, fine-tuning, evaluation, inference)
- Added comprehensive documentation (5 guides)
- Included sample Hindi/Hinglish dataset
- Complete configuration system
- Production-ready code (~3,100 lines)
- All objectives achieved"

echo "✅ Commit created"

echo "\n======================================"
echo "Git repository ready!"
echo "\nNext steps:"
echo "1. Create a new repository on GitHub"
echo "2. Run these commands:"
echo "   git remote add origin <your-github-repo-url>"
echo "   git branch -M main"
echo "   git push -u origin main"
echo "\nOr use GitHub CLI:"
echo "   gh repo create fine-tune-llm --public --source=. --remote=origin"
echo "   git push -u origin main"
echo "======================================"
