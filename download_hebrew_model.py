#!/usr/bin/env python3
"""
Hebrew Word2Vec Model Downloader

This script helps download the Hebrew Word2Vec model used by Semantle
for accurate semantic similarity calculations.
"""

import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def print_download_instructions():
    """Print detailed instructions for downloading the Hebrew Word2Vec model"""
    
    print("=" * 70)
    print("HEBREW WORD2VEC MODEL DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print()
    print("To use the SolveSemantle language model, you need the Hebrew Word2Vec model.")
    print("This is the EXACT same model used by Hebrew Semantle for maximum compatibility.")
    print()
    print("DOWNLOAD STEPS:")
    print("1. Go to: https://drive.google.com/drive/folders/1RDj6Gaa5t4jtd-VtsAqyZWyk6e7o2Xux")
    print("2. Download the Word2Vec model file")
    print("   - Look for files like: model.bin, hebrew_w2v.bin, word2vec.bin")
    print("   - Choose the binary Word2Vec format (.bin file)")
    print("3. Save it to one of these locations:")
    print(f"   - {os.path.abspath('model.bin')} (current directory)")
    print(f"   - {os.path.abspath('models/model.bin')} (models subdirectory)")
    print(f"   - {os.path.abspath('../model.bin')} (parent directory)")
    print()
    print("VERIFICATION:")
    print("After downloading, run:")
    print("  python src/language_model.py")
    print()
    print("This will test the model and confirm it works correctly.")
    print()
    print("MODEL DETAILS:")
    print("- Training Data: Hebrew Wikipedia corpus")
    print("- Tokenization: HebPipe (proper Hebrew tokenization)")
    print("- Format: Gensim KeyedVectors binary format")
    print("- Size: ~200-500MB (depending on vocabulary)")
    print("- Compatibility: Matches Hebrew Semantle exactly")
    print()
    print("TROUBLESHOOTING:")
    print("- If download fails, try different browsers")
    print("- Ensure file is not corrupted (size should be > 100MB)")
    print("- Check file extension is .bin for binary format")
    print("- Run 'pip install gensim numpy' if you get import errors")
    print()
    print("=" * 70)


def check_model_files():
    """Check if model files exist in common locations"""
    
    possible_paths = [
        "model.bin",
        "hebrew_w2v.bin", 
        "word2vec.bin",
        "models/model.bin",
        "models/hebrew_w2v.bin",
        "models/word2vec.bin",
        "../model.bin",
        "../hebrew_w2v.bin",
        "../word2vec.bin"
    ]
    
    found_files = []
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            file_size = os.path.getsize(abs_path) / (1024 * 1024)  # Size in MB
            found_files.append((abs_path, file_size))
    
    return found_files


def main():
    """Main model download helper"""
    
    print("Hebrew Word2Vec Model Setup")
    print("=" * 40)
    
    # Check for existing model files
    logger.info("Checking for existing model files...")
    existing_models = check_model_files()
    
    if existing_models:
        print(f"\n✅ Found {len(existing_models)} existing model file(s):")
        for path, size in existing_models:
            print(f"  - {path} ({size:.1f} MB)")
        print()
        print("You can test the model by running:")
        print("  python src/language_model.py")
        print()
        
        # Test if the largest file works
        if existing_models:
            largest_model = max(existing_models, key=lambda x: x[1])
            logger.info(f"Largest model file: {largest_model[0]} ({largest_model[1]:.1f} MB)")
            print(f"The solver will automatically use: {largest_model[0]}")
        
    else:
        print("\n❌ No model files found in common locations.")
        print("You need to download the Hebrew Word2Vec model.")
        print()
        print_download_instructions()
        
        # Create models directory if it doesn't exist
        models_dir = Path("models")
        if not models_dir.exists():
            models_dir.mkdir()
            logger.info("Created models/ directory for storing the model file")
    
    print("=" * 40)
    print("Setup complete!")


if __name__ == "__main__":
    main()
