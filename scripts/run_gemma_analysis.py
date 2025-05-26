#!/usr/bin/env python3
"""
Runner script for Gemma embeddings analysis with dependency checking
"""

import subprocess
import sys
import os

def check_and_install_dependencies():
    """Check and install required dependencies"""
    
    required_packages = {
        'transformers': 'transformers',
        'torch': 'torch',
        'econml': 'econml',
        'xgboost': 'xgboost',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'scikit-learn'
    }
    
    missing = []
    
    for import_name, pip_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    
    if missing:
        print(f"Missing packages: {missing}")
        print("Installing missing packages...")
        
        for package in missing:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            except subprocess.CalledProcessError as e:
                print(f"Error installing {package}: {e}")
                return False
    
    print("All dependencies satisfied!")
    return True

def main():
    """Run the Gemma analysis"""
    
    # Check dependencies
    if not check_and_install_dependencies():
        print("Failed to install dependencies. Please install manually.")
        return 1
    
    # Run the analysis
    script_path = os.path.join(os.path.dirname(__file__), 'gemma_embedding_analysis.py')
    
    try:
        subprocess.check_call([sys.executable, script_path])
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Analysis failed: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main())