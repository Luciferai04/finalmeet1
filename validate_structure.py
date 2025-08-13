#!/usr/bin/env python3
"""
Real-Time Translator - Structure Validation
==========================================

This script validates the project structure and import paths.
"""

import os
import sys
from pathlib import Path

def validate_structure():
    """Validate the project structure"""
    print("🔍 Validating Real-Time Translator project structure...")
    print("=" * 60)
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    
    errors = []
    successes = []
    
    # Required directories
    required_dirs = [
        "src",
        "src/api",
        "src/core", 
        "src/services",
        "src/ui",
        "src/utils",
        "config",
        "config/environments",
        "deploy",
        "deploy/docker",
        "data",
        "data/logs",
        "data/uploads",
        "docs",
        "tests"
    ]
    
    print("📁 Checking required directories...")
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            successes.append(f"✅ {dir_path}/")
        else:
            errors.append(f"❌ Missing directory: {dir_path}/")
    
    # Required files
    required_files = [
        "main.py",
        "wsgi.py", 
        "run_ui.py",
        "requirements.txt",
        "README.md",
        "src/__init__.py",
        "src/core/config.py",
        "src/api/flask_api_fixed.py",
        "config/environments/development.env",
        "config/environments/production.env",
        "docs/PROJECT_STRUCTURE.md"
    ]
    
    print("\n📄 Checking required files...")
    for file_path in required_files:
        if os.path.exists(file_path):
            successes.append(f"✅ {file_path}")
        else:
            errors.append(f"❌ Missing file: {file_path}")
    
    # Test imports
    print("\n🔗 Testing critical imports...")
    try:
        from src.core.config import Config
        successes.append("✅ Import src.core.config")
    except ImportError as e:
        errors.append(f"❌ Import error src.core.config: {e}")
    
    try:
        from src.api.flask_api_fixed import app
        successes.append("✅ Import src.api.flask_api_fixed")
    except ImportError as e:
        errors.append(f"❌ Import error src.api.flask_api_fixed: {e}")
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 VALIDATION RESULTS")
    print("=" * 60)
    
    if successes:
        print(f"\n✅ SUCCESSES ({len(successes)}):")
        for success in successes:
            print(f"   {success}")
    
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for error in errors:
            print(f"   {error}")
        print(f"\n❗ Fix the above {len(errors)} error(s) before proceeding.")
        return False
    else:
        print(f"\n🎉 All {len(successes)} checks passed! Structure is valid.")
        return True

def print_structure():
    """Print the current project structure"""
    print("\n📋 CURRENT PROJECT STRUCTURE:")
    print("=" * 60)
    
    def print_tree(directory, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
            
        items = []
        try:
            items = sorted(os.listdir(directory))
        except PermissionError:
            return
            
        # Filter out unwanted directories
        items = [item for item in items if not item.startswith('.') and 
                item not in ['__pycache__', 'flask_env', 'venv', 'node_modules']]
        
        for i, item in enumerate(items):
            if current_depth == 0 and len(items) > 20 and i >= 15:
                print(f"{prefix}└── ... ({len(items) - i} more items)")
                break
                
            item_path = os.path.join(directory, item)
            is_last = i == len(items) - 1
            
            if os.path.isdir(item_path):
                print(f"{prefix}{'└── ' if is_last else '├── '}{item}/")
                extension = "    " if is_last else "│   "
                print_tree(item_path, prefix + extension, max_depth, current_depth + 1)
            else:
                print(f"{prefix}{'└── ' if is_last else '├── '}{item}")
    
    print_tree(".")
    
if __name__ == "__main__":
    success = validate_structure()
    print_structure()
    
    if success:
        print("\n🚀 Ready for production deployment!")
        print("\nNext steps:")
        print("1. Set environment variables (GOOGLE_API_KEY, etc.)")
        print("2. Test with: python run_ui.py")
        print("3. Deploy with: docker-compose -f deploy/docker/docker-compose.prod.yml up")
    else:
        sys.exit(1)
