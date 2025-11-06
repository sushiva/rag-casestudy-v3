#!/usr/bin/env python3
"""
Setup deployment folder structure
Run from project root: python scripts/setup_deployment.py
"""

import os
from pathlib import Path

def setup_deployment_structure():
    """Create deployment folder structure"""
    
    print("📁 Setting up deployment structure...")
    
    # Define folder structure
    folders = [
        "scripts/deployment",
        "scripts/deployment/huggingface",
        "scripts/deployment/aws",
        "scripts/deployment/docker",
    ]
    
    # Create folders
    for folder in folders:
        try:
            os.makedirs(folder, exist_ok=True)  # Python equivalent of mkdir -p
            print(f"✅ Created: {folder}")
        except Exception as e:
            print(f"❌ Error creating {folder}: {e}")
    
    # Create __init__.py files
    init_files = [
        "scripts/deployment/__init__.py",
        "scripts/deployment/huggingface/__init__.py",
    ]
    
    for init_file in init_files:
        try:
            Path(init_file).touch()
            print(f"✅ Created: {init_file}")
        except Exception as e:
            print(f"❌ Error creating {init_file}: {e}")
    
    print("\n✅ Deployment structure created!")
    print("\nFolder structure:")
    print("""
scripts/deployment/
├── __init__.py
├── README.md
├── huggingface/
│   ├── __init__.py
│   ├── deploy.py
│   └── README.md
├── aws/
│   └── (coming soon)
└── docker/
    └── (coming soon)
    """)

if __name__ == "__main__":
    setup_deployment_structure()