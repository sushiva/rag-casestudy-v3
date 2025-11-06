cat > scripts/deploy_to_hf.py << 'EOF'
#!/usr/bin/env python3
"""
Reusable HuggingFace Spaces deployment script
"""

import os
import subprocess
import shutil
import argparse
from pathlib import Path

def deploy_to_hf(
    source_dir: str,
    hf_username: str,
    hf_space_name: str,
    exclude_dirs: list = None
):
    """
    Deploy project to HuggingFace Spaces
    
    Args:
        source_dir: Local project directory
        hf_username: HuggingFace username
        hf_space_name: HuggingFace Space name
        exclude_dirs: Directories to exclude (default: data, .git, __pycache__)
    """
    
    if exclude_dirs is None:
        exclude_dirs = ["data/chroma_db", ".git", "__pycache__", ".DS_Store", "venv"]
    
    hf_url = f"https://huggingface.co/spaces/{hf_username}/{hf_space_name}"
    temp_dir = f"/tmp/hf_{hf_space_name}"
    
    print(f"🚀 Deploying to HuggingFace Spaces")
    print(f"📍 Source: {source_dir}")
    print(f"🎯 Target: {hf_url}")
    print(f"📦 Temp dir: {temp_dir}")
    
    try:
        # STEP 1: Clone HF Space
        print("\n1️⃣ Cloning HuggingFace Space...")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        subprocess.run([
            "git", "clone", 
            f"https://huggingface.co/spaces/{hf_username}/{hf_space_name}",
            temp_dir
        ], check=True, capture_output=True)
        print("✅ Cloned")
        
        # STEP 2: Copy files
        print("\n2️⃣ Copying files...")
        for item in Path(source_dir).iterdir():
            if item.name in exclude_dirs:
                print(f"  ⏭️ Skipping: {item.name}")
                continue
            
            dest = Path(temp_dir) / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
                print(f"  📁 Copied: {item.name}/")
            else:
                shutil.copy2(item, dest)
                print(f"  📄 Copied: {item.name}")
        
        # STEP 3: Commit and push
        print("\n3️⃣ Committing and pushing...")
        os.chdir(temp_dir)
        
        subprocess.run(["git", "add", "."], check=True, capture_output=True)
        subprocess.run([
            "git", "commit", 
            "-m", "Auto-deploy from local repository"
        ], check=True, capture_output=True)
        subprocess.run(["git", "push"], check=True, capture_output=True)
        print("✅ Pushed to HuggingFace")
        
        print(f"\n✅ Deployment complete!")
        print(f"🌐 View at: {hf_url}")
        print(f"⏳ Space will build in 2-5 minutes...")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"📝 Make sure you're authenticated with: huggingface-cli login")
        exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        exit(1)
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            os.chdir("/")
            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up temp directory")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy to HuggingFace Spaces")
    parser.add_argument("--source", default=".", help="Source directory (default: current)")
    parser.add_argument("--username", required=True, help="HuggingFace username")
    parser.add_argument("--space", required=True, help="HuggingFace Space name")
    parser.add_argument("--exclude", nargs="+", help="Directories to exclude")
    
    args = parser.parse_args()
    
    deploy_to_hf(
        source_dir=args.source,
        hf_username=args.username,
        hf_space_name=args.space,
        exclude_dirs=args.exclude
    )
EOF