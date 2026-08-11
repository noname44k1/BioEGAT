import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def main():
    print("=== Hugging Face Spaces Deployer ===")
    
    # Try to read token from environment variable or prompt
    token = os.environ.get("HF_TOKEN")
    if not token:
        token = input("Please enter your Hugging Face write token (get one at https://huggingface.co/settings/tokens): ").strip()
        if not token:
            print("Error: Hugging Face token is required.")
            sys.exit(1)
            
    # Try to read repo_id
    repo_id = input("Enter repo ID (e.g. username/BioEGAT-Biomedical-Prediction): ").strip()
    if not repo_id:
        print("Error: Repo ID is required.")
        sys.exit(1)
        
    api = HfApi(token=token)
    
    print(f"Creating Space repository '{repo_id}' (if it does not exist)...")
    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            repo_type="space",
            space_sdk="gradio",
            private=False,
            exist_ok=True
        )
        print("Repository is ready.")
    except Exception as e:
        print(f"Error creating repository: {e}")
        sys.exit(1)
        
    print("Uploading hf_space folder contents to Hugging Face...")
    hf_space_dir = Path(__file__).resolve().parent

    # Add Gemini API key as a secret
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        try:
            from dotenv import load_dotenv
            load_dotenv(hf_space_dir / ".env")
            gemini_key = os.environ.get("GEMINI_API_KEY")
        except ImportError:
            pass
            
    if gemini_key:
        print("Setting GEMINI_API_KEY secret on Hugging Face Space...")
        try:
            api.add_space_secret(repo_id=repo_id, key="GEMINI_API_KEY", value=gemini_key)
        except Exception as e:
            print(f"Warning: Failed to set GEMINI_API_KEY secret: {e}")
    else:
        print("Warning: GEMINI_API_KEY not found in .env. You will need to set it manually in the Space settings.")

    try:
        api.upload_folder(
            folder_path=str(hf_space_dir),
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=["upload.py", ".env", "__pycache__/*", "*.pyc"]
        )
        print(f"\n🎉 Successfully uploaded! Your space is live at: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"Error uploading folder: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
