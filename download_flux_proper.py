#!/usr/bin/env python3
"""
Download FLUX.1-dev model properly for ai-toolkit
Creates a new folder with correct diffusers structure
"""

import os
import json
import base64
import requests

def download_flux_model():
    """Download FLUX.1-dev model to proper location for ai-toolkit"""
    
    # Get configuration from environment
    endpoint = os.getenv("RUNPOD_ENDPOINT")
    api_key = os.getenv("RUNPOD_API_KEY") 
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    
    if not all([endpoint, api_key, hf_token]):
        print("❌ Missing environment variables:")
        if not endpoint: print("  - RUNPOD_ENDPOINT")
        if not api_key: print("  - RUNPOD_API_KEY")
        if not hf_token: print("  - HUGGINGFACE_TOKEN")
        return
    
    print("🚀 Downloading FLUX.1-dev with Proper Structure")
    print(f"📡 Endpoint: {endpoint}")
    print(f"🤗 HF Token: {hf_token[:10]}...")
    
    # Check if test.zip exists
    if not os.path.exists("test.zip"):
        print("❌ test.zip not found")
        return
    
    with open("test.zip", 'rb') as f:
        dataset_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    print(f"📦 Dataset encoded: {len(dataset_b64)} chars")
    
    # Create payload that will trigger model download to proper location
    payload = {
        "input": {
            "model_name": "test_flux_download_1000",
            "dataset": dataset_b64,
            "trigger_word": "testsubject",
            
            # Use the HuggingFace model ID - this will trigger proper download
            "base_model": "black-forest-labs/FLUX.1-dev",
            
            # Add parameters to ensure proper model caching
            "download_to_network_storage": True,
            "model_cache_dir": "/runpod-volume/flux-models",
            
            "steps": 100,  # Shorter test run
            "learning_rate": 0.0003,
            "network_dim": 16,
            "network_alpha": 16,
            "resolution": "512x512",  # Smaller for faster testing
            "fp8_base": True,
            "gradient_checkpointing": True,
            "cache_latents": True,
            "sample_every_n_steps": 50,
            "save_every_n_steps": 50,
            "huggingface_token": hf_token
        }
    }
    
    # Send request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    print("📤 Sending request to download FLUX.1-dev properly...")
    print("⏳ This will download ~24GB and may take several minutes...")
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=10800)  # 3 hour timeout
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Download/Training request successful!")
            print(f"Status: {result.get('status', 'unknown')}")
            
            if result.get("id"):
                print(f"Job ID: {result['id']}")
                print("📥 FLUX.1-dev should now be downloading to /runpod-volume/flux-models/")
                print("📊 Monitor the logs for download progress")
            
            return result
        else:
            print(f"❌ Failed: {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    print("📥 FLUX.1-dev Proper Download Script")
    print("=" * 40)
    
    download_flux_model()