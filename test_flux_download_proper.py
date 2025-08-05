#!/usr/bin/env python3
"""
Test script to download FLUX properly and then train
"""

import os
import json
import base64
import requests
import time

def test_flux_download_and_train():
    """Test downloading FLUX to proper location and training"""
    
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
    
    print("🚀 FLUX Download & Training Test")
    print(f"📡 Endpoint: {endpoint}")
    print(f"🤗 HF Token: {hf_token[:10]}...")
    print(f"💾 Target: /runpod-volume/flux-models/")
    
    # Check if test.zip exists
    if not os.path.exists("test.zip"):
        print("❌ test.zip not found")
        return
    
    with open("test.zip", 'rb') as f:
        dataset_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    print(f"📦 Dataset encoded: {len(dataset_b64)} chars")
    
    # Create payload that forces proper download
    payload = {
        "input": {
            "model_name": "test_flux_proper_download",
            "dataset": dataset_b64,
            "trigger_word": "testsubject",
            
            # Use HuggingFace model ID with download flag
            "base_model": "black-forest-labs/FLUX.1-dev",
            "download_to_network_storage": True,
            "model_cache_dir": "/runpod-volume/flux-models",
            
            "steps": 200,  # Shorter for testing
            "learning_rate": 0.0003,
            "network_dim": 16,
            "network_alpha": 16,
            "resolution": "512x512",  # Smaller resolution for faster processing
            "fp8_base": True,
            "gradient_checkpointing": True,
            "cache_latents": True,
            "sample_every_n_steps": 50,
            "save_every_n_steps": 100,
            "huggingface_token": hf_token,
            
            # Additional settings to ensure proper download
            "mixed_precision": "bf16",
            "seed": 42
        }
    }
    
    # Send request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    print("📤 Sending request to download and train...")
    print("⏳ This will download FLUX.1-dev (~24GB) to proper location first...")
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=10800)  # 3 hour timeout
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Request successful!")
            print(f"Status: {result.get('status', 'unknown')}")
            
            if result.get("id"):
                print(f"Job ID: {result['id']}")
                print("📥 Expected behavior:")
                print("  1. Download FLUX.1-dev to /runpod-volume/flux-models/")
                print("  2. Detect proper model structure")
                print("  3. Start training with downloaded model")
                print("  4. Generate samples every 50 steps")
                print("  5. Save final LoRA model")
            
            return result
        else:
            print(f"❌ Failed: {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    print("📥 FLUX Proper Download & Training Test")
    print("=" * 45)
    
    test_flux_download_and_train()