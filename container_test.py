#!/usr/bin/env python3
"""
Container Test Script for RunPod
Uses environment variables set in the container
"""

import os
import json
import base64
import requests
import time
from pathlib import Path

def test_training():
    """Test training using environment variables from container"""
    
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
    
    print("🚀 Starting RunPod Training Test")
    print(f"📡 Endpoint: {endpoint}")
    print(f"🔑 API Key: {api_key[:10]}...")
    print(f"🤗 HF Token: {hf_token[:10]}...")
    
    # Encode test.zip
    if not os.path.exists("test.zip"):
        print("❌ test.zip not found")
        return
    
    with open("test.zip", 'rb') as f:
        dataset_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    print(f"📦 Dataset encoded: {len(dataset_b64)} chars")
    
    # Create payload
    payload = {
        "input": {
            "model_name": "test_flux_lora_1000",
            "dataset": dataset_b64,
            "trigger_word": "testsubject",
            "base_model": "flux1-dev",
            "steps": 1000,
            "learning_rate": 0.0003,
            "network_dim": 16,
            "network_alpha": 16,
            "resolution": "1024x1024",
            "fp8_base": True,
            "gradient_checkpointing": True,
            "cache_latents": True,
            "sample_every_n_steps": 100,
            "save_every_n_steps": 200,
            "huggingface_token": hf_token
        }
    }
    
    # Send request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    print("📤 Sending training request...")
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=7200)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Training completed successfully!")
            print(json.dumps(result, indent=2)[:500] + "...")
            return result
        else:
            print(f"❌ Failed: {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    test_training()