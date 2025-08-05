#!/usr/bin/env python3
"""
Test script specifically for cached FLUX weights in /runpod-volume/cache
"""

import os
import json
import base64
import requests
import time

def test_with_direct_cache():
    """Test training using direct cache path"""
    
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
    
    print("🚀 Testing with Direct Cache Path")
    print(f"📡 Endpoint: {endpoint}")
    print(f"💾 Cache Path: /runpod-volume/cache")
    
    # Encode test.zip
    if not os.path.exists("test.zip"):
        print("❌ test.zip not found")
        return
    
    with open("test.zip", 'rb') as f:
        dataset_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    print(f"📦 Dataset encoded: {len(dataset_b64)} chars")
    
    # Create payload with direct cache path
    payload = {
        "input": {
            "model_name": "test_flux_cache_1000",
            "dataset": dataset_b64,
            "trigger_word": "testsubject",
            
            # Use direct cache path - this should work now
            "base_model": "/runpod-volume/cache",
            
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
    
    print("📤 Sending training request with direct cache path...")
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=7200)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Training request successful!")
            print(f"Status: {result.get('status', 'unknown')}")
            
            if result.get("id"):
                print(f"Job ID: {result['id']}")
            
            return result
        else:
            print(f"❌ Failed: {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    # Set environment variables for local testing
    if not os.getenv("RUNPOD_ENDPOINT"):
        os.environ["RUNPOD_ENDPOINT"] = "https://api.runpod.ai/v2/9r23v15ekm7ci4/run"
    if not os.getenv("RUNPOD_API_KEY"):
        os.environ["RUNPOD_API_KEY"] = input("Enter RunPod API Key: ").strip()
    if not os.getenv("HUGGINGFACE_TOKEN"):
        os.environ["HUGGINGFACE_TOKEN"] = input("Enter HuggingFace Token: ").strip()
    
    test_with_direct_cache()