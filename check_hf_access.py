#!/usr/bin/env python3
"""
Check HuggingFace access to FLUX.1-dev
"""

import os
import json
import base64
import requests

def check_hf_access():
    """Check if we can access FLUX.1-dev"""
    
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or input("Enter HF Token: ").strip()
    
    print("🔍 Checking HuggingFace Access to FLUX.1-dev")
    print(f"🔑 Token: {hf_token[:10]}...")
    
    # Test HuggingFace API access
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # Check if we can access the model info
    try:
        print("\n1. Testing model info access...")
        response = requests.get(
            "https://huggingface.co/api/models/black-forest-labs/FLUX.1-dev",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            model_info = response.json()
            print("✅ Model info accessible")
            print(f"   Gated: {model_info.get('gated', 'Unknown')}")
            print(f"   Private: {model_info.get('private', 'Unknown')}")
        else:
            print(f"❌ Model info failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"❌ Model info error: {e}")
    
    # Test file listing
    try:
        print("\n2. Testing file listing...")
        response = requests.get(
            "https://huggingface.co/api/models/black-forest-labs/FLUX.1-dev/tree/main",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            files = response.json()
            print("✅ File listing accessible")
            print(f"   Found {len(files)} files/folders")
            for item in files[:5]:  # Show first 5 items
                print(f"   - {item.get('path', 'unknown')}")
        else:
            print(f"❌ File listing failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"❌ File listing error: {e}")
    
    # Test specific file access (model_index.json)
    try:
        print("\n3. Testing model_index.json access...")
        response = requests.get(
            "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/model_index.json",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ model_index.json accessible")
            model_index = response.json()
            print(f"   Components: {list(model_index.keys())}")
        else:
            print(f"❌ model_index.json failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"❌ model_index.json error: {e}")
    
    print("\n📋 Summary:")
    print("If you see ❌ errors above, you need to:")
    print("1. Go to https://huggingface.co/black-forest-labs/FLUX.1-dev")
    print("2. Log in and click 'Agree' to accept the license")
    print("3. Make sure your token has 'read' permissions")
    print("4. Retry the training")

if __name__ == "__main__":
    check_hf_access()