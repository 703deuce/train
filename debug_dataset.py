#!/usr/bin/env python3

import os
import requests
import json

def debug_dataset_structure():
    """Debug the dataset structure on RunPod"""
    
    # RunPod credentials
    RUNPOD_ENDPOINT = "https://api.runpod.ai/v2/9r23v15ekm7ci4/run"
    RUNPOD_API_KEY = "rpa_C55TBQG7H6FM7G3Q7A6JM7ZJCDKA3I2J3EO0TAH8fxyddo"
    
    # Create a debug payload
    debug_payload = {
        "input": {
            "debug_dataset": True,
            "dataset_path": "/runpod-volume/datasets/test_flux_dreambooth_1000"
        }
    }
    
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    print("🔍 Debugging dataset structure...")
    print(f"📡 Sending debug request to: {RUNPOD_ENDPOINT}")
    
    try:
        response = requests.post(RUNPOD_ENDPOINT, headers=headers, json=debug_payload)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Debug request sent successfully!")
            print(f"📋 Job ID: {result.get('id', 'Unknown')}")
            print(f"📊 Status: {result.get('status', 'Unknown')}")
            
            if 'output' in result:
                print("\n📁 Dataset Structure:")
                print(result['output'])
            else:
                print("\n⚠️ No output received yet")
                
        else:
            print(f"❌ Failed to send debug request: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error sending debug request: {e}")

if __name__ == "__main__":
    debug_dataset_structure() 