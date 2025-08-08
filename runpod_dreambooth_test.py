#!/usr/bin/env python3
"""
RunPod DreamBooth Training Test Script
Tests the DreamBooth fine-tuning API for FLUX models
"""

import os
import requests
import json
import base64
import time
from typing import Optional

def encode_zip_file(file_path: str) -> str:
    """Encode a zip file to base64"""
    try:
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None

def get_api_credentials() -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Get API credentials from environment or user input"""
    
    # Try to get from environment variables first
    RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT")
    RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    
    if not RUNPOD_ENDPOINT:
        RUNPOD_ENDPOINT = input("Enter your RunPod Endpoint URL: ").strip()
    
    if not RUNPOD_API_KEY:
        RUNPOD_API_KEY = input("Enter your RunPod API Key: ").strip()
    
    if not HUGGINGFACE_TOKEN:
        HUGGINGFACE_TOKEN = input("Enter your HuggingFace Token: ").strip()
    
    return RUNPOD_ENDPOINT, RUNPOD_API_KEY, HUGGINGFACE_TOKEN

def create_dreambooth_payload(hf_token: str):
    """Create the DreamBooth training payload for RunPod"""
    
    # Encode the test.zip file
    dataset_b64 = encode_zip_file("test.zip")
    
    payload = {
        "input": {
            # Basic model settings
            "model_name": "test_flux_dreambooth_1000",
            "dataset": dataset_b64,  # Base64 encoded zip file
            "instance_prompt": "a photo of testsubject",  # DreamBooth instance prompt
            "class_prompt": "a photo of a person",  # DreamBooth class prompt
            
            # Model and training settings
            "base_model": "flux1-dev", 
            "steps": 1000,  # As requested
            "learning_rate": 2e-6,  # DreamBooth typically uses lower learning rate
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "resolution": "1024x1024",
            
            # FLUX DreamBooth specific settings
            "train_text_encoder": True,
            "with_prior_preservation": True,
            "prior_loss_weight": 1.0,
            "num_class_images": 50,
            
            # Memory optimization for RunPod
            "gradient_checkpointing": True,
            "mixed_precision": "bf16",
            
            # Environment configuration for RunPod
            "huggingface_token": hf_token,
        }
    }
    
    return payload

def test_dreambooth_training():
    """Test DreamBooth training on RunPod"""
    
    print("🚀 Testing DreamBooth Training on RunPod")
    print("=" * 50)
    
    # Get credentials
    RUNPOD_ENDPOINT, RUNPOD_API_KEY, HUGGINGFACE_TOKEN = get_api_credentials()
    
    if not all([RUNPOD_ENDPOINT, RUNPOD_API_KEY, HUGGINGFACE_TOKEN]):
        print("❌ Missing required credentials")
        return False
    
    # Create payload
    payload = create_dreambooth_payload(HUGGINGFACE_TOKEN)
    
    # Set up headers
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"📡 Sending request to: {RUNPOD_ENDPOINT}")
    print(f"🎯 Model name: {payload['input']['model_name']}")
    print(f"📊 Training steps: {payload['input']['steps']}")
    print(f"📝 Instance prompt: {payload['input']['instance_prompt']}")
    print(f"📝 Class prompt: {payload['input']['class_prompt']}")
    
    try:
        # Send request
        response = requests.post(RUNPOD_ENDPOINT, json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Request sent successfully!")
            print(f"📋 Job ID: {result.get('id', 'N/A')}")
            print(f"📊 Status: {result.get('status', 'N/A')}")
            
            # Monitor job status
            job_id = result.get('id')
            if job_id:
                monitor_job_status(job_id, RUNPOD_ENDPOINT, headers)
            
            return True
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error sending request: {e}")
        return False

def monitor_job_status(job_id: str, endpoint: str, headers: dict):
    """Monitor the job status"""
    
    print(f"\n📊 Monitoring job {job_id}...")
    print("=" * 30)
    
    while True:
        try:
            # Get job status
            status_response = requests.get(f"{endpoint}/{job_id}", headers=headers)
            
            if status_response.status_code == 200:
                job_data = status_response.json()
                status = job_data.get('status', 'unknown')
                
                print(f"⏰ {time.strftime('%H:%M:%S')} - Status: {status}")
                
                if status == 'COMPLETED':
                    print("🎉 Training completed successfully!")
                    
                    # Get results
                    if 'output' in job_data:
                        output = job_data['output']
                        print(f"📁 Output: {output}")
                    
                    break
                    
                elif status == 'FAILED':
                    print("❌ Training failed!")
                    if 'error' in job_data:
                        print(f"Error: {job_data['error']}")
                    break
                    
                elif status in ['IN_QUEUE', 'IN_PROGRESS']:
                    # Check for logs
                    if 'logs' in job_data:
                        logs = job_data['logs']
                        if logs:
                            print(f"📝 Latest log: {logs[-1] if isinstance(logs, list) else logs}")
                    
                    time.sleep(10)  # Wait 10 seconds before checking again
                else:
                    print(f"⚠️  Unknown status: {status}")
                    time.sleep(10)
            else:
                print(f"❌ Failed to get job status: {status_response.status_code}")
                break
                
        except Exception as e:
            print(f"❌ Error monitoring job: {e}")
            break

def main():
    """Main function"""
    print("🚀 RunPod DreamBooth Training Test")
    print("=" * 50)
    
    # Check if test.zip exists
    if not os.path.exists("test.zip"):
        print("❌ test.zip not found!")
        print("Please create a test.zip file with your training images and captions.")
        print("The zip should contain:")
        print("  - image1.jpg, image1.txt")
        print("  - image2.jpg, image2.txt")
        print("  - etc...")
        return
    
    # Test DreamBooth training
    success = test_dreambooth_training()
    
    if success:
        print("\n✅ DreamBooth training test completed!")
    else:
        print("\n❌ DreamBooth training test failed!")

if __name__ == "__main__":
    main() 