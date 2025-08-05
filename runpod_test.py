#!/usr/bin/env python3
"""
RunPod Test Script for AI-Toolkit LoRA Training
Configured for RunPod endpoint with network storage

Environment Variables Required:
- RUNPOD_ENDPOINT: Your RunPod serverless endpoint URL
- RUNPOD_API_KEY: Your RunPod API key  
- HUGGINGFACE_TOKEN: Your HuggingFace access token

If not set as environment variables, the script will prompt for them.
"""

import os
import json
import base64
import requests
import time
from pathlib import Path

# RunPod configuration - All set via environment variables
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT", "")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

def get_api_credentials():
    """Get API credentials from environment or user input"""
    global RUNPOD_ENDPOINT, RUNPOD_API_KEY, HUGGINGFACE_TOKEN
    
    # Check RunPod Endpoint
    if not RUNPOD_ENDPOINT:
        RUNPOD_ENDPOINT = input("Enter your RunPod Endpoint URL: ").strip()
        if not RUNPOD_ENDPOINT:
            print("❌ RunPod Endpoint URL is required")
            return False
    
    # Check RunPod API Key
    if not RUNPOD_API_KEY:
        RUNPOD_API_KEY = input("Enter your RunPod API Key: ").strip()
        if not RUNPOD_API_KEY:
            print("❌ RunPod API Key is required")
            return False
    
    # Check HuggingFace Token
    if not HUGGINGFACE_TOKEN:
        HUGGINGFACE_TOKEN = input("Enter your HuggingFace Token: ").strip()
        if not HUGGINGFACE_TOKEN:
            print("❌ HuggingFace Token is required")
            return False
    
    return True

def encode_zip_file(zip_path):
    """Encode the test.zip file to base64"""
    print(f"📦 Encoding {zip_path}...")
    
    with open(zip_path, 'rb') as f:
        zip_data = f.read()
        encoded = base64.b64encode(zip_data).decode('utf-8')
    
    print(f"✅ Encoded {len(zip_data)} bytes to base64")
    return encoded

def create_training_payload():
    """Create the training payload for RunPod"""
    
    # Encode the test.zip file
    dataset_b64 = encode_zip_file("test.zip")
    
    payload = {
        "input": {
            # Basic model settings
            "model_name": "test_flux_lora_1000",
            "dataset": dataset_b64,  # Base64 encoded zip file
            "trigger_word": "testsubject",  # Adjust based on your images
            
            # Model and training settings
            "base_model": "flux1-dev", 
            "steps": 1000,  # As requested
            "learning_rate": 0.0003,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "resolution": "1024x1024",
            
            # LoRA network settings
            "network_type": "lora",
            "network_dim": 16,
            "network_alpha": 16,
            
            # Memory optimization for RunPod
            "fp8_base": True,
            "gradient_checkpointing": True,
            "cache_latents": True,
            "cache_text_encoder_outputs": True,
            "mixed_precision": "bf16",
            
            # Sampling settings
            "sample_every_n_steps": 100,  # Sample every 100 steps
            "sample_prompts": [
                "a portrait of testsubject",
                "testsubject smiling",
                "a professional photo of testsubject",
                "testsubject in different lighting"
            ],
            "guidance_scale": 3.5,
            "sample_steps": 20,
            
            # Save settings for network storage
            "save_every_n_steps": 200,
            "save_model_as": "safetensors",
            
            # Advanced settings
            "noise_offset": 0.05,
            "min_snr_gamma": 7,
            "seed": 42,
            
            # Environment configuration for RunPod
            "huggingface_token": HUGGINGFACE_TOKEN,
            
            # Network storage paths (RunPod specific)
            "use_network_storage": True,
            "network_storage_path": "/runpod-volume",
            "cache_path": "/runpod-volume/cache",
            "output_path": "/runpod-volume/outputs",
            
            # FLUX model path (using cached weights)
            "model_cache_path": "/runpod-volume/cache/flux1-dev"
        }
    }
    
    return payload

def send_training_request():
    """Send the training request to RunPod"""
    
    print("🚀 Preparing training request for RunPod...")
    print(f"📡 Endpoint: {RUNPOD_ENDPOINT}")
    print(f"🎯 Model: test_flux_lora_1000")
    print(f"📊 Steps: 1000")
    print(f"💾 Network Storage: /runpod-volume")
    
    # Create payload
    payload = create_training_payload()
    
    # Request headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    print(f"\n📤 Sending request...")
    print(f"Dataset size: {len(payload['input']['dataset'])} characters (base64)")
    print(f"Trigger word: {payload['input']['trigger_word']}")
    
    try:
        # Send request
        response = requests.post(
            RUNPOD_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=7200  # 2 hours timeout
        )
        
        print(f"\n📥 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Training request successful!")
            
            # Print result summary
            if result.get("status") == "completed":
                print(f"🎉 Training completed successfully!")
                
                if "outputs" in result:
                    outputs = result["outputs"]
                    print(f"📁 Models generated: {len(outputs.get('models', []))}")
                    print(f"🖼️  Samples generated: {len(outputs.get('samples', []))}")
                    print(f"📝 Logs generated: {len(outputs.get('logs', []))}")
                    
                    # List model files
                    for model in outputs.get('models', []):
                        print(f"  📄 {model['filename']} ({model.get('size', 0)} bytes)")
                
                return result
                
            elif result.get("status") == "failed":
                print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
                if "traceback" in result:
                    print(f"🔍 Error details:\n{result['traceback']}")
                return result
                
            else:
                print(f"⏳ Training status: {result.get('status', 'unknown')}")
                return result
        
        else:
            print(f"❌ Request failed with status: {response.status_code}")
            print(f"Response: {response.text}")
            return {"error": f"HTTP {response.status_code}", "details": response.text}
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (training may still be running)")
        return {"error": "timeout"}
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return {"error": str(e)}

def save_training_outputs(result, output_dir="runpod_outputs"):
    """Save the training outputs locally"""
    
    if "outputs" not in result:
        print("No outputs to save")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    outputs = result["outputs"]
    
    print(f"\n💾 Saving outputs to {output_dir}/...")
    
    # Save models
    for model in outputs.get("models", []):
        model_path = output_path / model["filename"]
        model_data = base64.b64decode(model["data"])
        
        with open(model_path, 'wb') as f:
            f.write(model_data)
        
        print(f"📄 Saved: {model_path} ({len(model_data)} bytes)")
    
    # Save sample images
    for sample in outputs.get("samples", []):
        sample_path = output_path / sample["filename"]
        
        # Handle data URL format
        image_data = sample["data"]
        if image_data.startswith('data:'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        
        with open(sample_path, 'wb') as f:
            f.write(image_bytes)
        
        print(f"🖼️  Saved: {sample_path} ({len(image_bytes)} bytes)")
    
    # Save logs
    for log in outputs.get("logs", []):
        log_path = output_path / log["filename"]
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(log["content"])
        
        print(f"📝 Saved: {log_path}")
    
    print(f"✅ All outputs saved to {output_dir}/")

def main():
    """Main test function"""
    
    print("🚀 RunPod AI-Toolkit LoRA Training Test")
    print("=" * 50)
    
    # Get API credentials
    if not get_api_credentials():
        return
    
    print(f"📂 Dataset: test.zip")
    print(f"🎯 Model: test_flux_lora_1000")
    print(f"📊 Steps: 1000")
    print(f"💾 Storage: Network volume (/runpod-volume)")
    print(f"🔑 HF Token: {HUGGINGFACE_TOKEN[:10]}...")
    print("=" * 50)
    
    # Check if test.zip exists
    if not os.path.exists("test.zip"):
        print("❌ test.zip not found in current directory")
        print("Please ensure test.zip contains your training images and caption files")
        return
    
    # Get file size
    zip_size = os.path.getsize("test.zip")
    print(f"📦 test.zip size: {zip_size:,} bytes")
    
    # Confirm before sending
    confirm = input("\n🚀 Ready to start training? This will use RunPod credits. (y/n): ").lower()
    if confirm != 'y':
        print("❌ Training cancelled")
        return
    
    # Send training request
    result = send_training_request()
    
    # Save outputs if successful
    if result and result.get("status") == "completed":
        save_training_outputs(result)
    
    # Print final summary
    print(f"\n📋 Training Summary:")
    print(f"Status: {result.get('status', 'unknown')}")
    print(f"Model: test_flux_lora_1000")
    print(f"Endpoint: {RUNPOD_ENDPOINT}")
    
    if result.get("status") == "completed":
        print(f"✅ Training completed successfully!")
        print(f"📁 Check runpod_outputs/ for downloaded files")
        print(f"💾 Original files saved to /runpod-volume on your RunPod")
    elif result.get("status") == "failed":
        print(f"❌ Training failed - check error details above")
    
    return result

if __name__ == "__main__":
    main()