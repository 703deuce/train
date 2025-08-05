#!/usr/bin/env python3
"""
Test script for AI-Toolkit LoRA Training API
This script demonstrates how to use the API with example data
"""

import json
import base64
import requests
import time
from pathlib import Path

def encode_file_content(file_path):
    """Encode file content to base64"""
    try:
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found, using placeholder")
        return base64.b64encode(b"placeholder content").decode('utf-8')

def create_test_dataset():
    """Create a test dataset with sample images and captions"""
    dataset = []
    
    # Example dataset files
    sample_files = [
        ("image1.jpg", "a portrait photo of testchar, high quality"),
        ("image2.jpg", "testchar smiling, professional photography"),
        ("image3.jpg", "a close-up of testchar, detailed face"),
        ("image4.jpg", "testchar in casual clothing, full body shot"),
        ("image5.jpg", "artistic photo of testchar, dramatic lighting")
    ]
    
    for image_name, caption in sample_files:
        # Add image file (placeholder since we don't have actual images)
        dataset.append({
            "filename": image_name,
            "content": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="  # 1x1 transparent PNG
        })
        
        # Add caption file
        caption_name = image_name.replace('.jpg', '.txt')
        dataset.append({
            "filename": caption_name,
            "content": base64.b64encode(caption.encode('utf-8')).decode('utf-8')
        })
    
    return dataset

def test_basic_training():
    """Test basic LoRA training"""
    
    payload = {
        "input": {
            # Required parameters
            "model_name": "test_character_lora",
            "dataset": create_test_dataset(),
            
            # Basic training settings
            "trigger_word": "testchar",
            "base_model": "flux1-dev",
            "steps": 100,  # Low steps for testing
            "learning_rate": 0.0003,
            "batch_size": 1,
            "resolution": "512x512",  # Lower resolution for faster testing
            
            # LoRA network settings
            "network_dim": 16,
            "network_alpha": 16,
            
            # Memory optimization
            "fp8_base": True,
            "gradient_checkpointing": True,
            "cache_latents": True,
            
            # Sampling settings
            "sample_every_n_steps": 50,
            "sample_prompts": [
                "a portrait of testchar",
                "testchar smiling",
                "testchar in a professional outfit"
            ],
            
            # Save settings
            "save_every_n_steps": 50,
        }
    }
    
    return payload

def test_advanced_training():
    """Test advanced LoRA training with custom parameters"""
    
    payload = {
        "input": {
            # Required parameters
            "model_name": "advanced_style_lora",
            "dataset": create_test_dataset(),
            
            # Advanced training settings
            "trigger_word": "artstyle",
            "base_model": "flux1-dev",
            "steps": 200,
            "learning_rate": 0.0001,
            "batch_size": 1,
            "gradient_accumulation_steps": 2,
            "resolution": "1024x1024",
            
            # Advanced LoRA settings
            "network_type": "lora",
            "network_dim": 32,
            "network_alpha": 32,
            "network_dropout": 0.1,
            
            # Optimizer settings
            "optimizer": "AdamW8bit",
            "lr_scheduler": "cosine",
            "lr_warmup_steps": 10,
            
            # Memory and performance
            "fp8_base": True,
            "mixed_precision": "bf16",
            "gradient_checkpointing": True,
            "cache_latents": True,
            "cache_text_encoder_outputs": True,
            
            # Advanced training parameters
            "noise_offset": 0.1,
            "min_snr_gamma": 5,
            
            # Layer-specific training
            "only_if_contains": [
                "transformer.single_transformer_blocks"
            ],
            
            # Sampling
            "sample_every_n_steps": 25,
            "guidance_scale": 4.0,
            "sample_steps": 30,
            "sample_prompts": [
                "abstract art in artstyle",
                "landscape painting, artstyle",
                "portrait in artstyle, detailed"
            ]
        }
    }
    
    return payload

def test_local_handler():
    """Test the handler locally (for development)"""
    
    print("Testing handler locally...")
    
    # Import the handler
    from handler import handler
    
    # Create test job
    test_job = {
        "input": test_basic_training()["input"]
    }
    
    print("Running local test...")
    result = handler(test_job)
    
    print("Local test result:")
    print(json.dumps(result, indent=2)[:500] + "..." if len(str(result)) > 500 else json.dumps(result, indent=2))
    
    return result

def test_api_endpoint(endpoint_url):
    """Test the deployed API endpoint"""
    
    print(f"Testing API endpoint: {endpoint_url}")
    
    # Test basic training
    print("\n1. Testing basic training...")
    payload = test_basic_training()
    
    try:
        response = requests.post(endpoint_url, json=payload, timeout=3600)  # 1 hour timeout
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Basic training test successful!")
            print(f"Status: {result.get('status', 'unknown')}")
            print(f"Model name: {result.get('model_name', 'unknown')}")
            
            if 'outputs' in result:
                outputs = result['outputs']
                print(f"Models generated: {len(outputs.get('models', []))}")
                print(f"Samples generated: {len(outputs.get('samples', []))}")
        else:
            print(f"❌ API request failed with status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
    except Exception as e:
        print(f"❌ Request failed: {e}")

def test_with_zip_dataset():
    """Test with zip file dataset format"""
    
    # Create a mock zip file content (base64 encoded)
    # In real usage, you would read an actual zip file
    mock_zip_content = "UEsDBBQAAAAIAOKOVlcAAAAAAAAAAAAEAAAAdGVzdC50eHRLT8ksrEqpzUvNLQQAAAAAUEsBAhQAFAAAAAgA4o5WVwAAAAAAAAAAAAQAAAB0ZXN0LnR4dFBLBQYAAAAAAQABADIAAAAiAAAAAAA="
    
    payload = {
        "input": {
            "model_name": "zip_test_lora",
            "dataset": mock_zip_content,  # Base64 encoded zip
            "trigger_word": "ziptest",
            "steps": 50,
            "resolution": "512x512"
        }
    }
    
    return payload

def validate_parameters():
    """Validate parameter examples"""
    
    print("Validating parameter configurations...")
    
    # Test different parameter combinations
    configs = [
        {
            "name": "Memory Optimized",
            "params": {
                "fp8_base": True,
                "gradient_checkpointing": True,
                "cache_latents": True,
                "mixed_precision": "bf16",
                "batch_size": 1
            }
        },
        {
            "name": "High Quality",
            "params": {
                "network_dim": 64,
                "network_alpha": 64,
                "steps": 3000,
                "learning_rate": 0.0001,
                "resolution": "1024x1024"
            }
        },
        {
            "name": "Fast Training",
            "params": {
                "steps": 500,
                "resolution": "512x512",
                "sample_every_n_steps": 100,
                "save_every_n_steps": 250
            }
        }
    ]
    
    for config in configs:
        print(f"\n✅ {config['name']} configuration:")
        for key, value in config['params'].items():
            print(f"  {key}: {value}")

def main():
    """Main test function"""
    
    print("🚀 AI-Toolkit LoRA Training API Test Suite")
    print("=" * 50)
    
    # Validate parameters
    validate_parameters()
    
    # Test local handler if available
    try:
        test_local_handler()
    except ImportError:
        print("Handler not available for local testing (deploy environment)")
    except Exception as e:
        print(f"Local test failed: {e}")
    
    # Get endpoint URL from user
    endpoint_url = input("\nEnter your RunPod endpoint URL (or press Enter to skip): ").strip()
    
    if endpoint_url:
        test_api_endpoint(endpoint_url)
    else:
        print("Skipping API endpoint test")
    
    print("\n📋 Test payload examples:")
    print("\nBasic training payload:")
    print(json.dumps(test_basic_training(), indent=2))
    
    print("\nAdvanced training payload:")
    print(json.dumps(test_advanced_training(), indent=2)[:1000] + "...")
    
    print("\n✅ Tests completed!")

if __name__ == "__main__":
    main()