#!/usr/bin/env python3
"""
Example client for AI-Toolkit LoRA Training API
Demonstrates practical usage patterns for different types of LoRA training
"""

import os
import json
import base64
import requests
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

class LoRATrainingClient:
    """Client for interacting with the AI-Toolkit LoRA Training API"""
    
    def __init__(self, endpoint_url: str, timeout: int = 3600):
        self.endpoint_url = endpoint_url
        self.timeout = timeout
    
    def encode_file(self, file_path: str) -> str:
        """Encode a file to base64"""
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def prepare_dataset_from_folder(self, folder_path: str) -> List[Dict[str, str]]:
        """Prepare dataset from a folder containing images and caption files"""
        dataset = []
        folder = Path(folder_path)
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        image_files = [f for f in folder.glob('*') if f.suffix.lower() in image_extensions]
        
        for image_file in image_files:
            # Add image
            dataset.append({
                "filename": image_file.name,
                "content": self.encode_file(str(image_file))
            })
            
            # Look for corresponding caption file
            caption_file = image_file.with_suffix('.txt')
            if caption_file.exists():
                dataset.append({
                    "filename": caption_file.name,
                    "content": self.encode_file(str(caption_file))
                })
            else:
                # Create a basic caption if none exists
                basic_caption = f"an image"
                dataset.append({
                    "filename": caption_file.name,
                    "content": base64.b64encode(basic_caption.encode()).decode()
                })
        
        print(f"Prepared dataset with {len(image_files)} images from {folder_path}")
        return dataset
    
    def train_character_lora(
        self,
        model_name: str,
        dataset_path: str,
        trigger_word: str,
        steps: int = 2000,
        **kwargs
    ) -> Dict[str, Any]:
        """Train a character LoRA"""
        
        print(f"🎭 Training character LoRA: {model_name}")
        
        payload = {
            "input": {
                "model_name": model_name,
                "dataset": self.prepare_dataset_from_folder(dataset_path),
                "trigger_word": trigger_word,
                "base_model": "flux1-dev",
                "steps": steps,
                
                # Character LoRA optimized settings
                "learning_rate": 0.0003,
                "network_dim": 32,
                "network_alpha": 32,
                "resolution": "1024x1024",
                "batch_size": 1,
                
                # Memory optimization
                "fp8_base": True,
                "gradient_checkpointing": True,
                "cache_latents": True,
                
                # Sampling for character
                "sample_every_n_steps": 200,
                "sample_prompts": [
                    f"a portrait of {trigger_word}",
                    f"{trigger_word} smiling",
                    f"a professional photo of {trigger_word}",
                    f"{trigger_word} in casual clothing"
                ],
                
                **kwargs
            }
        }
        
        return self._send_request(payload)
    
    def train_style_lora(
        self,
        model_name: str,
        dataset_path: str,
        trigger_word: str,
        steps: int = 1500,
        **kwargs
    ) -> Dict[str, Any]:
        """Train a style LoRA"""
        
        print(f"🎨 Training style LoRA: {model_name}")
        
        payload = {
            "input": {
                "model_name": model_name,
                "dataset": self.prepare_dataset_from_folder(dataset_path),
                "trigger_word": trigger_word,
                "base_model": "flux1-dev",
                "steps": steps,
                
                # Style LoRA optimized settings
                "learning_rate": 0.0001,
                "network_dim": 64,
                "network_alpha": 64,
                "resolution": "1024x1024",
                "batch_size": 1,
                
                # Focus on style layers
                "only_if_contains": [
                    "transformer.single_transformer_blocks"
                ],
                
                # Style sampling
                "sample_every_n_steps": 150,
                "sample_prompts": [
                    f"abstract art in {trigger_word} style",
                    f"landscape painting, {trigger_word}",
                    f"portrait in {trigger_word} style",
                    f"digital art, {trigger_word}"
                ],
                
                **kwargs
            }
        }
        
        return self._send_request(payload)
    
    def train_concept_lora(
        self,
        model_name: str,
        dataset_path: str,
        trigger_word: str,
        steps: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """Train a concept/object LoRA"""
        
        print(f"💡 Training concept LoRA: {model_name}")
        
        payload = {
            "input": {
                "model_name": model_name,
                "dataset": self.prepare_dataset_from_folder(dataset_path),
                "trigger_word": trigger_word,
                "base_model": "flux1-dev",
                "steps": steps,
                
                # Concept LoRA settings
                "learning_rate": 0.0005,
                "network_dim": 16,
                "network_alpha": 16,
                "resolution": "1024x1024",
                "batch_size": 1,
                
                # Concept sampling
                "sample_every_n_steps": 100,
                "sample_prompts": [
                    f"a {trigger_word}",
                    f"multiple {trigger_word}",
                    f"a detailed view of {trigger_word}",
                    f"{trigger_word} in different lighting"
                ],
                
                **kwargs
            }
        }
        
        return self._send_request(payload)
    
    def _send_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send training request to API"""
        
        print(f"📡 Sending training request to {self.endpoint_url}")
        print(f"Model: {payload['input']['model_name']}")
        print(f"Steps: {payload['input']['steps']}")
        print(f"Trigger: {payload['input']['trigger_word']}")
        
        try:
            response = requests.post(
                self.endpoint_url,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Training completed: {result.get('status', 'unknown')}")
                return result
            else:
                print(f"❌ Request failed with status: {response.status_code}")
                print(f"Response: {response.text}")
                return {"error": f"HTTP {response.status_code}", "details": response.text}
                
        except requests.exceptions.Timeout:
            print("❌ Request timed out")
            return {"error": "timeout"}
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return {"error": str(e)}
    
    def save_outputs(self, result: Dict[str, Any], output_dir: str = "outputs"):
        """Save training outputs to local directory"""
        
        if "outputs" not in result:
            print("No outputs to save")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        outputs = result["outputs"]
        
        # Save models
        for model in outputs.get("models", []):
            model_path = output_path / model["filename"]
            model_data = base64.b64decode(model["data"])
            
            with open(model_path, 'wb') as f:
                f.write(model_data)
            
            print(f"💾 Saved model: {model_path} ({model['size']} bytes)")
        
        # Save sample images
        for sample in outputs.get("samples", []):
            sample_path = output_path / sample["filename"]
            
            # Remove data URL prefix if present
            image_data = sample["data"]
            if image_data.startswith('data:'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            
            with open(sample_path, 'wb') as f:
                f.write(image_bytes)
            
            print(f"🖼️  Saved sample: {sample_path} ({sample['size']} bytes)")
        
        # Save logs
        for log in outputs.get("logs", []):
            log_path = output_path / log["filename"]
            
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(log["content"])
            
            print(f"📝 Saved log: {log_path} ({log['size']} bytes)")

def main():
    """Example usage of the LoRA training client"""
    
    print("🚀 AI-Toolkit LoRA Training Client Examples")
    print("=" * 50)
    
    # Get endpoint URL
    endpoint_url = input("Enter your RunPod endpoint URL: ").strip()
    if not endpoint_url:
        print("❌ Endpoint URL required")
        return
    
    # Initialize client
    client = LoRATrainingClient(endpoint_url)
    
    # Example 1: Character LoRA
    print("\n📋 Example 1: Character LoRA Training")
    character_example = {
        "model_name": "my_character",
        "dataset_path": "./datasets/character_photos",  # Update with your path
        "trigger_word": "mychar",
        "steps": 1500
    }
    
    print("Configuration:")
    for key, value in character_example.items():
        print(f"  {key}: {value}")
    
    if input("\nRun character LoRA training? (y/n): ").lower() == 'y':
        if os.path.exists(character_example["dataset_path"]):
            result = client.train_character_lora(**character_example)
            client.save_outputs(result, f"outputs/{character_example['model_name']}")
        else:
            print(f"❌ Dataset path not found: {character_example['dataset_path']}")
    
    # Example 2: Style LoRA
    print("\n📋 Example 2: Style LoRA Training")
    style_example = {
        "model_name": "anime_style",
        "dataset_path": "./datasets/anime_art",  # Update with your path
        "trigger_word": "animestyle",
        "steps": 2000
    }
    
    print("Configuration:")
    for key, value in style_example.items():
        print(f"  {key}: {value}")
    
    if input("\nRun style LoRA training? (y/n): ").lower() == 'y':
        if os.path.exists(style_example["dataset_path"]):
            result = client.train_style_lora(**style_example)
            client.save_outputs(result, f"outputs/{style_example['model_name']}")
        else:
            print(f"❌ Dataset path not found: {style_example['dataset_path']}")
    
    # Example 3: Concept LoRA
    print("\n📋 Example 3: Concept LoRA Training")
    concept_example = {
        "model_name": "cool_gadget",
        "dataset_path": "./datasets/gadget_photos",  # Update with your path
        "trigger_word": "coolgadget",
        "steps": 1000
    }
    
    print("Configuration:")
    for key, value in concept_example.items():
        print(f"  {key}: {value}")
    
    if input("\nRun concept LoRA training? (y/n): ").lower() == 'y':
        if os.path.exists(concept_example["dataset_path"]):
            result = client.train_concept_lora(**concept_example)
            client.save_outputs(result, f"outputs/{concept_example['model_name']}")
        else:
            print(f"❌ Dataset path not found: {concept_example['dataset_path']}")
    
    print("\n✅ Examples completed!")
    print("\nTips:")
    print("- Prepare your datasets in folders with images and .txt caption files")
    print("- Use descriptive trigger words that don't conflict with common terms") 
    print("- Start with lower step counts for testing, increase for better quality")
    print("- Monitor the sample outputs to judge training progress")

if __name__ == "__main__":
    main()