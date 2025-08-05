#!/usr/bin/env python3
"""
Deployment script for AI-Toolkit LoRA Training API
Helps build and deploy to various platforms including RunPod
"""

import os
import sys
import subprocess
import json
import argparse
from pathlib import Path

class DeploymentManager:
    def __init__(self):
        self.project_name = "ai-toolkit-lora-api"
        self.docker_image = None
        
    def build_docker_image(self, tag=None, registry=None):
        """Build Docker image"""
        
        if not tag:
            tag = "latest"
            
        if registry:
            self.docker_image = f"{registry}/{self.project_name}:{tag}"
        else:
            self.docker_image = f"{self.project_name}:{tag}"
            
        print(f"🔨 Building Docker image: {self.docker_image}")
        
        # Build command
        cmd = [
            "docker", "build",
            "-t", self.docker_image,
            "."
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Docker image built successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Docker build failed: {e}")
            print(f"Error output: {e.stderr}")
            return False
    
    def push_docker_image(self):
        """Push Docker image to registry"""
        
        if not self.docker_image:
            print("❌ No Docker image to push. Build first.")
            return False
            
        print(f"📤 Pushing Docker image: {self.docker_image}")
        
        cmd = ["docker", "push", self.docker_image]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Docker image pushed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Docker push failed: {e}")
            print(f"Error output: {e.stderr}")
            return False
    
    def test_local_container(self):
        """Test the Docker container locally"""
        
        if not self.docker_image:
            print("❌ No Docker image to test. Build first.")
            return False
            
        print(f"🧪 Testing Docker container locally: {self.docker_image}")
        
        # Run container with test input
        cmd = [
            "docker", "run", "--rm",
            "--gpus", "all",  # Requires nvidia-docker
            "-v", f"{os.getcwd()}:/workspace/host",
            self.docker_image,
            "python", "-c",
            """
import sys
sys.path.append('/workspace')
from handler import handler

# Simple test
test_job = {
    'input': {
        'model_name': 'test_local',
        'dataset': [],
        'steps': 10
    }
}

try:
    result = handler(test_job)
    print('✅ Local container test passed!')
    print(f'Result keys: {list(result.keys())}')
except Exception as e:
    print(f'❌ Local container test failed: {e}')
"""
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Local container test successful!")
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Local container test failed: {e}")
            print(f"Error output: {e.stderr}")
            return False
    
    def generate_runpod_template(self):
        """Generate RunPod template configuration"""
        
        template = {
            "name": "AI-Toolkit LoRA Training API",
            "image": self.docker_image or f"your-registry/{self.project_name}:latest",
            "containerDiskInGb": 20,
            "dockerArgs": "",
            "env": [
                {
                    "key": "PYTHONUNBUFFERED",
                    "value": "1"
                },
                {
                    "key": "CUDA_VISIBLE_DEVICES", 
                    "value": "0"
                }
            ],
            "volumeInGb": 50,
            "volumeMountPath": "/workspace",
            "ports": "8000/http",
            "startScript": "python handler.py"
        }
        
        template_file = "runpod_template.json"
        with open(template_file, 'w') as f:
            json.dump(template, f, indent=2)
            
        print(f"📋 RunPod template saved to: {template_file}")
        return template_file
    
    def generate_env_file(self):
        """Generate environment file template"""
        
        env_content = """# AI-Toolkit LoRA Training API Environment Variables

# Hugging Face Token (required for model downloads)
HF_TOKEN=your_hugging_face_token_here

# Weights & Biases (optional, for experiment tracking)
WANDB_API_KEY=your_wandb_api_key_here

# Memory optimization settings
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Logging level
LOG_LEVEL=INFO

# Cache directories
HF_HOME=/workspace/.cache/huggingface
TRANSFORMERS_CACHE=/workspace/.cache/transformers
"""
        
        env_file = ".env.example"
        with open(env_file, 'w') as f:
            f.write(env_content)
            
        print(f"📋 Environment template saved to: {env_file}")
        return env_file

def main():
    parser = argparse.ArgumentParser(description="Deploy AI-Toolkit LoRA Training API")
    parser.add_argument("--build", action="store_true", help="Build Docker image")
    parser.add_argument("--push", action="store_true", help="Push Docker image")
    parser.add_argument("--test", action="store_true", help="Test Docker container locally")
    parser.add_argument("--template", action="store_true", help="Generate RunPod template")
    parser.add_argument("--env", action="store_true", help="Generate environment file")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--tag", default="latest", help="Docker image tag")
    parser.add_argument("--registry", help="Docker registry (e.g., docker.io/username)")
    
    args = parser.parse_args()
    
    if not any([args.build, args.push, args.test, args.template, args.env, args.all]):
        parser.print_help()
        return
    
    deployer = DeploymentManager()
    
    print("🚀 AI-Toolkit LoRA Training API Deployment")
    print("=" * 50)
    
    success = True
    
    # Generate environment file
    if args.env or args.all:
        deployer.generate_env_file()
    
    # Build Docker image
    if args.build or args.all:
        success = deployer.build_docker_image(args.tag, args.registry)
        if not success:
            print("❌ Build failed, stopping deployment")
            sys.exit(1)
    
    # Test locally
    if (args.test or args.all) and success:
        # Skip local test if no GPU available
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True)
            if result.returncode == 0:
                deployer.test_local_container()
            else:
                print("⚠️  No GPU detected, skipping local container test")
        except FileNotFoundError:
            print("⚠️  nvidia-smi not found, skipping local container test")
    
    # Push to registry
    if (args.push or args.all) and success:
        if not args.registry:
            print("❌ Registry required for push. Use --registry flag")
            sys.exit(1)
        success = deployer.push_docker_image()
    
    # Generate RunPod template
    if args.template or args.all:
        deployer.generate_runpod_template()
    
    if success:
        print("\n✅ Deployment process completed successfully!")
        print("\nNext steps:")
        print("1. Copy your Docker image URL")
        print("2. Create a new RunPod Serverless template")
        print("3. Set up an endpoint with 24GB+ VRAM GPU")
        print("4. Test with the provided test_api.py script")
    else:
        print("\n❌ Deployment process failed")
        sys.exit(1)

if __name__ == "__main__":
    main()