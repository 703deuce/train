#!/usr/bin/env python3
"""
AI-Toolkit LoRA Training API Handler for RunPod Serverless
Supports FLUX LoRA training with comprehensive parameter configuration
"""

import os
import sys
import json
import yaml
import shutil
import logging
import tempfile
import traceback
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
import base64
import zipfile
import runpod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants - Updated for RunPod network storage support
WORKSPACE_PATH = "/workspace"
AI_TOOLKIT_PATH = "/workspace/ai-toolkit"

# Check for network storage (RunPod volume)
NETWORK_STORAGE_PATH = "/runpod-volume"
if os.path.exists(NETWORK_STORAGE_PATH):
    # Use network storage if available
    DATASETS_PATH = os.path.join(NETWORK_STORAGE_PATH, "datasets")
    OUTPUT_PATH = os.path.join(NETWORK_STORAGE_PATH, "outputs")
    CONFIG_PATH = os.path.join(NETWORK_STORAGE_PATH, "configs")
    CACHE_PATH = os.path.join(NETWORK_STORAGE_PATH, "cache")
else:
    # Fallback to workspace
    DATASETS_PATH = "/workspace/datasets"
    OUTPUT_PATH = "/workspace/outputs"
    CONFIG_PATH = "/workspace/configs"
    CACHE_PATH = "/workspace/cache"

# Ensure directories exist
os.makedirs(DATASETS_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True) 
os.makedirs(CONFIG_PATH, exist_ok=True)
os.makedirs(CACHE_PATH, exist_ok=True)

class LoRATrainingHandler:
    """Handler for LoRA training using ai-toolkit"""
    
    def __init__(self):
        self.setup_environment()
    
    def setup_environment(self):
        """Setup the training environment"""
        try:
            # Change to ai-toolkit directory if it exists
            if os.path.exists(AI_TOOLKIT_PATH):
                os.chdir(AI_TOOLKIT_PATH)
                logger.info(f"Changed directory to {AI_TOOLKIT_PATH}")
            else:
                logger.warning(f"AI-Toolkit path {AI_TOOLKIT_PATH} not found")
            
            # Log storage paths
            logger.info(f"Datasets path: {DATASETS_PATH}")
            logger.info(f"Output path: {OUTPUT_PATH}")
            logger.info(f"Cache path: {CACHE_PATH}")
            logger.info(f"Network storage available: {os.path.exists(NETWORK_STORAGE_PATH)}")
            
        except Exception as e:
            logger.error(f"Failed to setup environment: {e}")
    
    def validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize input parameters"""
        required_fields = ["model_name", "dataset"]
        
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Set defaults for optional parameters
        defaults = {
            # Basic settings
            "model_type": "flux",
            "base_model": "flux1-dev",
            "trigger_word": "",
            
            # Training parameters
            "steps": 2000,
            "learning_rate": 0.0003,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "resolution": "1024x1024",
            "max_bucket_resolution": 2048,
            "min_bucket_resolution": 256,
            
            # LoRA network settings
            "network_type": "lora",
            "network_dim": 16,
            "network_alpha": 16,
            "network_dropout": 0.0,
            
            # Optimizer settings
            "optimizer": "AdamW8bit",
            "lr_scheduler": "constant",
            "lr_warmup_steps": 0,
            
            # Memory optimization
            "fp8_base": True,
            "cache_latents": True,
            "cache_text_encoder_outputs": True,
            "gradient_checkpointing": True,
            
            # Sampling
            "sample_every_n_steps": 200,
            "sample_prompts": [],
            "guidance_scale": 3.5,
            "sample_steps": 20,
            
            # Saving
            "save_every_n_steps": 200,
            "save_model_as": "safetensors",
            
            # Advanced settings
            "noise_offset": 0.05,
            "min_snr_gamma": 7,
            "mixed_precision": "bf16",
            "seed": 42,
            
            # FLUX specific
            "blocks_to_swap": None,
            "apply_t5_attention_mask": True,
            "max_sequence_length": 512,
            "guidance_scale": 1.0,
            
            # LoRA specific layers (optional)
            "only_if_contains": [],
            "ignore_if_contains": [],
        }
        
        # Merge defaults with input
        for key, default_value in defaults.items():
            if key not in input_data:
                input_data[key] = default_value
        
        return input_data
    
    def process_dataset(self, dataset_data: Any, dataset_name: str) -> str:
        """Process and save dataset files"""
        dataset_path = os.path.join(DATASETS_PATH, dataset_name)
        os.makedirs(dataset_path, exist_ok=True)
        
        try:
            if isinstance(dataset_data, str):
                # Handle base64 encoded zip file
                if dataset_data.startswith('data:'):
                    # Remove data URL prefix
                    dataset_data = dataset_data.split(',')[1]
                
                # Decode base64
                zip_data = base64.b64decode(dataset_data)
                
                # Save and extract zip file
                zip_path = os.path.join(dataset_path, "dataset.zip")
                with open(zip_path, 'wb') as f:
                    f.write(zip_data)
                
                # Extract zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_path)
                
                # Remove zip file
                os.remove(zip_path)
                
            elif isinstance(dataset_data, list):
                # Handle list of files
                for i, file_data in enumerate(dataset_data):
                    if 'filename' in file_data and 'content' in file_data:
                        filename = file_data['filename']
                        content = file_data['content']
                        
                        if content.startswith('data:'):
                            content = content.split(',')[1]
                        
                        file_path = os.path.join(dataset_path, filename)
                        
                        # Decode and save file
                        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                            # Image file
                            image_data = base64.b64decode(content)
                            with open(file_path, 'wb') as f:
                                f.write(image_data)
                        else:
                            # Text file
                            text_data = base64.b64decode(content).decode('utf-8')
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(text_data)
            
            logger.info(f"Dataset processed successfully at {dataset_path}")
            return dataset_path
            
        except Exception as e:
            logger.error(f"Failed to process dataset: {e}")
            raise ValueError(f"Invalid dataset format: {e}")
    
    def generate_config(self, params: Dict[str, Any], dataset_path: str) -> str:
        """Generate YAML configuration file for training"""
        
        model_name = params["model_name"]
        output_dir = os.path.join(OUTPUT_PATH, model_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Build configuration dictionary
        config = {
            "job": "extension",
            "config": {
                # Model settings
                "name": model_name,
                "process": [
                    {
                        "type": "sd_trainer",
                        "training_folder": output_dir,
                        "device": "cuda:0",
                        "trigger_word": params.get("trigger_word", ""),
                        "network": {
                            "type": params["network_type"],
                            "linear": params["network_dim"],
                            "linear_alpha": params["network_alpha"],
                        },
                        "save": {
                            "dtype": params["mixed_precision"],
                            "save_every": params["save_every_n_steps"],
                            "max_step_saves_to_keep": 3,
                        },
                        "datasets": [
                            {
                                "folder_path": dataset_path,
                                "caption_ext": "txt",
                                "caption_dropout_rate": 0.0,
                                "shuffle_tokens": False,
                                "cache_latents_to_disk": params["cache_latents"],
                                "resolution": self._parse_resolution(params["resolution"]),
                            }
                        ],
                        "train": {
                            "batch_size": params["batch_size"],
                            "steps": params["steps"],
                            "gradient_accumulation_steps": params["gradient_accumulation_steps"],
                            "train_unet": True,
                            "train_text_encoder": False,
                            "gradient_checkpointing": params["gradient_checkpointing"],
                            "noise_scheduler": "flowmatch",
                            "optimizer": params["optimizer"],
                            "lr": params["learning_rate"],
                            "ema_config": {
                                "use_ema": True,
                                "ema_decay": 0.99,
                            },
                            "dtype": params["mixed_precision"],
                        },
                        "model": {
                            "name_or_path": self._get_model_path(params["base_model"]),
                            "is_flux": True,
                            "quantize": params.get("fp8_base", False),
                            "is_local": os.path.exists(self._get_model_path(params["base_model"])),
                        },
                        "sample": {
                            "enabled": True,
                            "every_n_steps": params["sample_every_n_steps"],
                            "seed": params["seed"],
                            "walk_seed": True,
                            "guidance_scale": params["guidance_scale"],
                            "sample_steps": params["sample_steps"],
                            "sampler": "euler",
                            "prompts": self._generate_sample_prompts(params),
                            "neg": "",
                            "width": self._parse_resolution(params["resolution"])[0],
                            "height": self._parse_resolution(params["resolution"])[1],
                        },
                    }
                ]
            }
        }
        
        # Add network kwargs if specified
        if params.get("only_if_contains") or params.get("ignore_if_contains"):
            network_kwargs = {}
            if params.get("only_if_contains"):
                network_kwargs["only_if_contains"] = params["only_if_contains"]
            if params.get("ignore_if_contains"):
                network_kwargs["ignore_if_contains"] = params["ignore_if_contains"]
            config["config"]["process"][0]["network"]["network_kwargs"] = network_kwargs
        
        # Add advanced training parameters
        if params.get("noise_offset", 0) > 0:
            config["config"]["process"][0]["train"]["noise_offset"] = params["noise_offset"]
        
        if params.get("min_snr_gamma"):
            config["config"]["process"][0]["train"]["min_snr_gamma"] = params["min_snr_gamma"]
        
        # FLUX specific settings
        if params.get("apply_t5_attention_mask"):
            config["config"]["process"][0]["model"]["apply_t5_attention_mask"] = True
        
        if params.get("max_sequence_length"):
            config["config"]["process"][0]["model"]["max_sequence_length"] = params["max_sequence_length"]
        
        # Save configuration file
        config_filename = f"{model_name}_config.yaml"
        config_path = os.path.join(CONFIG_PATH, config_filename)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
        return config_path
    
    def _parse_resolution(self, resolution: str) -> List[int]:
        """Parse resolution string to [width, height]"""
        if 'x' in resolution:
            width, height = map(int, resolution.split('x'))
            return [width, height]
        else:
            # Square resolution
            size = int(resolution)
            return [size, size]
    
    def _get_model_path(self, base_model: str) -> str:
        """Get the path for the base model, checking for cached versions first"""
        
        # Check for locally cached models in RunPod network storage
        potential_cache_paths = [
            os.path.join(CACHE_PATH, "flux1-dev"),
            os.path.join(CACHE_PATH, "FLUX.1-dev"),
            os.path.join(CACHE_PATH, "black-forest-labs--FLUX.1-dev"),
            os.path.join(CACHE_PATH, "models--black-forest-labs--FLUX.1-dev"),
        ]
        
        # Try different cache path patterns
        for cached_path in potential_cache_paths:
            if os.path.exists(cached_path):
                # Check if it contains model files
                if any(f.endswith(('.safetensors', '.bin', '.pt')) for f in os.listdir(cached_path) if os.path.isfile(os.path.join(cached_path, f))):
                    logger.info(f"Using cached model from: {cached_path}")
                    return cached_path
                # Check subdirectories
                for subdir in os.listdir(cached_path):
                    subpath = os.path.join(cached_path, subdir)
                    if os.path.isdir(subpath):
                        if any(f.endswith(('.safetensors', '.bin', '.pt')) for f in os.listdir(subpath) if os.path.isfile(os.path.join(subpath, f))):
                            logger.info(f"Using cached model from: {subpath}")
                            return subpath
        
        # Check for HuggingFace cache format
        hf_cache_path = os.path.expanduser("~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev")
        if os.path.exists(hf_cache_path):
            logger.info(f"Using HuggingFace cache: {hf_cache_path}")
            return hf_cache_path
        
        # If base_model looks like a local path, use it directly
        if os.path.exists(base_model):
            logger.info(f"Using direct path: {base_model}")
            return base_model
        
        # Fallback to HuggingFace Hub paths (will require authentication)
        model_paths = {
            "flux1-dev": "black-forest-labs/FLUX.1-dev",
            "flux1-schnell": "black-forest-labs/FLUX.1-schnell", 
            "flux1-dev2pro": "black-forest-labs/FLUX.1-dev",
        }
        
        hub_path = model_paths.get(base_model, base_model)
        logger.warning(f"No cached model found, trying HuggingFace Hub: {hub_path}")
        logger.warning(f"This may fail if model is gated. Check cache paths: {potential_cache_paths}")
        return hub_path
    
    def _generate_sample_prompts(self, params: Dict[str, Any]) -> List[str]:
        """Generate sample prompts for training validation"""
        trigger = params.get("trigger_word", "")
        
        if params.get("sample_prompts"):
            # Use custom prompts if provided
            prompts = params["sample_prompts"]
            if trigger:
                # Add trigger word to prompts that don't have it
                prompts = [p if trigger in p else f"{p}, {trigger}" for p in prompts]
            return prompts
        
        # Default prompts
        base_prompts = [
            "a portrait photo",
            "a professional headshot",
            "a close-up photo",
            "a full body photo",
        ]
        
        if trigger:
            prompts = [f"{prompt} of {trigger}" for prompt in base_prompts]
        else:
            prompts = base_prompts
        
        return prompts
    
    def run_training(self, config_path: str) -> Dict[str, Any]:
        """Execute the training process"""
        try:
            # Change to ai-toolkit directory
            original_dir = os.getcwd()
            if os.path.exists(AI_TOOLKIT_PATH):
                os.chdir(AI_TOOLKIT_PATH)
            
            # Run training command
            cmd = [sys.executable, "run.py", config_path]
            logger.info(f"Running training command: {' '.join(cmd)}")
            
            # Execute training
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output
            output_lines = []
            for line in process.stdout:
                line = line.strip()
                if line:
                    logger.info(f"Training: {line}")
                    output_lines.append(line)
            
            # Wait for completion
            return_code = process.wait()
            
            # Restore original directory
            os.chdir(original_dir)
            
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)
            
            logger.info("Training completed successfully")
            return {
                "status": "success",
                "output": output_lines[-50:] if len(output_lines) > 50 else output_lines  # Last 50 lines
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def collect_outputs(self, model_name: str) -> Dict[str, Any]:
        """Collect training outputs and prepare for download"""
        output_dir = os.path.join(OUTPUT_PATH, model_name)
        
        if not os.path.exists(output_dir):
            return {"error": "Output directory not found"}
        
        outputs = {
            "models": [],
            "samples": [],
            "logs": []
        }
        
        try:
            # Find model files
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, output_dir)
                    
                    if file.endswith('.safetensors'):
                        # Encode model file as base64
                        with open(file_path, 'rb') as f:
                            model_data = base64.b64encode(f.read()).decode('utf-8')
                        
                        outputs["models"].append({
                            "filename": file,
                            "path": relative_path,
                            "data": model_data,
                            "size": os.path.getsize(file_path)
                        })
                    
                    elif file.endswith(('.png', '.jpg', '.jpeg')):
                        # Encode sample images as base64
                        with open(file_path, 'rb') as f:
                            image_data = base64.b64encode(f.read()).decode('utf-8')
                        
                        outputs["samples"].append({
                            "filename": file,
                            "path": relative_path,
                            "data": f"data:image/png;base64,{image_data}",
                            "size": os.path.getsize(file_path)
                        })
                    
                    elif file.endswith('.log'):
                        # Include log files as text
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            log_content = f.read()
                        
                        outputs["logs"].append({
                            "filename": file,
                            "path": relative_path,
                            "content": log_content,
                            "size": os.path.getsize(file_path)
                        })
            
            return outputs
            
        except Exception as e:
            logger.error(f"Failed to collect outputs: {e}")
            return {"error": str(e)}


def handler(job):
    """Main RunPod handler function"""
    try:
        job_input = job["input"]
        logger.info(f"Received job with input keys: {list(job_input.keys())}")
        
        # Set up HuggingFace token if provided
        if "huggingface_token" in job_input:
            os.environ["HF_TOKEN"] = job_input["huggingface_token"]
            os.environ["HUGGING_FACE_HUB_TOKEN"] = job_input["huggingface_token"]
            logger.info("HuggingFace token configured")
        
        # Initialize handler
        trainer = LoRATrainingHandler()
        
        # Validate input
        params = trainer.validate_input(job_input)
        logger.info(f"Validated parameters for model: {params['model_name']}")
        
        # Process dataset
        dataset_path = trainer.process_dataset(
            params["dataset"], 
            params["model_name"]
        )
        logger.info(f"Dataset processed at: {dataset_path}")
        
        # Generate configuration
        config_path = trainer.generate_config(params, dataset_path)
        logger.info(f"Configuration generated: {config_path}")
        
        # Run training
        training_result = trainer.run_training(config_path)
        
        if training_result["status"] == "error":
            return {
                "error": training_result["error"],
                "traceback": training_result.get("traceback"),
                "status": "failed"
            }
        
        # Collect outputs
        outputs = trainer.collect_outputs(params["model_name"])
        
        return {
            "status": "completed",
            "model_name": params["model_name"],
            "training_steps": params["steps"],
            "training_output": training_result.get("output", []),
            "outputs": outputs
        }
        
    except Exception as e:
        logger.error(f"Handler error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": "failed"
        }


if __name__ == "__main__":
    # Start the serverless worker
    runpod.serverless.start({"handler": handler})