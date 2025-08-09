#!/usr/bin/env python3
"""
AI-Toolkit DreamBooth Training API Handler for RunPod Serverless
Supports FLUX DreamBooth fine-tuning with comprehensive parameter configuration
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

class DreamBoothTrainingHandler:
    """Handler for DreamBooth fine-tuning using ai-toolkit"""
    
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
        
        # Set defaults for optional parameters (only supported by FLUX DreamBooth)
        defaults = {
            # Basic settings
            "model_type": "flux",
            "base_model": "flux1-dev",
            "instance_prompt": "",
            "class_prompt": "",
            
            # Training parameters
            "steps": 2000,
            "learning_rate": 2e-6,  # DreamBooth typically uses lower learning rate
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "resolution": "1024x1024",
            
            # DreamBooth settings
            "train_text_encoder": True,
            "with_prior_preservation": True,
            "prior_loss_weight": 1.0,
            "num_class_images": 50,
            
            # Optimizer settings
            "lr_scheduler": "constant",
            
            # Memory optimization (only supported ones)
            "gradient_checkpointing": True,
            "mixed_precision": "bf16",
            
            # Model caching and download settings
            "download_to_network_storage": False,
            "model_cache_dir": "",
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
                
                # Flatten directory structure if needed
                self._flatten_dataset_directory(dataset_path)
                
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
    
    def _flatten_dataset_directory(self, dataset_path: str):
        """Flatten dataset directory structure to ensure images are directly accessible"""
        try:
            logger.info(f"Checking dataset structure at: {dataset_path}")
            
            # Get all items in the dataset directory
            items = os.listdir(dataset_path)
            logger.info(f"Found items in dataset: {items}")
            
            # Check if we have image files directly in the dataset directory
            image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
            direct_images = [item for item in items if os.path.isfile(os.path.join(dataset_path, item)) 
                           and any(item.lower().endswith(ext) for ext in image_extensions)]
            
            if direct_images:
                logger.info(f"Found {len(direct_images)} images directly in dataset directory")
                return  # Images are already at the root level
            
            # Look for subdirectories that might contain images
            subdirs = [item for item in items if os.path.isdir(os.path.join(dataset_path, item))]
            logger.info(f"Found subdirectories: {subdirs}")
            
            if not subdirs:
                logger.warning("No subdirectories found and no images at root level")
                return
            
            # If there's only one subdirectory, flatten it
            if len(subdirs) == 1:
                subdir_name = subdirs[0]
                subdir_path = os.path.join(dataset_path, subdir_name)
                logger.info(f"Flattening single subdirectory: {subdir_name}")
                
                # Move all files from subdirectory to main directory
                for item in os.listdir(subdir_path):
                    src = os.path.join(subdir_path, item)
                    dst = os.path.join(dataset_path, item)
                    
                    if os.path.exists(dst):
                        # If file already exists, remove it first
                        os.remove(dst)
                    
                    # Move file
                    shutil.move(src, dst)
                
                # Remove the now-empty subdirectory
                os.rmdir(subdir_path)
                logger.info(f"Successfully flattened dataset structure")
                
            else:
                # Multiple subdirectories - check each one for images
                logger.info(f"Multiple subdirectories found, checking each for images")
                for subdir_name in subdirs:
                    subdir_path = os.path.join(dataset_path, subdir_name)
                    subdir_images = [item for item in os.listdir(subdir_path) 
                                   if os.path.isfile(os.path.join(subdir_path, item)) 
                                   and any(item.lower().endswith(ext) for ext in image_extensions)]
                    
                    if subdir_images:
                        logger.info(f"Found {len(subdir_images)} images in subdirectory: {subdir_name}")
                        # Move images from this subdirectory to root
                        for item in subdir_images:
                            src = os.path.join(subdir_path, item)
                            dst = os.path.join(dataset_path, item)
                            
                            if os.path.exists(dst):
                                os.remove(dst)
                            
                            shutil.move(src, dst)
                        
                        # Remove the subdirectory if it's now empty
                        if not os.listdir(subdir_path):
                            os.rmdir(subdir_path)
                            logger.info(f"Removed empty subdirectory: {subdir_name}")
                
        except Exception as e:
            logger.warning(f"Failed to flatten dataset directory: {e}")
            # Don't raise error, continue with original structure

    def generate_config(self, params: Dict[str, Any], dataset_path: str) -> str:
        """Generate command-line arguments for FLUX DreamBooth training"""
        
        model_name = params["model_name"]
        output_dir = os.path.join(OUTPUT_PATH, model_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Build command-line arguments for FLUX DreamBooth training
        cmd_args = [
            "accelerate", "launch", "/workspace/dreambooth/train_dreambooth_flux.py",
            "--pretrained_model_name_or_path", self._get_model_path(params["base_model"], params),
            "--instance_data_dir", dataset_path,
            "--output_dir", output_dir,
            "--instance_prompt", params.get("instance_prompt", ""),
            "--mixed_precision", params["mixed_precision"],
            "--resolution", str(self._parse_resolution(params["resolution"])[0]),
            "--train_batch_size", str(params["batch_size"]),
            "--guidance_scale", "1",
            "--gradient_accumulation_steps", str(params["gradient_accumulation_steps"]),
            "--optimizer", "prodigy",
            "--learning_rate", "1.0",
            "--lr_scheduler", params["lr_scheduler"],
            "--lr_warmup_steps", "0",
            "--max_train_steps", str(params["steps"]),
            "--train_text_encoder",
            "--logging_dir", "logs",
            "--max_sequence_length", "512",
            "--dataloader_num_workers", "0",
            "--gradient_checkpointing",
        ]
        
        # Add prior preservation if enabled
        if params.get("with_prior_preservation", False):  # Changed from True to False
            cmd_args.extend([
                "--with_prior_preservation",
                "--prior_loss_weight", str(params.get("prior_loss_weight", 1.0)),
                "--num_class_images", str(params.get("num_class_images", 50)),
            ])
            
            # Add class data directory (required for prior preservation)
            class_data_dir = params.get("class_data_dir")
            if not class_data_dir:
                # Create a default class data directory
                class_data_dir = os.path.join(DATASETS_PATH, f"{model_name}_class_images")
                os.makedirs(class_data_dir, exist_ok=True)
                logger.info(f"Created default class data directory: {class_data_dir}")
            
            cmd_args.extend(["--class_data_dir", class_data_dir])
        
        # Add validation parameters for better training monitoring
        cmd_args.extend([
            "--validation_prompt", params.get("instance_prompt", ""),
            "--validation_epochs", "25",
            "--seed", "0",
        ])
        
        # Save command to file for execution
        config_filename = f"{model_name}_cmd.json"
        config_path = os.path.join(CONFIG_PATH, config_filename)
        
        with open(config_path, 'w') as f:
            json.dump(cmd_args, f)
        
        logger.info(f"Command saved to {config_path}")
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
    
    def _get_model_path(self, base_model: str, params: Dict[str, Any]) -> str:
        """Get the path for the base model, checking for cached versions first"""
        
        # Check if custom model cache directory is specified
        custom_cache_dir = params.get("model_cache_dir", "")
        if custom_cache_dir and os.path.exists(custom_cache_dir):
            logger.info(f"Checking custom model cache directory: {custom_cache_dir}")
            # Look for FLUX model in custom cache
            for model_dir in os.listdir(custom_cache_dir):
                model_path = os.path.join(custom_cache_dir, model_dir)
                if os.path.isdir(model_path):
                    # Check if this looks like a HuggingFace model (has model_index.json)
                    if os.path.exists(os.path.join(model_path, "model_index.json")):
                        logger.info(f"Using custom cached model from: {model_path}")
                        return model_path
        
        # First check if FLUX weights are directly in the cache directory
        if base_model in ["flux1-dev", "flux1-schnell"]:
            # Check if model files are directly in CACHE_PATH
            if os.path.exists(CACHE_PATH):
                try:
                    cache_files = os.listdir(CACHE_PATH)
                    model_files = [f for f in cache_files if f.endswith(('.safetensors', '.bin', '.pt'))]
                    if model_files:
                        logger.info(f"Using cached model files directly from: {CACHE_PATH}")
                        logger.info(f"Found model files: {model_files[:3]}{'...' if len(model_files) > 3 else ''}")
                        return CACHE_PATH
                except Exception as e:
                    logger.warning(f"Error accessing cache directory: {e}")
        
        # Check for subdirectories in cache
        potential_cache_paths = [
            os.path.join(CACHE_PATH, "flux1-dev"),
            os.path.join(CACHE_PATH, "FLUX.1-dev"),
            os.path.join(CACHE_PATH, "black-forest-labs--FLUX.1-dev"),
            os.path.join(CACHE_PATH, "models--black-forest-labs--FLUX.1-dev"),
        ]
        
        # Add flux-models directory to potential paths
        flux_models_dir = os.path.join(NETWORK_STORAGE_PATH, "flux-models")
        if os.path.exists(flux_models_dir):
            potential_cache_paths.insert(0, flux_models_dir)
            # Also check for subdirectories in flux-models
            try:
                for subdir in os.listdir(flux_models_dir):
                    subpath = os.path.join(flux_models_dir, subdir)
                    if os.path.isdir(subpath):
                        potential_cache_paths.insert(0, subpath)
            except:
                pass
        
        # Try different cache path patterns
        for cached_path in potential_cache_paths:
            if os.path.exists(cached_path):
                try:
                    # Check if this is a proper HuggingFace model directory
                    if os.path.exists(os.path.join(cached_path, "model_index.json")):
                        logger.info(f"Using HuggingFace model from: {cached_path}")
                        return cached_path
                    
                    # Check if it contains model files
                    files = os.listdir(cached_path)
                    model_files = [f for f in files if f.endswith(('.safetensors', '.bin', '.pt'))]
                    if model_files:
                        logger.info(f"Using cached model from: {cached_path}")
                        return cached_path
                    
                    # Check subdirectories
                    for subdir in files:
                        subpath = os.path.join(cached_path, subdir)
                        if os.path.isdir(subpath):
                            try:
                                if os.path.exists(os.path.join(subpath, "model_index.json")):
                                    logger.info(f"Using HuggingFace model from: {subpath}")
                                    return subpath
                                
                                subfiles = os.listdir(subpath)
                                sub_model_files = [f for f in subfiles if f.endswith(('.safetensors', '.bin', '.pt'))]
                                if sub_model_files:
                                    logger.info(f"Using cached model from: {subpath}")
                                    return subpath
                            except:
                                continue
                except Exception as e:
                    logger.warning(f"Error accessing {cached_path}: {e}")
        
        # Check for HuggingFace cache format
        hf_cache_path = os.path.expanduser("~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev")
        if os.path.exists(hf_cache_path):
            logger.info(f"Using HuggingFace cache: {hf_cache_path}")
            return hf_cache_path
        
        # If base_model looks like a local path, use it directly
        if os.path.exists(base_model):
            logger.info(f"Using direct path: {base_model}")
            return base_model
        
        # Setup environment for HuggingFace download to network storage
        if params.get("download_to_network_storage", False):
            flux_models_dir = os.path.join(NETWORK_STORAGE_PATH, "flux-models")
            os.makedirs(flux_models_dir, exist_ok=True)
            os.environ["HF_HOME"] = flux_models_dir
            os.environ["TRANSFORMERS_CACHE"] = flux_models_dir
            logger.info(f"Set HuggingFace cache to network storage: {flux_models_dir}")
        
        # Ensure HF_TOKEN is available for diffusers
        if not os.getenv("HF_TOKEN") and os.getenv("HUGGINGFACE_TOKEN"):
            os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACE_TOKEN")
            logger.info("Set HF_TOKEN from HUGGINGFACE_TOKEN environment variable")
        
        # Fallback to HuggingFace Hub paths (will download if authenticated)
        model_paths = {
            "flux1-dev": "black-forest-labs/FLUX.1-dev",
            "flux1-schnell": "black-forest-labs/FLUX.1-schnell", 
            "flux1-dev2pro": "black-forest-labs/FLUX.1-dev",
        }
        
        hub_path = model_paths.get(base_model, base_model)
        logger.info(f"Will download from HuggingFace Hub: {hub_path}")
        logger.info(f"Cache directories checked: {potential_cache_paths}")
        return hub_path
    
    def _generate_sample_prompts(self, params: Dict[str, Any]) -> List[str]:
        """Generate sample prompts for DreamBooth training validation"""
        instance_prompt = params.get("instance_prompt", "")
        class_prompt = params.get("class_prompt", "")
        
        if params.get("sample_prompts"):
            # Use custom prompts if provided
            prompts = params["sample_prompts"]
            if instance_prompt:
                # Add instance prompt to prompts that don't have it
                prompts = [p if instance_prompt in p else f"{p}, {instance_prompt}" for p in prompts]
            return prompts
        
        # Default prompts for DreamBooth
        if instance_prompt:
            prompts = [
                instance_prompt,
                f"{instance_prompt}, high quality, detailed",
                f"{instance_prompt}, professional photography",
                f"{instance_prompt}, close-up portrait",
            ]
        elif class_prompt:
            prompts = [
                class_prompt,
                f"{class_prompt}, high quality",
                f"{class_prompt}, detailed",
            ]
        else:
            # Fallback prompts
            prompts = [
                "a portrait photo",
                "a professional headshot",
                "a close-up photo",
                "a full body photo",
            ]
        
        return prompts
    
    def run_training(self, config_path: str) -> Dict[str, Any]:
        """Execute FLUX DreamBooth training process"""
        try:
            # Read the command from the config file
            with open(config_path, 'r') as f:
                cmd_args = json.load(f)
            logger.info(f"Running FLUX DreamBooth command: {' '.join(cmd_args)}")
            
            # Validate that cmd_args is a list
            if not isinstance(cmd_args, list):
                raise ValueError(f"Expected list for command arguments, got {type(cmd_args)}")
            
            # Prepare environment with HuggingFace authentication
            env = os.environ.copy()
            
            # Ensure HuggingFace token is available
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
            if hf_token:
                env["HF_TOKEN"] = hf_token
                env["HUGGING_FACE_HUB_TOKEN"] = hf_token
                env["HF_HOME"] = CACHE_PATH
                logger.info("HuggingFace token configured for subprocess")
            else:
                logger.warning("No HuggingFace token found in environment")
            
            # Add device management environment variables
            env["CUDA_VISIBLE_DEVICES"] = "0"
            env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
            env["CUDA_LAUNCH_BLOCKING"] = "1"  # Force synchronous CUDA operations
            env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
            env["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads
            env["MKL_NUM_THREADS"] = "1"  # Limit MKL threads
            env["NUMEXPR_NUM_THREADS"] = "1"  # Limit NumExpr threads
            env["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism
            env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"  # Disable CUDA memory caching
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Consistent device ordering
            logger.info("Enhanced device management environment variables configured")
            
            # Test that the training script can be imported and basic validation
            try:
                # Check if script file exists
                script_path = "/workspace/dreambooth/train_dreambooth_flux.py"
                if os.path.exists(script_path):
                    logger.info(f"Training script found at: {script_path}")
                    logger.info(f"Script file size: {os.path.getsize(script_path)} bytes")
                else:
                    logger.error(f"Training script not found at: {script_path}")
                    # List contents of dreambooth directory
                    dreambooth_dir = "/workspace/dreambooth"
                    if os.path.exists(dreambooth_dir):
                        files = os.listdir(dreambooth_dir)
                        logger.info(f"Files in dreambooth directory: {files}")
                    else:
                        logger.error(f"Dreambooth directory not found: {dreambooth_dir}")
                
                test_result = subprocess.run(
                    ["python", "-c", "import sys; sys.path.append('/workspace/dreambooth'); import train_dreambooth_flux; print('Script import successful')"],
                    capture_output=True,
                    text=True,
                    env=env,
                    cwd="/workspace"
                )
                logger.info(f"Script import test: {test_result.stdout}")
                if test_result.stderr:
                    logger.warning(f"Script import warnings: {test_result.stderr}")
            except Exception as e:
                logger.warning(f"Script import test failed: {e}")

            # Run HuggingFace login first to ensure authentication
            if hf_token:
                login_cmd = [sys.executable, "-c", f"from huggingface_hub import login; login(token='{hf_token}')"]
                logger.info("Running HuggingFace login...")
                login_process = subprocess.run(
                    login_cmd,
                    capture_output=True,
                    text=True,
                    env=env
                )
                if login_process.returncode == 0:
                    logger.info("HuggingFace login successful")
                else:
                    logger.warning(f"HuggingFace login failed: {login_process.stderr}")
            
            # Run the training command
            try:
                logger.info(f"Starting training with command: {' '.join(cmd_args)}")
                
                # First, let's test if the script can be run directly without accelerate
                logger.info("Testing direct script execution...")
                test_cmd = ["python", "/workspace/dreambooth/train_dreambooth_flux.py", "--help"]
                test_result = subprocess.run(
                    test_cmd,
                    capture_output=True,
                    text=True,
                    env=env,
                    cwd="/workspace"
                )
                logger.info(f"Script help test stdout: {test_result.stdout[:500]}...")
                if test_result.stderr:
                    logger.warning(f"Script help test stderr: {test_result.stderr[:500]}...")
                
                # Now run the actual training command
                logger.info("Running accelerate command...")
                result = subprocess.run(
                    cmd_args,
                    capture_output=True,
                    text=True,
                    env=env,
                    cwd="/workspace",
                    timeout=300  # 5 minute timeout
                )
                
                # Log the full output for debugging
                logger.info(f"Accelerate command completed with return code: {result.returncode}")
                if result.stdout:
                    logger.info(f"Training stdout: {result.stdout}")
                if result.stderr:
                    logger.error(f"Training stderr: {result.stderr}")
                
                if result.returncode != 0:
                    logger.error(f"Training failed with exit code {result.returncode}")
                    logger.error(f"Command: {' '.join(cmd_args)}")
                    
                    # Try running without accelerate to see if that's the issue
                    logger.info("Trying to run script directly without accelerate...")
                    direct_cmd = ["python", "/workspace/dreambooth/train_dreambooth_flux.py"] + cmd_args[2:]  # Skip accelerate launch
                    logger.info(f"Direct command: {' '.join(direct_cmd)}")
                    
                    try:
                        direct_result = subprocess.run(
                            direct_cmd,
                            capture_output=True,
                            text=True,
                            env=env,
                            cwd="/workspace",
                            timeout=300  # 5 minute timeout
                        )
                        logger.info(f"Direct execution completed with return code: {direct_result.returncode}")
                        logger.info(f"Direct execution stdout: {direct_result.stdout}")
                        if direct_result.stderr:
                            logger.error(f"Direct execution stderr: {direct_result.stderr}")
                    except subprocess.TimeoutExpired:
                        logger.error("Direct execution timed out after 5 minutes")
                    except Exception as e:
                        logger.error(f"Direct execution failed with exception: {e}")
                    
                    raise subprocess.CalledProcessError(result.returncode, cmd_args)
                    
            except subprocess.CalledProcessError as e:
                logger.error(f"Training failed: {e}")
                raise
            except subprocess.TimeoutExpired:
                logger.error("Training command timed out after 5 minutes")
                raise
            except Exception as e:
                logger.error(f"Unexpected error during training: {e}")
                logger.error(f"Error type: {type(e)}")
                raise
            
            logger.info("FLUX DreamBooth training completed successfully")
            return {
                "status": "success",
                "output": result.stdout.splitlines()[-50:] if result.stdout else [],  # Last 50 lines
                "error": result.stderr
            }
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": str(type(e)),
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
    # Updated: 2025-01-08 - Enhanced HuggingFace token handling for FLUX training
    # Updated: 2025-01-08 - Fixed subprocess environment and added HF login
    # Updated: 2025-01-08 - Removed all scheduler configurations to let FLUX use internal scheduler
    # Updated: 2025-01-08 - Converted from LoRA to DreamBooth training
    # Updated: 2025-01-08 - Fixed to use official FLUX DreamBooth train_dreambooth_flux.py script
    # Updated: 2025-01-08 - Added FLUX DreamBooth script download and fixed script path
    # Updated: 2025-01-08 - Fixed prompt quoting and removed unsupported arguments
    # Updated: 2025-01-08 - Fixed command serialization to preserve prompt arguments properly
    try:
        job_input = job["input"]
        logger.info(f"Received job with input keys: {list(job_input.keys())}")
        
        # Set up HuggingFace token if provided
        if "huggingface_token" in job_input:
            hf_token = job_input["huggingface_token"]
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
            os.environ["HF_HOME"] = CACHE_PATH  # Set cache directory
            logger.info("HuggingFace token configured")
        elif os.getenv("HUGGINGFACE_TOKEN"):
            # Use container environment variable if available
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
            os.environ["HF_HOME"] = CACHE_PATH
            logger.info("Using HuggingFace token from container environment")
        
        # Initialize handler
        trainer = DreamBoothTrainingHandler()
        
        # Validate input
        params = trainer.validate_input(job_input)
        logger.info(f"Validated parameters for model: {params['model_name']}")
        
        # Use existing dataset folder directly
        dataset_path = os.path.join(DATASETS_PATH, params["model_name"])
        logger.info(f"Using existing dataset at: {dataset_path}")
        
        # Check if dataset folder exists
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset folder not found: {dataset_path}")
            return {
                "error": f"Dataset folder not found: {dataset_path}",
                "status": "failed"
            }
        
        # List contents of dataset folder for debugging
        try:
            items = os.listdir(dataset_path)
            logger.info(f"Found {len(items)} items in dataset folder: {items}")
            
            # Check for any directories
            directories = []
            files = []
            for item in items:
                item_path = os.path.join(dataset_path, item)
                if os.path.isdir(item_path):
                    directories.append(item)
                    # List contents of subdirectory
                    try:
                        sub_items = os.listdir(item_path)
                        logger.info(f"Subdirectory '{item}' contains: {sub_items}")
                    except Exception as e:
                        logger.error(f"Error listing subdirectory '{item}': {e}")
                else:
                    files.append(item)
            
            logger.info(f"Directories found: {directories}")
            logger.info(f"Files found: {files}")
            
            if directories:
                logger.error(f"Found directories in dataset folder: {directories}")
                logger.info("Attempting to clean up directories automatically...")
                
                # Clean up directories
                for directory in directories:
                    dir_path = os.path.join(dataset_path, directory)
                    try:
                        logger.info(f"Removing directory: {directory}")
                        shutil.rmtree(dir_path)
                        logger.info(f"Successfully removed directory: {directory}")
                    except Exception as e:
                        logger.error(f"Error removing directory {directory}: {e}")
                        return {
                            "error": f"Failed to remove directory {directory}: {e}",
                            "status": "failed"
                        }
                
                # Verify cleanup
                try:
                    final_items = os.listdir(dataset_path)
                    final_dirs = [item for item in final_items if os.path.isdir(os.path.join(dataset_path, item))]
                    if final_dirs:
                        logger.error(f"Still found directories after cleanup: {final_dirs}")
                        return {
                            "error": f"Failed to clean up all directories. Remaining: {final_dirs}",
                            "status": "failed"
                        }
                    else:
                        logger.info(f"Cleanup successful. Final contents: {final_items}")
                except Exception as e:
                    logger.error(f"Error verifying cleanup: {e}")
                    return {
                        "error": f"Error verifying cleanup: {e}",
                        "status": "failed"
                    }
                
        except Exception as e:
            logger.error(f"Error listing dataset folder: {e}")
            return {
                "error": f"Error accessing dataset folder: {e}",
                "status": "failed"
            }
        
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