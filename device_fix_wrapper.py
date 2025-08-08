#!/usr/bin/env python3
"""
Device Fix Wrapper for FLUX DreamBooth Training
Automatically patches PyTorch operations to prevent device mismatch errors
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def patch_torch_operations():
    """Patch PyTorch operations to automatically handle device placement"""
    
    # Store original functions
    original_index_select = torch.Tensor.index_select
    original_gather = torch.Tensor.gather
    original_scatter = torch.Tensor.scatter
    original_scatter_add = torch.Tensor.scatter_add
    
    def safe_index_select(self, dim, index):
        """Safe index_select that ensures tensors are on the same device"""
        if hasattr(self, 'device') and hasattr(index, 'device'):
            if self.device != index.device:
                logger.warning(f"Device mismatch in index_select: tensor on {self.device}, index on {index.device}")
                index = index.to(self.device)
        return original_index_select(self, dim, index)
    
    def safe_gather(self, dim, index):
        """Safe gather that ensures tensors are on the same device"""
        if hasattr(self, 'device') and hasattr(index, 'device'):
            if self.device != index.device:
                logger.warning(f"Device mismatch in gather: tensor on {self.device}, index on {index.device}")
                index = index.to(self.device)
        return original_gather(self, dim, index)
    
    def safe_scatter(self, dim, index, src):
        """Safe scatter that ensures tensors are on the same device"""
        target_device = self.device if hasattr(self, 'device') else torch.device('cuda:0')
        if hasattr(index, 'device') and index.device != target_device:
            logger.warning(f"Device mismatch in scatter: index on {index.device}, target on {target_device}")
            index = index.to(target_device)
        if hasattr(src, 'device') and src.device != target_device:
            logger.warning(f"Device mismatch in scatter: src on {src.device}, target on {target_device}")
            src = src.to(target_device)
        return original_scatter(self, dim, index, src)
    
    def safe_scatter_add(self, dim, index, src):
        """Safe scatter_add that ensures tensors are on the same device"""
        target_device = self.device if hasattr(self, 'device') else torch.device('cuda:0')
        if hasattr(index, 'device') and index.device != target_device:
            logger.warning(f"Device mismatch in scatter_add: index on {index.device}, target on {target_device}")
            index = index.to(target_device)
        if hasattr(src, 'device') and src.device != target_device:
            logger.warning(f"Device mismatch in scatter_add: src on {src.device}, target on {target_device}")
            src = src.to(target_device)
        return original_scatter_add(self, dim, index, src)
    
    # Apply patches
    torch.Tensor.index_select = safe_index_select
    torch.Tensor.gather = safe_gather
    torch.Tensor.scatter = safe_scatter
    torch.Tensor.scatter_add = safe_scatter_add
    
    logger.info("Applied PyTorch operation patches for device safety")

def patch_dataloader():
    """Patch DataLoader to automatically move tensors to device"""
    
    original_iter = DataLoader.__iter__
    
    def safe_iter(self):
        """Safe DataLoader iterator that moves tensors to device"""
        iterator = original_iter(self)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        for batch in iterator:
            if isinstance(batch, torch.Tensor):
                if batch.device != device:
                    batch = batch.to(device)
            elif isinstance(batch, (list, tuple)):
                batch = [item.to(device) if isinstance(item, torch.Tensor) and item.device != device else item for item in batch]
            elif isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) and v.device != device else v for k, v in batch.items()}
            yield batch
    
    DataLoader.__iter__ = safe_iter
    logger.info("Applied DataLoader patch for automatic device placement")

def setup_device_environment():
    """Setup device environment and patches"""
    
    # Set device
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache")
    
    # Apply patches
    patch_torch_operations()
    patch_dataloader()
    
    # Set default tensor type to CUDA if available
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        logger.info("Set default tensor type to CUDA")
    
    return device

def run_training_with_device_fix(training_script_path, *args):
    """Run training script with device fixes applied"""
    
    # Setup device environment
    device = setup_device_environment()
    
    # Build the command to run the training script
    cmd = [sys.executable, training_script_path] + list(args)
    logger.info(f"Running training script: {training_script_path}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the training script as a subprocess
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
        
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
        
        logger.info("Training completed successfully")
        return {
            "status": "success",
            "output": output_lines[-50:] if len(output_lines) > 50 else output_lines
        }
            
    except Exception as e:
        logger.error(f"Error running training script: {e}")
        raise

if __name__ == "__main__":
    # This will be called by the handler
    if len(sys.argv) < 2:
        print("Usage: python device_fix_wrapper.py <training_script> [args...]")
        sys.exit(1)
    
    training_script = sys.argv[1]
    training_args = sys.argv[2:]
    
    try:
        result = run_training_with_device_fix(training_script, *training_args)
        if result["status"] == "success":
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 