#!/usr/bin/env python3
"""
Device Fix Wrapper for FLUX DreamBooth Training
Comprehensive PyTorch device mismatch prevention
"""

import sys
import os
import subprocess
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def patch_torch_operations():
    """Patch PyTorch operations to automatically handle device mismatches"""
    
    # Store original methods
    original_index_select = torch.Tensor.index_select
    original_gather = torch.Tensor.gather
    original_scatter = torch.Tensor.scatter
    original_scatter_add = torch.Tensor.scatter_add
    
    def safe_index_select(self, dim, index):
        """Patched index_select that moves index to tensor's device"""
        if index.device != self.device:
            index = index.to(self.device)
        return original_index_select(self, dim, index)
    
    def safe_gather(self, dim, index):
        """Patched gather that moves index to tensor's device"""
        if index.device != self.device:
            index = index.to(self.device)
        return original_gather(self, dim, index)
    
    def safe_scatter(self, dim, index, src):
        """Patched scatter that moves index and src to tensor's device"""
        if index.device != self.device:
            index = index.to(self.device)
        if src.device != self.device:
            src = src.to(self.device)
        return original_scatter(self, dim, index, src)
    
    def safe_scatter_add(self, dim, index, src):
        """Patched scatter_add that moves index and src to tensor's device"""
        if index.device != self.device:
            index = index.to(self.device)
        if src.device != self.device:
            src = src.to(self.device)
        return original_scatter_add(self, dim, index, src)
    
    # Apply patches
    torch.Tensor.index_select = safe_index_select
    torch.Tensor.gather = safe_gather
    torch.Tensor.scatter = safe_scatter
    torch.Tensor.scatter_add = safe_scatter_add
    
    logger.info("Patched PyTorch tensor operations for device safety")

def patch_dataloader():
    """Patch DataLoader to automatically move all batch tensors to CUDA"""
    
    original_iter = DataLoader.__iter__
    
    def safe_iter(self):
        """Patched DataLoader iterator that moves all tensors to CUDA"""
        iterator = original_iter(self)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for batch in iterator:
            yield move_batch_to_device(batch, device)
    
    def move_batch_to_device(batch, device):
        """Recursively move all tensors in a batch to the specified device"""
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, (list, tuple)):
            return type(batch)(move_batch_to_device(item, device) for item in batch)
        elif isinstance(batch, dict):
            return {key: move_batch_to_device(value, device) for key, value in batch.items()}
        else:
            return batch
    
    # Apply patch
    DataLoader.__iter__ = safe_iter
    logger.info("Patched DataLoader for automatic device movement")

def patch_collate_functions():
    """Patch common collate functions to ensure device consistency"""
    
    # Patch default_collate if it exists
    try:
        from torch.utils.data._utils.collate import default_collate
        original_default_collate = default_collate
        
        def safe_default_collate(batch):
            """Patched default_collate that ensures CUDA device"""
            result = original_default_collate(batch)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return move_batch_to_device(result, device)
        
        # Replace the function in the module
        import torch.utils.data._utils.collate
        torch.utils.data._utils.collate.default_collate = safe_default_collate
        logger.info("Patched default_collate for device safety")
        
    except ImportError:
        logger.warning("Could not patch default_collate - not available")

def setup_device_environment():
    """Setup optimal device environment for training"""
    
    # Set default tensor type to CUDA
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.empty_cache()
        device = torch.device("cuda")
        logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU")
    
    return device

def run_training_with_device_fix(training_script_path, *args):
    """Run training script with comprehensive device fixes applied"""
    
    # Setup device environment
    device = setup_device_environment()
    
    # Apply all patches
    patch_torch_operations()
    patch_dataloader()
    patch_collate_functions()
    
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

def main():
    """Main function to run training with device fixes"""
    
    if len(sys.argv) < 2:
        logger.error("Usage: python device_fix_wrapper.py <training_script> [args...]")
        sys.exit(1)
    
    training_script = sys.argv[1]
    training_args = sys.argv[2:]
    
    try:
        result = run_training_with_device_fix(training_script, *training_args)
        logger.info("Training completed successfully")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 