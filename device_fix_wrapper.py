#!/usr/bin/env python3
"""
Device Fix Wrapper for FLUX DreamBooth Training
Aggressive GPU-only tensor management - ALL tensors on GPU, NONE on CPU
"""

import sys
import os
import subprocess
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_gpu_only_environment():
    """Setup environment to force ALL operations to GPU"""
    
    # Set environment variables to force GPU usage
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
    
    logger.info("Set GPU-only environment variables")
    
    # Import torch after setting environment
    try:
        import torch
        
        # Force CUDA as default device
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            torch.cuda.empty_cache()
            
            device = torch.device("cuda")
            logger.info(f"GPU-only environment: {torch.cuda.get_device_name()}")
            logger.info("ALL tensors will be forced to GPU")
            return device
        else:
            logger.error("CUDA not available - cannot run GPU-only training")
            sys.exit(1)
            
    except ImportError as e:
        logger.error(f"Failed to import torch: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error setting up GPU environment: {e}")
        sys.exit(1)

def apply_pytorch_patches():
    """Apply all PyTorch patches for GPU-only operation"""
    
    try:
        import torch
        from torch.utils.data import DataLoader
        
        logger.info("Applying PyTorch patches for GPU-only operation...")
        
        # Store original methods
        original_index_select = torch.Tensor.index_select
        original_gather = torch.Tensor.gather
        original_scatter = torch.Tensor.scatter
        original_scatter_add = torch.Tensor.scatter_add
        original_cat = torch.cat
        original_stack = torch.stack
        original_zeros = torch.zeros
        original_ones = torch.ones
        original_randn = torch.randn
        original_rand = torch.rand
        original_arange = torch.arange
        original_tensor = torch.tensor
        
        def force_gpu_tensor(tensor):
            """Force any tensor to GPU, regardless of current device"""
            if hasattr(tensor, 'device') and tensor.device.type != 'cuda':
                return tensor.cuda()
            return tensor
        
        def safe_index_select(self, dim, index):
            """Force index to GPU and ensure result is on GPU"""
            index = force_gpu_tensor(index)
            result = original_index_select(self, dim, index)
            return force_gpu_tensor(result)
        
        def safe_gather(self, dim, index):
            """Force index to GPU and ensure result is on GPU"""
            index = force_gpu_tensor(index)
            result = original_gather(self, dim, index)
            return force_gpu_tensor(result)
        
        def safe_scatter(self, dim, index, src):
            """Force all tensors to GPU"""
            index = force_gpu_tensor(index)
            src = force_gpu_tensor(src)
            result = original_scatter(self, dim, index, src)
            return force_gpu_tensor(result)
        
        def safe_scatter_add(self, dim, index, src):
            """Force all tensors to GPU"""
            index = force_gpu_tensor(index)
            src = force_gpu_tensor(src)
            result = original_scatter_add(self, dim, index, src)
            return force_gpu_tensor(result)
        
        def safe_cat(tensors, dim=0, out=None):
            """Force all tensors to GPU before concatenation"""
            tensors = [force_gpu_tensor(t) for t in tensors]
            result = original_cat(tensors, dim, out)
            return force_gpu_tensor(result)
        
        def safe_stack(tensors, dim=0, out=None):
            """Force all tensors to GPU before stacking"""
            tensors = [force_gpu_tensor(t) for t in tensors]
            result = original_stack(tensors, dim, out)
            return force_gpu_tensor(result)
        
        def safe_zeros(*args, **kwargs):
            """Always create zeros tensor on GPU"""
            kwargs['device'] = 'cuda'
            return original_zeros(*args, **kwargs)
        
        def safe_ones(*args, **kwargs):
            """Always create ones tensor on GPU"""
            kwargs['device'] = 'cuda'
            return original_ones(*args, **kwargs)
        
        def safe_randn(*args, **kwargs):
            """Always create random tensor on GPU"""
            kwargs['device'] = 'cuda'
            return original_randn(*args, **kwargs)
        
        def safe_rand(*args, **kwargs):
            """Always create random tensor on GPU"""
            kwargs['device'] = 'cuda'
            return original_rand(*args, **kwargs)
        
        def safe_arange(*args, **kwargs):
            """Always create arange tensor on GPU"""
            kwargs['device'] = 'cuda'
            return original_arange(*args, **kwargs)
        
        def safe_tensor(data, **kwargs):
            """Always create tensor on GPU"""
            kwargs['device'] = 'cuda'
            return original_tensor(data, **kwargs)
        
        # Apply patches
        torch.Tensor.index_select = safe_index_select
        torch.Tensor.gather = safe_gather
        torch.Tensor.scatter = safe_scatter
        torch.Tensor.scatter_add = safe_scatter_add
        torch.cat = safe_cat
        torch.stack = safe_stack
        torch.zeros = safe_zeros
        torch.ones = safe_ones
        torch.randn = safe_randn
        torch.rand = safe_rand
        torch.arange = safe_arange
        torch.tensor = safe_tensor
        
        # Patch DataLoader
        original_iter = DataLoader.__iter__
        
        def force_gpu_iter(self):
            """Force ALL tensors in every batch to GPU"""
            iterator = original_iter(self)
            
            for batch in iterator:
                yield force_batch_to_gpu(batch)
        
        def force_batch_to_gpu(batch):
            """Recursively force ALL tensors in batch to GPU"""
            if isinstance(batch, torch.Tensor):
                return batch.cuda() if batch.device.type != 'cuda' else batch
            elif isinstance(batch, (list, tuple)):
                return type(batch)(force_batch_to_gpu(item) for item in batch)
            elif isinstance(batch, dict):
                return {key: force_batch_to_gpu(value) for key, value in batch.items()}
            else:
                return batch
        
        DataLoader.__iter__ = force_gpu_iter
        
        # Patch collate functions
        try:
            from torch.utils.data._utils.collate import default_collate
            original_default_collate = default_collate
            
            def force_gpu_collate(batch):
                """Force collated result to GPU"""
                result = original_default_collate(batch)
                return force_batch_to_gpu(result)
            
            import torch.utils.data._utils.collate
            torch.utils.data._utils.collate.default_collate = force_gpu_collate
            
        except ImportError:
            logger.warning("Could not patch default_collate - not available")
        
        logger.info("Successfully applied all PyTorch patches for GPU-only operation")
        
    except Exception as e:
        logger.error(f"Failed to apply PyTorch patches: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

def run_training_with_gpu_only_fix(training_script_path, *args):
    """Run training script with aggressive GPU-only tensor management"""
    
    try:
        # Setup GPU-only environment first
        device = setup_gpu_only_environment()
        
        # Apply all aggressive patches
        apply_pytorch_patches()
        
        # Build the command to run the training script
        cmd = [sys.executable, training_script_path] + list(args)
        logger.info(f"Running training script with GPU-only enforcement: {training_script_path}")
        logger.info(f"Command: {' '.join(cmd)}")
        
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
        
        logger.info("Training completed successfully with GPU-only tensors")
        return {
            "status": "success",
            "output": output_lines[-50:] if len(output_lines) > 50 else output_lines
        }
            
    except Exception as e:
        logger.error(f"Error running training script: {e}")
        logger.error(traceback.format_exc())
        raise

def main():
    """Main function to run training with aggressive GPU-only enforcement"""
    
    if len(sys.argv) < 2:
        logger.error("Usage: python device_fix_wrapper.py <training_script> [args...]")
        sys.exit(1)
    
    training_script = sys.argv[1]
    training_args = sys.argv[2:]
    
    try:
        result = run_training_with_gpu_only_fix(training_script, *training_args)
        logger.info("Training completed successfully with GPU-only enforcement")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 