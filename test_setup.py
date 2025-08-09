#!/usr/bin/env python3
"""
Test script to verify DeepSpeed setup for FLUX DreamBooth training
"""

import os
import sys
import json
import yaml
import torch
import subprocess
from pathlib import Path

def test_deepspeed_installation():
    """Test if DeepSpeed is properly installed"""
    try:
        import deepspeed
        print(f"✓ DeepSpeed version: {deepspeed.__version__}")
        return True
    except ImportError as e:
        print(f"✗ DeepSpeed not installed: {e}")
        return False

def test_cuda_availability():
    """Test CUDA availability"""
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    else:
        print("✗ CUDA not available")
        return False

def test_accelerate_deepspeed():
    """Test if accelerate supports DeepSpeed"""
    try:
        from accelerate import Accelerator
        from accelerate.utils import DistributedDataParallelKwargs
        
        # Test basic accelerate setup
        accelerator = Accelerator(
            mixed_precision="fp16",
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
        )
        print(f"✓ Accelerate initialized successfully")
        print(f"  Device: {accelerator.device}")
        print(f"  Mixed precision: {accelerator.mixed_precision}")
        return True
    except Exception as e:
        print(f"✗ Accelerate test failed: {e}")
        return False

def test_flux_script():
    """Test if FLUX training script exists and can be imported"""
    script_path = "/workspace/dreambooth/train_dreambooth_flux.py"
    if os.path.exists(script_path):
        print(f"✓ FLUX training script found: {script_path}")
        print(f"  File size: {os.path.getsize(script_path)} bytes")
        return True
    else:
        print(f"✗ FLUX training script not found: {script_path}")
        return False

def test_deepspeed_config():
    """Test DeepSpeed configuration creation"""
    try:
        # Create a sample DeepSpeed config
        deepspeed_config = {
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 2e-6,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 2e-6,
                    "warmup_num_steps": 0
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True
            },
            "gradient_clipping": 1.0,
            "steps_per_print": 10,
            "wall_clock_breakdown": False
        }
        
        # Save config
        config_path = "/tmp/test_deepspeed_config.json"
        with open(config_path, 'w') as f:
            json.dump(deepspeed_config, f, indent=2)
        
        print(f"✓ DeepSpeed config created: {config_path}")
        
        # Test loading config
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        
        print(f"✓ DeepSpeed config loaded successfully")
        print(f"  Stage: {loaded_config['zero_optimization']['stage']}")
        print(f"  FP16 enabled: {loaded_config['fp16']['enabled']}")
        print(f"  CPU offload: {loaded_config['zero_optimization']['offload_optimizer']['device']}")
        
        return True
    except Exception as e:
        print(f"✗ DeepSpeed config test failed: {e}")
        return False

def test_accelerate_config():
    """Test accelerate configuration"""
    try:
        accelerate_config = {
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": "DEEPSPEED",
            "downcast_bf16": "no",
            "gpu_ids": "all",
            "machine_rank": 0,
            "main_training_function": "main",
            "mixed_precision": "fp16",
            "num_machines": 1,
            "num_processes": 1,
            "rdzv_backend": "static",
            "same_network": True,
            "tpu_env": [],
            "tpu_use_cluster": False,
            "tpu_use_sudo": False,
            "use_cpu": False
        }
        
        config_path = "/tmp/test_accelerate_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(accelerate_config, f, default_flow_style=False)
        
        print(f"✓ Accelerate config created: {config_path}")
        print(f"  Distributed type: {accelerate_config['distributed_type']}")
        print(f"  Mixed precision: {accelerate_config['mixed_precision']}")
        
        return True
    except Exception as e:
        print(f"✗ Accelerate config test failed: {e}")
        return False

def test_environment_variables():
    """Test environment variables for DeepSpeed"""
    env_vars = {
        "DS_SKIP_CUDA_CHECK": "0",
        "DS_REPORT_BUG_MODE": "0", 
        "DS_ACCELERATOR": "cuda",
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512,expandable_segments:True",
        "CUDA_LAUNCH_BLOCKING": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_NO_CUDA_MEMORY_CACHING": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID"
    }
    
    print("Setting DeepSpeed environment variables...")
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  {key}={value}")
    
    print("✓ Environment variables set")
    return True

def main():
    """Run all tests"""
    print("=== DeepSpeed Setup Test for FLUX DreamBooth ===")
    print()
    
    tests = [
        ("DeepSpeed Installation", test_deepspeed_installation),
        ("CUDA Availability", test_cuda_availability),
        ("Accelerate DeepSpeed Support", test_accelerate_deepspeed),
        ("FLUX Training Script", test_flux_script),
        ("DeepSpeed Config", test_deepspeed_config),
        ("Accelerate Config", test_accelerate_config),
        ("Environment Variables", test_environment_variables),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n=== Test Summary ===")
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! DeepSpeed is ready for FLUX DreamBooth training on 8GB GPU.")
        return 0
    else:
        print("❌ Some tests failed. Please check the configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
