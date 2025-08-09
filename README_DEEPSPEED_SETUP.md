# DeepSpeed Setup for FLUX DreamBooth Training on 8GB GPU

This guide explains the complete setup for training FLUX DreamBooth models on 8GB GPUs using DeepSpeed for memory optimization.

## Overview

The setup includes:
- **DeepSpeed ZeRO Stage 2**: Optimizes memory usage by partitioning optimizer states and gradients
- **CPU Offloading**: Offloads optimizer states and parameters to CPU memory
- **FP16 Mixed Precision**: Reduces memory usage while maintaining training quality
- **Gradient Checkpointing**: Further reduces memory usage by recomputing activations

## Files Modified

### 1. `handler.py`
- Added DeepSpeed configuration creation
- Added Accelerate configuration setup
- Enhanced command generation with DeepSpeed support
- Added comprehensive logging for debugging

### 2. `Dockerfile`
- Updated to install DeepSpeed with CUDA support
- Added necessary system dependencies
- Configured for ai-toolkit integration
- Set up proper environment variables

### 3. `requirements.txt`
- Added DeepSpeed and related dependencies
- Updated PyTorch and other packages for compatibility

### 4. `dreambooth/requirements_flux.txt`
- Added DeepSpeed dependency for FLUX training

## DeepSpeed Configuration

The handler automatically creates a DeepSpeed configuration optimized for 8GB GPUs:

```json
{
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
  "fp16": {
    "enabled": true,
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
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  }
}
```

## Accelerate Configuration

The handler creates an Accelerate configuration for DeepSpeed:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
downcast_bf16: no
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

## Environment Variables

The handler sets these environment variables for optimal DeepSpeed performance:

```bash
DS_SKIP_CUDA_CHECK=0
DS_REPORT_BUG_MODE=0
DS_ACCELERATOR=cuda
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
CUDA_LAUNCH_BLOCKING=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false
PYTORCH_NO_CUDA_MEMORY_CACHING=1
CUDA_DEVICE_ORDER=PCI_BUS_ID
```

## Training Command

The handler generates a training command like this:

```bash
accelerate launch --config_file /workspace/configs/accelerate_config.yaml \
  /workspace/dreambooth/train_dreambooth_flux.py \
  --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev \
  --instance_data_dir /workspace/datasets/my_model \
  --output_dir /workspace/outputs/my_model \
  --instance_prompt "a photo of sks person" \
  --resolution 1024x1024 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-6 \
  --max_train_steps 2000 \
  --lr_scheduler constant \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --deepspeed /workspace/configs/deepspeed_config_my_model.json \
  --with_prior_preservation \
  --prior_loss_weight 1.0 \
  --num_class_images 50 \
  --class_prompt "a photo of a person"
```

## Memory Requirements

- **GPU VRAM**: ~6-8 GB (with DeepSpeed optimization)
- **CPU RAM**: ~25 GB (for offloaded tensors)
- **Storage**: ~10 GB for model weights and checkpoints

## Testing

Run the test script to verify the setup:

```bash
python test_setup.py
```

This will test:
- DeepSpeed installation
- CUDA availability
- Accelerate DeepSpeed support
- FLUX training script availability
- Configuration file creation
- Environment variable setup

## Usage Example

```json
{
  "model_name": "my_dreambooth_model",
  "dataset": "base64_encoded_dataset_zip",
  "instance_prompt": "a photo of sks person",
  "class_prompt": "a photo of a person",
  "steps": 2000,
  "learning_rate": 2e-6,
  "batch_size": 1,
  "gradient_accumulation_steps": 1,
  "resolution": "1024x1024",
  "with_prior_preservation": true,
  "prior_loss_weight": 1.0,
  "num_class_images": 50,
  "mixed_precision": "fp16",
  "gradient_checkpointing": true
}
```

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Ensure batch_size=1
   - Check that gradient_accumulation_steps is set
   - Verify CPU has sufficient RAM (25GB+)

2. **DeepSpeed Import Error**
   - DeepSpeed will be installed during Docker build
   - Check that CUDA toolkit matches PyTorch version

3. **Slow Training**
   - This is expected with CPU offloading
   - Monitor CPU and GPU utilization
   - Consider using NVMe offloading if available

4. **Configuration Errors**
   - Check that accelerate config is properly set
   - Verify DeepSpeed config JSON is valid
   - Ensure all required environment variables are set

### Performance Optimization

1. **CPU Memory**: Ensure sufficient RAM for offloaded tensors
2. **Storage**: Use fast storage (SSD/NVMe) for better I/O performance
3. **Batch Size**: Start with batch_size=1 and increase if memory allows
4. **Gradient Accumulation**: Use gradient_accumulation_steps to simulate larger batch sizes

## Deployment

1. **Build Docker Image**:
   ```bash
   docker build -t flux-dreambooth-deepspeed .
   ```

2. **Run Container**:
   ```bash
   docker run --gpus all -p 8000:8000 flux-dreambooth-deepspeed
   ```

3. **Deploy to RunPod**:
   - Push to GitHub
   - Configure RunPod serverless with the Docker image
   - Set environment variables for HuggingFace token

## Notes

- DeepSpeed training is slower than standard training due to CPU offloading
- The trade-off is reduced GPU memory requirements
- This configuration is specifically optimized for 8GB GPUs
- For larger GPUs (16GB+), standard training may be more efficient
- The setup automatically handles DeepSpeed configuration creation
- All memory optimizations are applied by default for 8GB GPU compatibility
