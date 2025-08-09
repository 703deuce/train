# DeepSpeed Configuration for FLUX DreamBooth Training

This guide explains how to use DeepSpeed for training FLUX DreamBooth models on 8GB GPUs with CPU offloading.

## Overview

DeepSpeed is a deep learning optimization library that enables training large models on limited GPU memory by offloading tensors to CPU or NVMe storage. This implementation specifically targets 8GB GPU training with FLUX DreamBooth.

## Key Features

- **ZeRO Stage 2**: Optimizes memory usage by partitioning optimizer states and gradients
- **CPU Offloading**: Offloads optimizer states and parameters to CPU memory
- **FP16 Mixed Precision**: Reduces memory usage while maintaining training quality
- **Gradient Checkpointing**: Further reduces memory usage by recomputing activations

## Memory Requirements

- **GPU VRAM**: ~6-8 GB (with DeepSpeed optimization)
- **CPU RAM**: ~25 GB (for offloaded tensors)
- **Storage**: ~10 GB for model weights and checkpoints

## Configuration Details

### DeepSpeed Configuration

The handler automatically creates a DeepSpeed configuration with the following settings:

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

### Accelerate Configuration

The handler also creates an Accelerate configuration:

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

## Usage

### Basic Training Request

```json
{
  "model_name": "my_dreambooth_model",
  "dataset": "base64_encoded_dataset_zip",
  "instance_prompt": "a photo of sks person",
  "class_prompt": "a photo of a person",
  "steps": 2000,
  "learning_rate": 2e-6,
  "batch_size": 1,
  "use_deepspeed": true
}
```

### Advanced Configuration

```json
{
  "model_name": "my_dreambooth_model",
  "dataset": "base64_encoded_dataset_zip",
  "instance_prompt": "a photo of sks person",
  "class_prompt": "a photo of a person",
  "steps": 2000,
  "learning_rate": 2e-6,
  "batch_size": 1,
  "gradient_accumulation_steps": 2,
  "resolution": "1024x1024",
  "with_prior_preservation": true,
  "prior_loss_weight": 1.0,
  "num_class_images": 50,
  "use_deepspeed": true,
  "mixed_precision": "fp16",
  "gradient_checkpointing": true
}
```

## Environment Variables

The handler automatically sets the following environment variables for optimal DeepSpeed performance:

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

## Testing DeepSpeed Configuration

Run the test script to verify DeepSpeed is properly configured:

```bash
python test_deepspeed.py
```

This will test:
- DeepSpeed installation
- Accelerate DeepSpeed support
- CUDA availability
- Configuration file creation
- Environment variable setup

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce batch size to 1
   - Increase gradient accumulation steps
   - Ensure CPU has sufficient RAM (25GB+)

2. **DeepSpeed Import Error**
   - Install DeepSpeed: `pip install deepspeed>=0.12.0`
   - Ensure CUDA toolkit matches PyTorch version

3. **Slow Training**
   - This is expected with CPU offloading
   - Consider using NVMe offloading if available
   - Monitor CPU and GPU utilization

4. **Configuration Errors**
   - Check that accelerate config is properly set
   - Verify DeepSpeed config JSON is valid
   - Ensure all required environment variables are set

### Performance Optimization

1. **CPU Memory**: Ensure sufficient RAM for offloaded tensors
2. **Storage**: Use fast storage (SSD/NVMe) for better I/O performance
3. **Batch Size**: Start with batch_size=1 and increase if memory allows
4. **Gradient Accumulation**: Use gradient_accumulation_steps to simulate larger batch sizes

## Comparison with Standard Training

| Aspect | Standard Training | DeepSpeed Training |
|--------|------------------|-------------------|
| GPU Memory | 16GB+ | 6-8GB |
| CPU Memory | 8GB | 25GB+ |
| Training Speed | Fast | Slower (due to offloading) |
| Model Size | Limited by GPU | Can train larger models |
| Cost | Higher GPU requirements | Lower GPU requirements |

## References

- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [Accelerate DeepSpeed Guide](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)
- [FLUX DreamBooth Training](https://github.com/black-forest-labs/FLUX)

## Notes

- DeepSpeed training is slower than standard training due to CPU offloading
- The trade-off is reduced GPU memory requirements
- This configuration is specifically optimized for 8GB GPUs
- For larger GPUs (16GB+), standard training may be more efficient 