# AI-Toolkit DreamBooth Training API for RunPod Serverless

This project provides a comprehensive API for DreamBooth fine-tuning using the [ai-toolkit](https://github.com/ostris/ai-toolkit) on RunPod Serverless infrastructure. It supports FLUX model training with extensive parameter customization.

## Features

- **FLUX DreamBooth Training**: Full support for FLUX.1-dev and FLUX.1-schnell models
- **Flexible Dataset Handling**: Support for various dataset formats (zip files, individual files)
- **Comprehensive Parameters**: All major DreamBooth training parameters exposed via API
- **Memory Optimization**: Built-in support for fp8, gradient checkpointing, and memory-efficient training
- **Real-time Monitoring**: Training progress logging and sample generation
- **Output Management**: Automatic collection and encoding of trained models and samples

## Quick Start

### 1. Deploy to RunPod

1. Build and push your Docker image:
```bash
docker build -t your-username/ai-toolkit-api:latest .
docker push your-username/ai-toolkit-api:latest
```

2. Create a new RunPod Serverless template with your image
3. Set up an endpoint with appropriate GPU resources (minimum 24GB VRAM recommended)

### 2. Basic API Usage

```python
import requests
import json
import base64

# Prepare your dataset (images + captions)
def encode_file(file_path):
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

# Example API call
payload = {
    "input": {
        "model_name": "my_character_dreambooth",
        "instance_prompt": "a photo of mychar",
        "class_prompt": "a photo of a person",
        "base_model": "flux1-dev",
        "steps": 2000,
        "learning_rate": 2e-6,
        "train_text_encoder": True,
        "with_prior_preservation": True,
        "prior_loss_weight": 1.0,
        "num_class_images": 50,
        "resolution": "1024x1024",
        "batch_size": 1,
        "dataset": [
            {
                "filename": "image1.jpg",
                "content": encode_file("path/to/image1.jpg")
            },
            {
                "filename": "image1.txt", 
                "content": base64.b64encode("a photo of mychar".encode()).decode()
            },
            # ... more files
        ]
    }
}

response = requests.post("YOUR_RUNPOD_ENDPOINT", json=payload)
result = response.json()
```

## API Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_name` | string | Unique name for your DreamBooth model |
| `dataset` | array/string | Training dataset (files or base64 zip) |

### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_model` | string | `"flux1-dev"` | Base model (`flux1-dev`, `flux1-schnell`) |
| `instance_prompt` | string | `""` | Instance prompt for your subject |
| `class_prompt` | string | `""` | Class prompt for prior preservation |
| `model_type` | string | `"flux"` | Model type (currently supports `flux`) |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `steps` | integer | `2000` | Number of training steps |
| `learning_rate` | float | `2e-6` | Learning rate for optimizer (DreamBooth uses lower rates) |
| `batch_size` | integer | `1` | Training batch size |
| `gradient_accumulation_steps` | integer | `1` | Steps to accumulate gradients |
| `resolution` | string | `"1024x1024"` | Training resolution (`"1024x1024"`, `"512x512"`) |
| `max_bucket_resolution` | integer | `2048` | Maximum bucket resolution |
| `min_bucket_resolution` | integer | `256` | Minimum bucket resolution |

### DreamBooth Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_text_encoder` | boolean | `true` | Train text encoder (required for DreamBooth) |
| `with_prior_preservation` | boolean | `true` | Use prior preservation to maintain general knowledge |
| `prior_loss_weight` | float | `1.0` | Weight for prior preservation loss |
| `num_class_images` | integer | `50` | Number of class images for prior preservation |

### Optimizer Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer` | string | `"AdamW8bit"` | Optimizer type |
| `lr_scheduler` | string | `"constant"` | Learning rate scheduler |
| `lr_warmup_steps` | integer | `0` | Warmup steps for learning rate |

### Memory Optimization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fp8_base` | boolean | `true` | Use FP8 precision for base model |
| `cache_latents` | boolean | `true` | Cache latents to save VRAM |
| `cache_text_encoder_outputs` | boolean | `true` | Cache text encoder outputs |
| `gradient_checkpointing` | boolean | `true` | Enable gradient checkpointing |
| `mixed_precision` | string | `"bf16"` | Mixed precision mode |

### Sampling & Validation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sample_every_n_steps` | integer | `200` | Generate samples every N steps |
| `sample_prompts` | array | `[]` | Custom prompts for sampling |
| `guidance_scale` | float | `3.5` | Guidance scale for sampling |
| `sample_steps` | integer | `20` | Inference steps for sampling |

### Advanced Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `noise_offset` | float | `0.05` | Noise offset for training |
| `min_snr_gamma` | float | `7` | Minimum SNR gamma |
| `seed` | integer | `42` | Random seed |

## Dataset Format

### Option 1: File Array
```json
{
  "dataset": [
    {
      "filename": "image1.jpg",
      "content": "base64_encoded_image_data"
    },
    {
      "filename": "image1.txt",
      "content": "base64_encoded_caption"
    }
  ]
}
```

### Option 2: Base64 Zip File
```json
{
  "dataset": "base64_encoded_zip_file_content"
}
```

## Example Responses

### Successful Training
```json
{
  "status": "completed",
  "model_name": "my_character_dreambooth",
  "training_steps": 2000,
  "outputs": {
    "models": [
      {
        "filename": "my_character_dreambooth.safetensors",
        "data": "base64_encoded_model_data",
        "size": 67108864
      }
    ],
    "samples": [
      {
        "filename": "sample_001.png",
        "data": "data:image/png;base64,encoded_image",
        "size": 1234567
      }
    ]
  }
}
```

### Error Response
```json
{
  "error": "Invalid dataset format: Missing required files",
  "status": "failed",
  "traceback": "Detailed error traceback..."
}
```

## Best Practices

### Dataset Preparation
- Use high-quality images (minimum 512x512)
- Include 10-50 training images for character DreamBooth
- Write descriptive captions including your instance prompt
- Use consistent naming: `image1.jpg` + `image1.txt`

### Training Parameters
- Start with `steps: 1000-2000` for most DreamBooth training
- Use `learning_rate: 2e-6` (lower than LoRA for stability)
- Keep `batch_size: 1` on 24GB VRAM
- Use `resolution: "1024x1024"` for FLUX training
- Enable `train_text_encoder: true` for full fine-tuning

### Memory Management
- Enable `fp8_base: true` for VRAM savings
- Use `gradient_checkpointing: true` 
- Set `cache_latents: true` for faster training
- Consider `blocks_to_swap` for very low VRAM

## GPU Requirements

| Model | Minimum VRAM | Recommended VRAM | Notes |
|-------|--------------|------------------|-------|
| FLUX.1-dev | 24GB | 40GB+ | DreamBooth requires more VRAM than LoRA |
| FLUX.1-schnell | 20GB | 24GB+ | Faster training variant |

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `batch_size`, enable `fp8_base`, increase `blocks_to_swap`
2. **Slow Training**: Enable `cache_latents` and `cache_text_encoder_outputs`
3. **Poor Results**: Increase training steps, improve dataset quality, adjust learning rate
4. **Dataset Errors**: Ensure images are .jpg/.png and captions are .txt files

### Monitoring Training

The API provides real-time training logs and generates sample images every N steps. Monitor the logs for:
- Loss values (should generally decrease)
- Memory usage warnings
- Sample quality improvements

## Advanced Usage

### Prior Preservation
DreamBooth uses prior preservation to maintain general knowledge while learning your specific subject. The `class_prompt` should describe the general category of your subject.

### Multi-Concept Training
Use different instance prompts for different subjects. DreamBooth can learn multiple concepts in a single training run.

### Style Transfer
DreamBooth can learn artistic styles by using style images with appropriate instance prompts.

## Support

For issues related to:
- **ai-toolkit**: Check the [original repository](https://github.com/ostris/ai-toolkit)
- **RunPod**: Check RunPod documentation and support
- **This API**: Open an issue in this repository

## License

This project follows the same license as ai-toolkit. Check the original repository for details.