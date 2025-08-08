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
        "model_name": "my_character_lora",
        "trigger_word": "mychar",
        "base_model": "flux1-dev",
        "steps": 2000,
        "learning_rate": 0.0003,
        "network_dim": 16,
        "network_alpha": 16,
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
| `model_name` | string | Unique name for your LoRA model |
| `dataset` | array/string | Training dataset (files or base64 zip) |

### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_model` | string | `"flux1-dev"` | Base model (`flux1-dev`, `flux1-schnell`) |
| `trigger_word` | string | `""` | Trigger word for your LoRA |
| `model_type` | string | `"flux"` | Model type (currently supports `flux`) |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `steps` | integer | `2000` | Number of training steps |
| `learning_rate` | float | `0.0003` | Learning rate for optimizer |
| `batch_size` | integer | `1` | Training batch size |
| `gradient_accumulation_steps` | integer | `1` | Steps to accumulate gradients |
| `resolution` | string | `"1024x1024"` | Training resolution (`"1024x1024"`, `"512x512"`) |
| `max_bucket_resolution` | integer | `2048` | Maximum bucket resolution |
| `min_bucket_resolution` | integer | `256` | Minimum bucket resolution |

### LoRA Network Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `network_type` | string | `"lora"` | Network type (`lora`, `lokr`) |
| `network_dim` | integer | `16` | LoRA rank/dimension |
| `network_alpha` | integer | `16` | LoRA alpha parameter |
| `network_dropout` | float | `0.0` | Network dropout rate |

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
| `only_if_contains` | array | `[]` | Train only layers containing these strings |
| `ignore_if_contains` | array | `[]` | Ignore layers containing these strings |

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
  "model_name": "my_character_lora",
  "training_steps": 2000,
  "outputs": {
    "models": [
      {
        "filename": "my_character_lora.safetensors",
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
- Include 10-50 training images for character LoRAs
- Write descriptive captions including your trigger word
- Use consistent naming: `image1.jpg` + `image1.txt`

### Training Parameters
- Start with `steps: 1000-2000` for most LoRAs
- Use `network_dim: 16-64` (higher for complex subjects)
- Keep `batch_size: 1` on 24GB VRAM
- Use `resolution: "1024x1024"` for FLUX training

### Memory Management
- Enable `fp8_base: true` for VRAM savings
- Use `gradient_checkpointing: true` 
- Set `cache_latents: true` for faster training
- Consider `blocks_to_swap` for very low VRAM

## GPU Requirements

| Model | Minimum VRAM | Recommended VRAM | Notes |
|-------|--------------|------------------|-------|
| FLUX.1-dev | 24GB | 40GB+ | With memory optimizations |
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

### Custom Layer Training
```json
{
  "only_if_contains": [
    "transformer.single_transformer_blocks.7.proj_out",
    "transformer.single_transformer_blocks.20.proj_out"
  ]
}
```

### Multi-Concept Training
Use different trigger words in your captions and adjust the network dimension accordingly.

### Style Transfer LoRAs
Use artistic images without trigger words and focus on style consistency.

## Support

For issues related to:
- **ai-toolkit**: Check the [original repository](https://github.com/ostris/ai-toolkit)
- **RunPod**: Check RunPod documentation and support
- **This API**: Open an issue in this repository

## License

This project follows the same license as ai-toolkit. Check the original repository for details.