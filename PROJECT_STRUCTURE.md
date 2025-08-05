# AI-Toolkit LoRA Training API - Project Structure

This document outlines all the files in this project and their purposes.

## Core Files

### 🔧 `handler.py`
**Main RunPod serverless handler**
- Processes API requests for LoRA training
- Handles dataset uploads (zip files or individual files)
- Generates ai-toolkit configuration files
- Executes training and collects outputs
- Returns trained models and sample images

**Key Features:**
- Full FLUX LoRA parameter support
- Memory optimization settings
- Real-time training monitoring
- Automatic output collection and encoding

### 📦 `requirements.txt`
**Python dependencies**
- Core libraries: runpod, torch, transformers, diffusers
- Image processing: Pillow, opencv-python
- Memory optimization: bitsandbytes, xformers
- Configuration: PyYAML, safetensors

### 🐳 `Dockerfile`
**Container configuration for RunPod**
- NVIDIA CUDA 12.1 base image
- Python 3.10 environment
- ai-toolkit installation and setup
- All required dependencies
- Optimized for GPU training

## Deployment & Testing

### 🚀 `deploy.py`
**Deployment automation script**
- Build Docker images
- Push to container registries
- Test containers locally
- Generate RunPod templates
- Environment configuration

**Usage:**
```bash
python deploy.py --build --registry your-username
python deploy.py --push --template
```

### 🧪 `test_api.py`
**API testing and validation**
- Local handler testing
- API endpoint testing
- Parameter validation
- Example payload generation

**Usage:**
```bash
python test_api.py
```

### 📱 `example_client.py`
**Production-ready client examples**
- Character LoRA training
- Style LoRA training  
- Concept LoRA training
- Dataset preparation utilities
- Output management

**Usage:**
```bash
python example_client.py
```

### ⚙️ `setup.sh`
**Environment setup script** (Linux/Mac)
- Creates directory structure
- Generates example datasets
- Sets up configuration templates
- Makes scripts executable

**Usage:**
```bash
chmod +x setup.sh && ./setup.sh
```

## Documentation

### 📖 `README.md`
**Comprehensive documentation**
- API parameter reference
- Usage examples
- Best practices
- Troubleshooting guide
- GPU requirements

### 📋 `PROJECT_STRUCTURE.md` (this file)
**Project overview and file descriptions**

## Generated Files (After Setup)

### 📁 Directory Structure
```
ftrain/
├── handler.py              # Main API handler
├── requirements.txt         # Python dependencies
├── Dockerfile              # Container configuration
├── deploy.py               # Deployment scripts
├── test_api.py             # Testing utilities
├── example_client.py       # Client examples
├── setup.sh                # Setup script
├── README.md               # Documentation
├── PROJECT_STRUCTURE.md    # This file
├── datasets/               # Training datasets
├── outputs/                # Training outputs
├── configs/                # Generated configurations
└── examples/               # Example data
    ├── character_dataset/  # Character training example
    ├── style_dataset/      # Style training example
    ├── concept_dataset/    # Concept training example
    ├── character_config.json
    └── style_config.json
```

### 🔧 Runtime Generated Files
- `runpod_template.json` - RunPod template configuration
- `.env.example` - Environment variables template
- Training configs in `configs/` directory
- Model outputs in `outputs/` directory

## Usage Workflow

### 1. **Setup Environment**
```bash
# Linux/Mac
./setup.sh

# Windows (manual)
mkdir datasets outputs configs examples
```

### 2. **Prepare Dataset**
- Add images to dataset folder
- Create corresponding .txt caption files
- Include trigger word in captions

### 3. **Build & Deploy**
```bash
python deploy.py --build --registry your-username
python deploy.py --push
```

### 4. **Train LoRA**
```bash
# Using example client
python example_client.py

# Using direct API calls
python test_api.py
```

### 5. **Download Results**
- Models: `.safetensors` files
- Samples: Generated images during training
- Logs: Training progress and metrics

## API Endpoints

### POST `/run`
**Main training endpoint**

**Input:**
```json
{
  "input": {
    "model_name": "string",
    "dataset": "array|string",
    "trigger_word": "string",
    "steps": "integer",
    // ... additional parameters
  }
}
```

**Output:**
```json
{
  "status": "completed|failed",
  "model_name": "string",
  "outputs": {
    "models": [...],
    "samples": [...],
    "logs": [...]
  }
}
```

## Parameter Categories

### 🎯 **Training Parameters**
- `steps`, `learning_rate`, `batch_size`
- `resolution`, `gradient_accumulation_steps`

### 🧠 **LoRA Network**
- `network_dim`, `network_alpha`, `network_dropout`
- `only_if_contains`, `ignore_if_contains`

### 💾 **Memory Optimization**
- `fp8_base`, `gradient_checkpointing`
- `cache_latents`, `mixed_precision`

### 🎨 **Sampling**
- `sample_every_n_steps`, `sample_prompts`
- `guidance_scale`, `sample_steps`

## Best Practices

### 📊 **Dataset Quality**
- 10-50 high-quality images
- Consistent lighting and composition
- Descriptive captions with trigger words
- Proper file naming (image1.jpg + image1.txt)

### ⚡ **Performance**
- Use `fp8_base: true` for memory savings
- Enable `cache_latents` for speed
- Start with lower steps for testing
- Monitor sample outputs for quality

### 🔧 **Troubleshooting**
- Check GPU memory usage
- Validate dataset format
- Review training logs
- Adjust parameters based on results

## Support & Resources

- **ai-toolkit**: https://github.com/ostris/ai-toolkit
- **RunPod Docs**: https://docs.runpod.io/
- **FLUX Models**: https://huggingface.co/black-forest-labs
- **LoRA Guide**: See README.md for detailed documentation

## License

This project follows the same license terms as the original ai-toolkit project.