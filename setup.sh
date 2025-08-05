#!/bin/bash

# AI-Toolkit LoRA Training API Setup Script
# This script prepares the environment for deployment

set -e  # Exit on any error

echo "🚀 Setting up AI-Toolkit LoRA Training API"
echo "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "✅ Docker found"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

echo "✅ Python found"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p datasets
mkdir -p outputs
mkdir -p configs
mkdir -p examples

# Create example dataset structure
mkdir -p examples/character_dataset
mkdir -p examples/style_dataset
mkdir -p examples/concept_dataset

echo "📁 Directory structure created"

# Create example dataset files
echo "📝 Creating example files..."

# Example character dataset
cat > examples/character_dataset/image1.txt << 'EOF'
a portrait photo of mychar, high quality, detailed face
EOF

cat > examples/character_dataset/image2.txt << 'EOF'
mychar smiling, professional photography, well lit
EOF

cat > examples/character_dataset/image3.txt << 'EOF'
a close-up of mychar, detailed facial features, sharp focus
EOF

# Example style dataset
cat > examples/style_dataset/artwork1.txt << 'EOF'
abstract painting in vibrant colors, artstyle
EOF

cat > examples/style_dataset/artwork2.txt << 'EOF'
digital art landscape, artstyle, detailed brushwork
EOF

cat > examples/style_dataset/artwork3.txt << 'EOF'
portrait painting in artstyle, expressive brushstrokes
EOF

# Example concept dataset
cat > examples/concept_dataset/object1.txt << 'EOF'
a detailed view of futuregadget, sleek design, metallic surface
EOF

cat > examples/concept_dataset/object2.txt << 'EOF'
futuregadget in use, functional design, modern technology
EOF

echo "✅ Example caption files created"

# Create placeholder images (1x1 transparent PNGs)
echo "📷 Creating placeholder images..."

# Base64 for 1x1 transparent PNG
PLACEHOLDER_PNG="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

for dir in examples/*/; do
    for i in {1..3}; do
        filename=$(basename "$dir")
        if [[ "$filename" == "character_dataset" ]]; then
            echo "$PLACEHOLDER_PNG" | base64 -d > "${dir}image${i}.jpg"
        elif [[ "$filename" == "style_dataset" ]]; then
            echo "$PLACEHOLDER_PNG" | base64 -d > "${dir}artwork${i}.jpg"
        elif [[ "$filename" == "concept_dataset" ]]; then
            echo "$PLACEHOLDER_PNG" | base64 -d > "${dir}object${i}.jpg"
        fi
    done
done

echo "✅ Placeholder images created"

# Create configuration examples
echo "📋 Creating configuration examples..."

cat > examples/character_config.json << 'EOF'
{
  "model_name": "my_character_lora",
  "trigger_word": "mychar",
  "base_model": "flux1-dev",
  "steps": 2000,
  "learning_rate": 0.0003,
  "network_dim": 32,
  "network_alpha": 32,
  "resolution": "1024x1024",
  "sample_prompts": [
    "a portrait of mychar",
    "mychar smiling",
    "mychar in professional attire",
    "a close-up photo of mychar"
  ]
}
EOF

cat > examples/style_config.json << 'EOF'
{
  "model_name": "art_style_lora",
  "trigger_word": "artstyle",
  "base_model": "flux1-dev",
  "steps": 1500,
  "learning_rate": 0.0001,
  "network_dim": 64,
  "network_alpha": 64,
  "resolution": "1024x1024",
  "only_if_contains": ["transformer.single_transformer_blocks"],
  "sample_prompts": [
    "abstract art in artstyle",
    "landscape painting, artstyle",
    "portrait in artstyle",
    "digital artwork, artstyle"
  ]
}
EOF

echo "✅ Configuration examples created"

# Make scripts executable
chmod +x deploy.py
chmod +x test_api.py
chmod +x example_client.py

echo "✅ Scripts made executable"

# Check for requirements
echo "📦 Checking Python requirements..."
if [ -f "requirements.txt" ]; then
    echo "Requirements file found. Install with:"
    echo "  pip install -r requirements.txt"
else
    echo "❌ requirements.txt not found"
fi

# Display next steps
echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Prepare your dataset:"
echo "   - Add images to examples/character_dataset/"
echo "   - Add corresponding .txt caption files"
echo "   - Replace placeholder images with real training data"
echo ""
echo "2. Build Docker image:"
echo "   python deploy.py --build --registry your-dockerhub-username"
echo ""
echo "3. Test locally (if you have GPU):"
echo "   python deploy.py --test"
echo ""
echo "4. Push to registry:"
echo "   python deploy.py --push --registry your-dockerhub-username"
echo ""
echo "5. Deploy to RunPod:"
echo "   - Create new Serverless template with your image"
echo "   - Set up endpoint with 24GB+ GPU"
echo "   - Test with: python test_api.py"
echo ""
echo "📚 Files created:"
echo "   📁 examples/character_dataset/ - Example character training data"
echo "   📁 examples/style_dataset/ - Example style training data"
echo "   📁 examples/concept_dataset/ - Example concept training data"
echo "   📄 examples/character_config.json - Character LoRA config"
echo "   📄 examples/style_config.json - Style LoRA config"
echo ""
echo "🔧 Available commands:"
echo "   python deploy.py --help - View deployment options"
echo "   python test_api.py - Test API functionality"
echo "   python example_client.py - Interactive training examples"
echo ""
echo "⚠️  Remember to:"
echo "   - Set your Docker registry username"
echo "   - Configure your HuggingFace token for model access"
echo "   - Use real training images (not the placeholders)"
echo ""
echo "✅ You're ready to start training LoRAs!"