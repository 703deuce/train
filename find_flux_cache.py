#!/usr/bin/env python3
"""
Debug script to find FLUX model cache location
"""

import os
from pathlib import Path

def find_flux_models():
    """Find FLUX model files in various cache locations"""
    
    print("🔍 Searching for FLUX model files...")
    
    # Potential cache locations
    search_paths = [
        "/runpod-volume/cache",
        "/runpod-volume",
        "/workspace/cache", 
        "/workspace",
        "~/.cache/huggingface/hub",
        "/root/.cache/huggingface/hub",
        "/tmp",
        "/models",
    ]
    
    # Model file patterns
    model_patterns = [
        "flux1-dev",
        "FLUX.1-dev", 
        "black-forest-labs",
        "flux",
        "transformer",
    ]
    
    found_models = []
    
    for base_path in search_paths:
        try:
            # Expand user path
            expanded_path = os.path.expanduser(base_path)
            if not os.path.exists(expanded_path):
                continue
                
            print(f"\n📂 Checking: {expanded_path}")
            
            # Walk through directory tree
            for root, dirs, files in os.walk(expanded_path):
                # Check if any files match model patterns
                model_files = [f for f in files if f.endswith(('.safetensors', '.bin', '.pt', '.ckpt'))]
                
                if model_files:
                    # Check if path contains FLUX-related keywords
                    root_lower = root.lower()
                    if any(pattern.lower() in root_lower for pattern in model_patterns):
                        print(f"  ✅ Found model files: {root}")
                        print(f"     Files: {model_files[:5]}{'...' if len(model_files) > 5 else ''}")
                        found_models.append({
                            "path": root,
                            "files": model_files,
                            "size_gb": sum(os.path.getsize(os.path.join(root, f)) for f in model_files) / (1024**3)
                        })
                
                # Also check for config.json files (indicates HF model)
                if "config.json" in files:
                    config_path = os.path.join(root, "config.json")
                    try:
                        with open(config_path, 'r') as f:
                            content = f.read()
                            if "flux" in content.lower() or "transformer" in content.lower():
                                print(f"  📄 Found HF config: {root}")
                                print(f"     Config: config.json")
                                if root not in [m["path"] for m in found_models]:
                                    found_models.append({
                                        "path": root,
                                        "files": files,
                                        "size_gb": sum(os.path.getsize(os.path.join(root, f)) for f in files) / (1024**3)
                                    })
                    except:
                        pass
                        
        except Exception as e:
            print(f"  ❌ Error accessing {expanded_path}: {e}")
    
    print(f"\n📊 Summary - Found {len(found_models)} potential FLUX model locations:")
    print("=" * 60)
    
    for i, model in enumerate(found_models, 1):
        print(f"{i}. {model['path']}")
        print(f"   Size: {model['size_gb']:.2f} GB")
        print(f"   Files: {len(model['files'])} files")
        print()
    
    # Try to detect the correct one
    if found_models:
        # Sort by size (FLUX should be several GB)
        large_models = [m for m in found_models if m['size_gb'] > 1.0]
        if large_models:
            best_match = max(large_models, key=lambda x: x['size_gb'])
            print(f"🎯 Best match (largest): {best_match['path']}")
            print(f"   Size: {best_match['size_gb']:.2f} GB")
            return best_match['path']
    
    return None

def test_path_in_container():
    """Test the path that would be used in the container"""
    
    cache_path = "/runpod-volume/cache"
    
    print(f"\n🧪 Testing container cache path: {cache_path}")
    
    if os.path.exists(cache_path):
        print("✅ Cache directory exists")
        
        # List contents
        try:
            contents = os.listdir(cache_path)
            print(f"📁 Contents ({len(contents)} items):")
            for item in contents[:10]:  # Show first 10 items
                item_path = os.path.join(cache_path, item)
                if os.path.isdir(item_path):
                    print(f"  📂 {item}/")
                else:
                    print(f"  📄 {item}")
            if len(contents) > 10:
                print(f"  ... and {len(contents) - 10} more items")
        except Exception as e:
            print(f"❌ Error listing contents: {e}")
    else:
        print("❌ Cache directory not found")

if __name__ == "__main__":
    print("🚀 FLUX Model Cache Finder")
    print("=" * 30)
    
    # Test container cache
    test_path_in_container()
    
    # Find all FLUX models
    best_path = find_flux_models()
    
    if best_path:
        print(f"\n✅ Recommended model path: {best_path}")
        print("\nTo use this path in your training, set:")
        print(f'  "base_model": "{best_path}"')
    else:
        print("\n❌ No FLUX models found in cache")
        print("You may need to download the model first or check the cache location")