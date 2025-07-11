#!/usr/bin/env python3
"""
Test script for embedding models.

Verifies that embedding models are properly installed and can be loaded from local paths.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.utils.logger import get_logger

logger = get_logger(__name__)


def test_embedding_model(model_name: str, model_path: str) -> bool:
    """Test a single embedding model."""
    print(f"\n🔍 Testing {model_name}")
    print(f"📁 Model path: {model_path}")
    
    # Check if model directory exists
    if not os.path.exists(model_path):
        print(f"❌ Model directory not found: {model_path}")
        return False
        
    # Check for required files
    required_files = [
        "config.json",
        "tokenizer.json", 
        "tokenizer_config.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            missing_files.append(file)
            
    if missing_files:
        print(f"⚠️  Missing files: {missing_files}")
        print("   Model may still work if pytorch_model.bin or model.safetensors exists")
    
    try:
        # Test model loading
        print("📦 Loading model...")
        start_time = time.time()
        
        model = SentenceTransformer(
            model_path,
            device='cpu',  # Use CPU for testing
            local_files_only=True
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f}s")
        
        # Test embedding generation
        print("🧠 Testing embedding generation...")
        test_sentences = [
            "This is a test sentence for embedding generation.",
            "Another test sentence to verify model functionality.",
            "Machine learning models require proper testing."
        ]
        
        start_time = time.time()
        embeddings = model.encode(test_sentences, show_progress_bar=False)
        embed_time = time.time() - start_time
        
        # Verify embeddings
        assert isinstance(embeddings, np.ndarray), "Embeddings should be numpy array"
        assert embeddings.shape[0] == len(test_sentences), "Should have one embedding per sentence"
        assert len(embeddings.shape) == 2, "Should be 2D array"
        
        dimensions = embeddings.shape[1]
        print(f"✅ Generated {len(test_sentences)} embeddings in {embed_time:.2f}s")
        print(f"📏 Embedding dimensions: {dimensions}")
        print(f"📊 Embedding shape: {embeddings.shape}")
        
        # Test similarity calculation
        similarity = np.dot(embeddings[0], embeddings[1])
        print(f"🔗 Sample similarity: {similarity:.4f}")
        
        # Memory cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False


def test_all_embedding_models() -> bool:
    """Test all configured embedding models."""
    print("🚀 Testing Embedding Models")
    print("=" * 50)
    
    # Model configurations
    models_to_test = [
        {
            "name": "intfloat/e5-large-v2",
            "path": "./models/embeddings/e5-large-v2",
            "type": "Primary embedding model"
        },
        {
            "name": "sentence-transformers/all-MiniLM-L12-v2", 
            "path": "./models/embeddings/all-MiniLM-L12-v2",
            "type": "Fallback embedding model"
        }
    ]
    
    all_passed = True
    results = []
    
    for model_config in models_to_test:
        print(f"\n📋 {model_config['type']}")
        success = test_embedding_model(model_config["name"], model_config["path"])
        results.append({
            "name": model_config["name"],
            "type": model_config["type"],
            "success": success
        })
        
        if not success:
            all_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{status} {result['type']}: {result['name']}")
    
    if all_passed:
        print("\n🎉 All embedding models are working correctly!")
        print("\n💡 Next steps:")
        print("   - Models are ready for use")
        print("   - You can start the Tyra MCP server")
    else:
        print("\n⚠️  Some models failed tests!")
        print("\n🔧 Troubleshooting:")
        print("   - Ensure all models are downloaded using:")
        print("     huggingface-cli download <model_name> --local-dir <path>")
        print("   - Check internet connection for downloads")
        print("   - Verify disk space and permissions")
    
    return all_passed


def main():
    """Main test function."""
    try:
        success = test_all_embedding_models()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()