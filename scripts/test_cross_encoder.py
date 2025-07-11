#!/usr/bin/env python3
"""
Test script for cross-encoder models.

Verifies that cross-encoder models are properly installed and can be loaded from local paths.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.utils.logger import get_logger

logger = get_logger(__name__)


def test_cross_encoder_model(model_name: str, model_path: str) -> bool:
    """Test a single cross-encoder model."""
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
        print("📦 Loading cross-encoder model...")
        start_time = time.time()
        
        model = CrossEncoder(
            model_path,
            device='cpu',  # Use CPU for testing
            local_files_only=True
        )
        
        load_time = time.time() - start_time
        print(f"✅ Cross-encoder loaded successfully in {load_time:.2f}s")
        
        # Test reranking
        print("🔄 Testing reranking functionality...")
        
        # Test query-document pairs
        query = "What is machine learning?"
        documents = [
            "Machine learning is a subset of artificial intelligence (AI) that focuses on algorithms.",
            "The weather today is sunny and warm with clear skies.",
            "Deep learning uses neural networks with multiple layers to process data.",
            "Pizza is a popular Italian dish with cheese and toppings.",
            "Natural language processing helps computers understand human language."
        ]
        
        # Create query-document pairs for scoring
        pairs = [[query, doc] for doc in documents]
        
        start_time = time.time()
        scores = model.predict(pairs, show_progress_bar=False)
        score_time = time.time() - start_time
        
        # Verify scores
        assert len(scores) == len(documents), "Should have one score per document"
        assert all(isinstance(s, (int, float, np.number)) for s in scores), "Scores should be numeric"
        
        print(f"✅ Generated {len(scores)} relevance scores in {score_time:.2f}s")
        
        # Display results
        print("\n📊 Relevance Scores:")
        sorted_results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        
        for i, (doc, score) in enumerate(sorted_results[:3]):  # Top 3
            print(f"  {i+1}. Score: {score:.4f} - {doc[:60]}...")
        
        # Test single pair scoring
        print("\n🎯 Testing single pair scoring...")
        single_score = model.predict([[query, documents[0]]], show_progress_bar=False)
        print(f"✅ Single pair score: {single_score[0]:.4f}")
        
        # Memory cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return True
        
    except Exception as e:
        print(f"❌ Cross-encoder test failed: {e}")
        return False


def test_all_cross_encoders() -> bool:
    """Test all configured cross-encoder models."""
    print("🚀 Testing Cross-Encoder Models")
    print("=" * 50)
    
    # Model configurations
    models_to_test = [
        {
            "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "path": "./models/cross-encoders/ms-marco-MiniLM-L-6-v2",
            "type": "Primary cross-encoder"
        }
    ]
    
    # Add optional models if they exist
    optional_models = [
        {
            "name": "cross-encoder/stsb-roberta-base",
            "path": "./models/cross-encoders/stsb-roberta-base", 
            "type": "Alternative cross-encoder"
        }
    ]
    
    for model in optional_models:
        if os.path.exists(model["path"]):
            models_to_test.append(model)
    
    all_passed = True
    results = []
    
    for model_config in models_to_test:
        print(f"\n📋 {model_config['type']}")
        success = test_cross_encoder_model(model_config["name"], model_config["path"])
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
        print("\n🎉 All cross-encoder models are working correctly!")
        print("\n💡 Next steps:")
        print("   - Cross-encoders are ready for reranking")
        print("   - You can start the Tyra MCP server")
    else:
        print("\n⚠️  Some cross-encoders failed tests!")
        print("\n🔧 Troubleshooting:")
        print("   - Ensure all models are downloaded using:")
        print("     huggingface-cli download <model_name> --local-dir <path>")
        print("   - Check internet connection for downloads")
        print("   - Verify disk space and permissions")
    
    return all_passed


def main():
    """Main test function."""
    try:
        success = test_all_cross_encoders()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()