#!/usr/bin/env python3
"""
Test script for complete model pipeline.

Tests the integration of embedding models and cross-encoders in a complete RAG pipeline.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.utils.logger import get_logger

logger = get_logger(__name__)


def test_complete_pipeline() -> bool:
    """Test the complete embedding + reranking pipeline."""
    print("🚀 Testing Complete Model Pipeline")
    print("=" * 50)
    
    try:
        # Test data
        query = "How does artificial intelligence work?"
        documents = [
            "Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans.",
            "The weather forecast shows rain is expected tomorrow with temperatures around 20 degrees Celsius.",
            "Machine learning is a subset of AI that enables systems to automatically learn and improve from experience without being explicitly programmed.",
            "Pizza recipes often include ingredients like flour, tomatoes, cheese, and various toppings depending on preferences.",
            "Neural networks are computing systems inspired by biological neural networks that constitute animal brains.",
            "Basketball is a team sport played on a rectangular court with two teams of five players each.",
            "Deep learning uses artificial neural networks with multiple layers to model and understand complex patterns in data.",
            "Cooking techniques vary widely across different cultures and can significantly impact the flavor of food."
        ]
        
        print(f"📝 Query: {query}")
        print(f"📚 Testing with {len(documents)} documents")
        
        # Step 1: Load embedding model
        print("\n1️⃣ Loading embedding model...")
        embedding_path = "./models/embeddings/e5-large-v2"
        
        if not os.path.exists(embedding_path):
            print(f"❌ Primary embedding model not found, trying fallback...")
            embedding_path = "./models/embeddings/all-MiniLM-L12-v2"
            
        if not os.path.exists(embedding_path):
            print(f"❌ No embedding models found!")
            return False
            
        start_time = time.time()
        embedding_model = SentenceTransformer(
            embedding_path,
            device='cpu',
            local_files_only=True
        )
        embed_load_time = time.time() - start_time
        print(f"✅ Embedding model loaded in {embed_load_time:.2f}s")
        
        # Step 2: Generate embeddings
        print("\n2️⃣ Generating embeddings...")
        start_time = time.time()
        
        query_embedding = embedding_model.encode([query])[0]
        doc_embeddings = embedding_model.encode(documents)
        
        embed_time = time.time() - start_time
        print(f"✅ Generated embeddings in {embed_time:.2f}s")
        print(f"📏 Query embedding shape: {query_embedding.shape}")
        print(f"📏 Document embeddings shape: {doc_embeddings.shape}")
        
        # Step 3: Vector similarity search
        print("\n3️⃣ Computing similarity scores...")
        start_time = time.time()
        
        # Compute cosine similarity
        similarities = np.dot(doc_embeddings, query_embedding) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top candidates for reranking
        top_k = 5
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_docs = [documents[i] for i in top_indices]
        top_similarities = [similarities[i] for i in top_indices]
        
        similarity_time = time.time() - start_time
        print(f"✅ Computed similarities in {similarity_time:.2f}s")
        print(f"🔝 Selected top {top_k} candidates for reranking")
        
        # Display top candidates
        print("\n📊 Top candidates by similarity:")
        for i, (doc, sim) in enumerate(zip(top_docs, top_similarities)):
            print(f"  {i+1}. Score: {sim:.4f} - {doc[:60]}...")
        
        # Step 4: Load cross-encoder
        print("\n4️⃣ Loading cross-encoder for reranking...")
        ce_path = "./models/cross-encoders/ms-marco-MiniLM-L-6-v2"
        
        if not os.path.exists(ce_path):
            print(f"❌ Cross-encoder model not found at {ce_path}")
            return False
            
        start_time = time.time()
        cross_encoder = CrossEncoder(
            ce_path,
            device='cpu',
            local_files_only=True
        )
        ce_load_time = time.time() - start_time
        print(f"✅ Cross-encoder loaded in {ce_load_time:.2f}s")
        
        # Step 5: Rerank top candidates
        print("\n5️⃣ Reranking top candidates...")
        start_time = time.time()
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in top_docs]
        rerank_scores = cross_encoder.predict(pairs, show_progress_bar=False)
        
        rerank_time = time.time() - start_time
        print(f"✅ Reranked {len(pairs)} pairs in {rerank_time:.2f}s")
        
        # Sort by rerank score
        reranked_results = sorted(
            zip(top_docs, top_similarities, rerank_scores),
            key=lambda x: x[2],
            reverse=True
        )
        
        # Display final results
        print("\n🏆 Final Reranked Results:")
        for i, (doc, sim_score, rerank_score) in enumerate(reranked_results):
            print(f"  {i+1}. Rerank: {rerank_score:.4f} | Similarity: {sim_score:.4f}")
            print(f"     {doc[:80]}...")
            print()
        
        # Step 6: Performance summary
        total_time = embed_load_time + embed_time + similarity_time + ce_load_time + rerank_time
        
        print("⏱️  Performance Summary:")
        print(f"   • Embedding model load: {embed_load_time:.2f}s")
        print(f"   • Embedding generation: {embed_time:.2f}s")
        print(f"   • Similarity computation: {similarity_time:.2f}s")
        print(f"   • Cross-encoder load: {ce_load_time:.2f}s")
        print(f"   • Reranking: {rerank_time:.2f}s")
        print(f"   • Total pipeline time: {total_time:.2f}s")
        
        # Memory cleanup
        del embedding_model, cross_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False


def test_model_compatibility() -> bool:
    """Test model compatibility and configurations."""
    print("\n🔧 Testing Model Compatibility")
    print("=" * 30)
    
    try:
        # Check model directories
        models_to_check = [
            ("Primary Embedding", "./models/embeddings/e5-large-v2"),
            ("Fallback Embedding", "./models/embeddings/all-MiniLM-L12-v2"),
            ("Cross-Encoder", "./models/cross-encoders/ms-marco-MiniLM-L-6-v2")
        ]
        
        all_models_present = True
        
        for model_name, model_path in models_to_check:
            if os.path.exists(model_path):
                print(f"✅ {model_name}: Found at {model_path}")
            else:
                print(f"❌ {model_name}: Missing at {model_path}")
                all_models_present = False
        
        if not all_models_present:
            print("\n⚠️  Some models are missing!")
            print("   Run the download commands in INSTALLATION.md")
            return False
        
        # Check CUDA availability
        print(f"\n💻 System Information:")
        print(f"   • PyTorch version: {torch.__version__}")
        print(f"   • CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   • CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"     - GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        return False


def main():
    """Main test function."""
    try:
        # Test model compatibility first
        if not test_model_compatibility():
            print("\n❌ Model compatibility test failed!")
            sys.exit(1)
        
        # Test complete pipeline
        if not test_complete_pipeline():
            print("\n❌ Pipeline test failed!")
            sys.exit(1)
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 50)
        print("\n✅ Your Tyra MCP Memory Server is ready!")
        print("\n🚀 Next steps:")
        print("   1. Start the server: python main.py")
        print("   2. Test MCP tools with Claude or other agents")
        print("   3. Monitor performance in production")
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()