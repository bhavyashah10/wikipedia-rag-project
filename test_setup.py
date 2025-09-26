#!/usr/bin/env python3
"""Test script to verify the setup is working."""

import sys
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

def test_imports():
    """Test that all critical packages can be imported."""
    try:
        print("✓ Core packages imported successfully")
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ NumPy version: {np.__version__}")
        print(f"✓ Pandas version: {pd.__version__}")
        print(f"✓ FAISS available: {faiss.__version__ if hasattr(faiss, '__version__') else 'Yes'}")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_sentence_transformers():
    """Test sentence transformers (lightweight test)."""
    try:
        # This will download the model if not present (~90MB)
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Test encoding
        sentences = ["This is a test sentence.", "This is another test."]
        embeddings = model.encode(sentences)
        
        print(f"✓ Sentence Transformers working")
        print(f"✓ Embedding shape: {embeddings.shape}")
        return True
    except Exception as e:
        print(f"✗ Sentence Transformers error: {e}")
        return False

def test_faiss():
    """Test FAISS functionality."""
    try:
        # Create a simple FAISS index
        dimension = 384
        index = faiss.IndexFlatL2(dimension)
        
        # Add some random vectors
        vectors = np.random.random((10, dimension)).astype('float32')
        index.add(vectors)
        
        # Search
        query = np.random.random((1, dimension)).astype('float32')
        distances, indices = index.search(query, 3)
        
        print(f"✓ FAISS working - Index size: {index.ntotal}")
        return True
    except Exception as e:
        print(f"✗ FAISS error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Wikipedia RAG setup...\n")
    
    success = True
    success &= test_imports()
    success &= test_sentence_transformers()
    success &= test_faiss()
    
    if success:
        print(f"\n🎉 All tests passed! Setup is ready.")
        print(f"Python version: {sys.version}")
        print(f"Platform: {sys.platform}")
    else:
        print(f"\n❌ Some tests failed. Check the errors above.")
        sys.exit(1)