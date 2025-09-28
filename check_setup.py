#!/usr/bin/env python3
"""Check if the Wikipedia RAG system is properly set up."""

import sys
from pathlib import Path

def check_files():
    """Check if all required files exist."""
    required_files = [
        "config/config.yaml",
        "src/__init__.py",
        "src/utils.py",
        "src/data_processing/__init__.py",
        "src/data_processing/wikipedia_parser.py",
        "src/data_processing/text_chunker.py",
        "src/embeddings/__init__.py", 
        "src/embeddings/embedding_generator.py",
        "src/retrieval/__init__.py",
        "src/retrieval/faiss_indexer.py",
        "requirements.txt",
        "test_setup.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   {file}")
        return False
    else:
        print("✅ All required files present")
        return True

def check_data():
    """Check if data processing has been completed."""
    data_checks = [
        ("Wikipedia dump", "data/raw/simplewiki-latest-pages-articles.xml.bz2"),
        ("Processed chunks", "data/processed/chunks.json"),
        ("Embeddings", "data/embeddings/embeddings.npy"),
        ("FAISS index", "data/embeddings/faiss_index.bin"),
        ("Metadata", "data/embeddings/metadata.json")
    ]
    
    print("\n📊 Data Processing Status:")
    all_present = True
    
    for name, path in data_checks:
        path_obj = Path(path)
        if path_obj.exists():
            size_mb = path_obj.stat().st_size / 1024 / 1024
            print(f"✅ {name}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {name}: {path} (missing)")
            all_present = False
    
    return all_present

def check_imports():
    """Check if all required packages can be imported."""
    required_packages = [
        "numpy", "pandas", "tqdm", "lxml", "sentence_transformers", 
        "faiss", "yaml", "json", "pathlib"
    ]
    
    print("\n🐍 Package Import Status:")
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_imports.append(package)
    
    return len(failed_imports) == 0

def main():
    print("🔍 Wikipedia RAG System Setup Check\n")
    
    files_ok = check_files()
    data_ok = check_data()
    imports_ok = check_imports()
    
    print(f"\n📋 Summary:")
    print(f"   Files: {'✅ OK' if files_ok else '❌ Missing files'}")
    print(f"   Data: {'✅ Complete' if data_ok else '❌ Incomplete'}")
    print(f"   Packages: {'✅ All imported' if imports_ok else '❌ Missing packages'}")
    
    if files_ok and data_ok and imports_ok:
        print(f"\n🎉 System is ready for Ollama integration!")
        return True
    else:
        print(f"\n⚠️  System needs setup completion.")
        if not files_ok:
            print("   Run: git pull to get missing files")
        if not imports_ok:
            print("   Run: pip install -r requirements.txt")
        if not data_ok:
            print("   Run the processing pipeline step by step")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)