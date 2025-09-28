"""FAISS index builder and searcher for Wikipedia embeddings."""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import pickle
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FAISSIndexer:
    """Build and manage FAISS index for fast similarity search."""
    
    def __init__(self, config: dict):
        self.config = config
        self.index_type = config['faiss']['index_type']
        self.metric = config['faiss']['metric']
        self.dimension = config['embeddings']['dimension']
        
        self.index = None
        self.metadata = None
        
    def build_index(self, embeddings_dir: str) -> Dict[str, any]:
        """Build FAISS index from embeddings."""
        embeddings_path = Path(embeddings_dir)
        
        # Load embeddings
        embeddings_file = embeddings_path / "embeddings.npy"
        metadata_file = embeddings_path / "metadata.json"
        
        logger.info(f"Loading embeddings from {embeddings_file}")
        embeddings = np.load(embeddings_file)
        
        logger.info(f"Loading metadata from {metadata_file}")
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        logger.info(f"Building FAISS index for {embeddings.shape[0]} vectors of dimension {embeddings.shape[1]}")
        
        # Ensure embeddings are float32 (FAISS requirement)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Create FAISS index based on configuration
        if self.index_type.lower() == "flat":
            if self.metric.lower() == "cosine":
                # For cosine similarity, normalize vectors and use L2
                faiss.normalize_L2(embeddings)
                index = faiss.IndexFlatL2(self.dimension)
            else:
                # Euclidean distance
                index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type.lower() == "ivf":
            # IVF (Inverted File) for approximate search - faster but less accurate
            nlist = min(100, embeddings.shape[0] // 100)  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            
            # Train the index
            logger.info(f"Training IVF index with {nlist} clusters...")
            index.train(embeddings)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # Add vectors to index
        logger.info("Adding vectors to index...")
        index.add(embeddings)
        
        # Store index and metadata
        self.index = index
        self.metadata = metadata
        
        # Save index to disk
        index_file = embeddings_path / "faiss_index.bin"
        logger.info(f"Saving FAISS index to {index_file}")
        faiss.write_index(index, str(index_file))
        
        # Save metadata mapping
        metadata_mapping_file = embeddings_path / "index_metadata.pkl"
        with open(metadata_mapping_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save index configuration
        index_config = {
            'index_type': self.index_type,
            'metric': self.metric,
            'dimension': self.dimension,
            'total_vectors': embeddings.shape[0],
            'index_file': str(index_file),
            'metadata_file': str(metadata_mapping_file)
        }
        
        config_file = embeddings_path / "faiss_config.json"
        with open(config_file, 'w') as f:
            json.dump(index_config, f, indent=2)
        
        stats = {
            'total_vectors': embeddings.shape[0],
            'index_type': self.index_type,
            'metric': self.metric,
            'dimension': self.dimension,
            'index_size_mb': index_file.stat().st_size / 1024 / 1024,
            'files': {
                'index': str(index_file),
                'metadata': str(metadata_mapping_file),
                'config': str(config_file)
            }
        }
        
        logger.info(f"FAISS index built successfully: {stats}")
        return stats
    
    def load_index(self, embeddings_dir: str):
        """Load existing FAISS index from disk."""
        embeddings_path = Path(embeddings_dir)
        
        index_file = embeddings_path / "faiss_index.bin"
        metadata_file = embeddings_path / "index_metadata.pkl"
        
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_file}")
        
        logger.info(f"Loading FAISS index from {index_file}")
        self.index = faiss.read_index(str(index_file))
        
        logger.info(f"Loading metadata from {metadata_file}")
        with open(metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)
        
        logger.info(f"Loaded index with {self.index.ntotal} vectors")
    
    def search(self, query_vector: np.ndarray, top_k: int = 5, score_threshold: float = None) -> List[Dict[str, any]]:
        """Search for similar vectors in the index."""
        if self.index is None:
            raise RuntimeError("Index not loaded. Call build_index() or load_index() first.")
        
        # Ensure query vector is the right shape and type
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)
        
        # Normalize for cosine similarity if needed
        if self.metric.lower() == "cosine":
            faiss.normalize_L2(query_vector)
        
        # Search
        scores, indices = self.index.search(query_vector, top_k)
        
        # Convert results to list of dictionaries
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue
            
            # Apply score threshold if specified
            if score_threshold is not None and score > score_threshold:
                continue
            
            # Get metadata for this chunk
            chunk_metadata = self.metadata[idx].copy()
            chunk_metadata['similarity_score'] = float(score)
            chunk_metadata['rank'] = i + 1
            
            results.append(chunk_metadata)
        
        return results
    
    def get_chunk_text(self, chunk_metadata: Dict[str, any], chunks_file: str = None) -> str:
        """Retrieve the full text for a chunk given its metadata."""
        if chunks_file is None:
            chunks_file = "data/processed/chunks.json"
        
        # For efficiency, we could cache this, but for now, load on demand
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # Find the chunk by matching metadata
        embedding_index = chunk_metadata['embedding_index']
        
        # The embedding index should correspond to the position in the chunks array
        if embedding_index < len(chunks):
            return chunks[embedding_index]['text']
        
        # Fallback: search by title and chunk_id
        for chunk in chunks:
            if (chunk['title'] == chunk_metadata['title'] and 
                chunk['chunk_id'] == chunk_metadata['chunk_id']):
                return chunk['text']
        
        return None


def test_faiss_indexer():
    """Test the FAISS indexer with sample queries."""
    
    # Simple config for testing
    config = {
        'faiss': {
            'index_type': 'flat',
            'metric': 'cosine'
        },
        'embeddings': {
            'dimension': 384
        }
    }
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    indexer = FAISSIndexer(config)
    embeddings_dir = "data/embeddings/"
    
    # Check if embeddings exist
    if not Path(f"{embeddings_dir}/embeddings.npy").exists():
        print(f"❌ Embeddings not found in {embeddings_dir}")
        print("Run the embedding generator first: python src/embeddings/embedding_generator.py")
        return False
    
    print(f"🔄 Building FAISS index...")
    stats = indexer.build_index(embeddings_dir)
    
    print(f"\n📊 FAISS Index Results:")
    print(f"   Total vectors: {stats['total_vectors']:,}")
    print(f"   Index type: {stats['index_type']}")
    print(f"   Metric: {stats['metric']}")
    print(f"   Dimension: {stats['dimension']}")
    print(f"   Index size: {stats['index_size_mb']:.1f} MB")
    
    # Test search functionality
    print(f"\n🔍 Testing search functionality...")
    
    # Load the embedding model to create test queries
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Test queries
    test_queries = [
        "What is artificial intelligence?",
        "How do computers work?",
        "Tell me about space exploration",
        "What is machine learning?"
    ]
    
    for query in test_queries:
        print(f"\n--- Query: '{query}' ---")
        
        # Generate query embedding
        query_embedding = model.encode([query], normalize_embeddings=True)
        
        # Search
        results = indexer.search(query_embedding, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']} (score: {result['similarity_score']:.3f})")
            
            # Get the actual text
            chunk_text = indexer.get_chunk_text(result)
            if chunk_text:
                preview = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
                print(f"   Preview: {preview}")
    
    return True


if __name__ == "__main__":
    print("Testing FAISS Indexer...\n")
    success = test_faiss_indexer()
    
    if success:
        print(f"\n🎉 FAISS index built and tested successfully!")
        print(f"Next step: Integrate with Ollama LLM for RAG pipeline")
    else:
        print(f"\n❌ FAISS indexer test failed.")