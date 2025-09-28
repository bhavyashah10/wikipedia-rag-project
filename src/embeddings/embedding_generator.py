"""Embedding generation for Wikipedia chunks using sentence-transformers."""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import logging
from sentence_transformers import SentenceTransformer
import pickle

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text chunks using sentence-transformers."""
    
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config['embeddings']['model_name']
        self.dimension = config['embeddings']['dimension']
        self.normalize = config['embeddings'].get('normalize', True)
        self.model = None
        
    def load_model(self):
        """Load the sentence transformer model."""
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if self.model is None:
            self.load_model()
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Process in batches to manage memory
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        
        return embeddings
    
    def process_chunks_file(self, chunks_file: str, output_dir: str, batch_size: int = 1000) -> Dict[str, any]:
        """Process chunks file and generate embeddings."""
        
        # Load chunks
        logger.info(f"Loading chunks from {chunks_file}")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        logger.info(f"Loaded {len(chunks)} chunks")
        
        # Extract texts for embedding
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings in batches
        all_embeddings = []
        all_metadata = []
        
        total_batches = len(chunks) // batch_size + (1 if len(chunks) % batch_size != 0 else 0)
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(chunks))
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches}: chunks {start_idx} to {end_idx}")
            
            # Get batch
            batch_chunks = chunks[start_idx:end_idx]
            batch_texts = [chunk['text'] for chunk in batch_chunks]
            
            # Generate embeddings
            batch_embeddings = self.generate_embeddings(batch_texts, batch_size=32)
            all_embeddings.append(batch_embeddings)
            
            # Store metadata (everything except text to save space)
            batch_metadata = []
            for chunk in batch_chunks:
                metadata = {k: v for k, v in chunk.items() if k != 'text'}
                metadata['embedding_index'] = len(all_metadata) + len(batch_metadata)
                batch_metadata.append(metadata)
            
            all_metadata.extend(batch_metadata)
            
            # Save intermediate results to avoid data loss
            if batch_idx % 5 == 0:  # Save every 5 batches
                self._save_intermediate(all_embeddings, all_metadata, output_dir, batch_idx)
        
        # Combine all embeddings
        final_embeddings = np.vstack(all_embeddings)
        
        # Save final results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        embeddings_file = output_path / "embeddings.npy"
        np.save(embeddings_file, final_embeddings)
        logger.info(f"Saved embeddings to {embeddings_file}")
        
        # Save metadata
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(all_metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata to {metadata_file}")
        
        # Save configuration
        config_file = output_path / "embedding_config.json"
        embedding_config = {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'normalize': self.normalize,
            'total_chunks': len(chunks),
            'embedding_shape': final_embeddings.shape,
            'created_from': chunks_file
        }
        with open(config_file, 'w') as f:
            json.dump(embedding_config, f, indent=2)
        logger.info(f"Saved config to {config_file}")
        
        # Clean up intermediate files
        self._cleanup_intermediate(output_dir)
        
        stats = {
            'total_chunks': len(chunks),
            'embedding_shape': final_embeddings.shape,
            'model_name': self.model_name,
            'output_files': {
                'embeddings': str(embeddings_file),
                'metadata': str(metadata_file),
                'config': str(config_file)
            }
        }
        
        logger.info(f"Embedding generation complete: {stats}")
        return stats
    
    def _save_intermediate(self, embeddings_list: List[np.ndarray], metadata: List[Dict], 
                          output_dir: str, batch_idx: int):
        """Save intermediate results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings checkpoint
        if embeddings_list:
            combined_embeddings = np.vstack(embeddings_list)
            checkpoint_file = output_path / f"embeddings_checkpoint_{batch_idx}.npy"
            np.save(checkpoint_file, combined_embeddings)
        
        # Save metadata checkpoint
        metadata_checkpoint = output_path / f"metadata_checkpoint_{batch_idx}.json"
        with open(metadata_checkpoint, 'w', encoding='utf-8') as f:
            json.dump(metadata, f)
    
    def _cleanup_intermediate(self, output_dir: str):
        """Clean up intermediate checkpoint files."""
        output_path = Path(output_dir)
        
        # Remove checkpoint files
        for checkpoint_file in output_path.glob("*_checkpoint_*.npy"):
            checkpoint_file.unlink()
        for checkpoint_file in output_path.glob("*_checkpoint_*.json"):
            checkpoint_file.unlink()
        
        logger.info("Cleaned up intermediate checkpoint files")


def test_embedding_generator():
    """Test the embedding generator with sample chunks."""
    
    # Simple config for testing
    config = {
        'embeddings': {
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'dimension': 384,
            'normalize': True
        }
    }
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    generator = EmbeddingGenerator(config)
    
    # Test with actual chunks file
    chunks_file = "data/processed/chunks.json"
    output_dir = "data/embeddings/"
    
    if not Path(chunks_file).exists():
        print(f"❌ Chunks file not found: {chunks_file}")
        print("Run the chunker first: python src/data_processing/text_chunker.py")
        return False
    
    print(f"🔄 Generating embeddings for all chunks...")
    print(f"📁 Input: {chunks_file}")
    print(f"📁 Output: {output_dir}")
    print(f"🤖 Model: {config['embeddings']['model_name']}")
    
    stats = generator.process_chunks_file(chunks_file, output_dir, batch_size=1000)
    
    print(f"\n📊 Embedding Generation Results:")
    print(f"   Total chunks: {stats['total_chunks']:,}")
    print(f"   Embedding shape: {stats['embedding_shape']}")
    print(f"   Model: {stats['model_name']}")
    print(f"   Output files:")
    for name, path in stats['output_files'].items():
        file_size = Path(path).stat().st_size / 1024 / 1024  # MB
        print(f"     {name}: {path} ({file_size:.1f} MB)")
    
    return True


if __name__ == "__main__":
    print("Testing Embedding Generator...\n")
    success = test_embedding_generator()
    
    if success:
        print(f"\n🎉 Embedding generation completed successfully!")
        print(f"Next step: Build FAISS index for fast similarity search")
    else:
        print(f"\n❌ Embedding generation failed.")