"""Text chunking for optimal RAG retrieval."""

import json
import re
from pathlib import Path
from typing import List, Dict, Generator
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class TextChunker:
    """Intelligent text chunker for Wikipedia articles."""
    
    def __init__(self, config: dict):
        self.config = config
        self.chunk_size = config['processing']['chunk_size']
        self.overlap = config['processing']['chunk_overlap']
    
    def split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple rules."""
        # Simple sentence splitting - handles most cases
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_chunks(self, text: str, title: str) -> List[Dict[str, str]]:
        """Create overlapping chunks from text."""
        if len(text) <= self.chunk_size:
            # Text is short enough, return as single chunk
            return [{
                'text': text,
                'title': title,
                'chunk_id': 0,
                'char_start': 0,
                'char_end': len(text)
            }]
        
        chunks = []
        sentences = self.split_by_sentences(text)
        
        current_chunk = ""
        current_start = 0
        chunk_id = 0
        
        for i, sentence in enumerate(sentences):
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'text': current_chunk.strip(),
                    'title': title,
                    'chunk_id': chunk_id,
                    'char_start': current_start,
                    'char_end': current_start + len(current_chunk)
                })
                
                # Start new chunk with overlap
                # Find overlap point (last few sentences)
                overlap_text = ""
                if self.overlap > 0:
                    words = current_chunk.split()
                    if len(words) > 20:  # Only add overlap if chunk is substantial
                        overlap_words = words[-min(len(words)//4, 50):]  # Last 25% or 50 words
                        overlap_text = " ".join(overlap_words)
                
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                current_start = current_start + len(current_chunk) - len(overlap_text) if overlap_text else current_start + len(current_chunk)
                chunk_id += 1
            else:
                current_chunk = potential_chunk
        
        # Add final chunk if there's remaining text
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'title': title,
                'chunk_id': chunk_id,
                'char_start': current_start,
                'char_end': current_start + len(current_chunk)
            })
        
        return chunks
    
    def chunk_articles(self, articles: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert articles to chunks."""
        all_chunks = []
        
        for article in tqdm(articles, desc="Chunking articles"):
            title = article['title']
            text = article['text']
            article_id = article.get('id', '')
            
            chunks = self.create_chunks(text, title)
            
            # Add metadata to each chunk
            for chunk in chunks:
                chunk.update({
                    'article_id': article_id,
                    'article_length': len(text),
                    'chunk_length': len(chunk['text']),
                    'source': 'wikipedia'
                })
                all_chunks.append(chunk)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(articles)} articles")
        return all_chunks
    
    def save_chunks(self, chunks: List[Dict[str, str]], output_file: str):
        """Save chunks to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(chunks)} chunks to {output_file}")
    
    def process_article_files(self, input_dir: str, output_file: str) -> Dict[str, int]:
        """Process all article JSON files and create chunks."""
        input_path = Path(input_dir)
        all_chunks = []
        stats = {'articles': 0, 'chunks': 0}
        
        # Find all JSON files
        json_files = list(input_path.glob("*.json"))
        logger.info(f"Found {len(json_files)} article files to process")
        
        for json_file in tqdm(json_files, desc="Processing files"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
                
                if not isinstance(articles, list):
                    logger.warning(f"Skipping {json_file}: not a list of articles")
                    continue
                
                chunks = self.chunk_articles(articles)
                all_chunks.extend(chunks)
                stats['articles'] += len(articles)
                
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
                continue
        
        stats['chunks'] = len(all_chunks)
        
        # Save all chunks
        if all_chunks:
            self.save_chunks(all_chunks, output_file)
        
        logger.info(f"Chunking complete: {stats}")
        return stats


def test_chunker():
    """Test the text chunker."""
    
    # Simple config for testing
    config = {
        'processing': {
            'chunk_size': 1000,
            'chunk_overlap': 200
        }
    }
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    chunker = TextChunker(config)
    
    # Process the parsed articles
    input_dir = "data/processed/"
    output_file = "data/processed/chunks.json"
    
    print("🔄 Creating text chunks...")
    stats = chunker.process_article_files(input_dir, output_file)
    
    print(f"📊 Chunking Results:")
    print(f"   Articles processed: {stats['articles']:,}")
    print(f"   Chunks created: {stats['chunks']:,}")
    print(f"   Average chunks per article: {stats['chunks']/stats['articles']:.1f}")
    
    # Show sample chunks
    if Path(output_file).exists():
        with open(output_file, 'r') as f:
            chunks = json.load(f)
        
        print(f"\n📝 Sample chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Title: {chunk['title']}")
            print(f"Length: {chunk['chunk_length']} chars")
            print(f"Text preview: {chunk['text'][:200]}...")


if __name__ == "__main__":
    test_chunker()