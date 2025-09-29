"""RAG pipeline connecting FAISS search with Ollama LLM."""

import json
import requests
from pathlib import Path
from typing import List, Dict, Optional
import logging
from sentence_transformers import SentenceTransformer
import sys
sys.path.append(str(Path(__file__).parent.parent))

from retrieval.faiss_indexer import FAISSIndexer

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline with retrieval and generation."""
    
    def __init__(self, config: dict):
        self.config = config
        self.llm_config = config['llm']
        self.rag_config = config['rag']
        
        # Initialize components
        self.embedding_model = None
        self.faiss_indexer = None
        self.chunks_cache = None
        
    def initialize(self):
        """Load all required models and indices."""
        logger.info("Initializing RAG pipeline...")
        
        # Load embedding model
        logger.info(f"Loading embedding model: {self.config['embeddings']['model_name']}")
        self.embedding_model = SentenceTransformer(self.config['embeddings']['model_name'])
        
        # Load FAISS index
        logger.info("Loading FAISS index...")
        self.faiss_indexer = FAISSIndexer(self.config)
        embeddings_dir = self.config['data']['embeddings_dir']
        self.faiss_indexer.load_index(embeddings_dir)
        
        # Load chunks for text retrieval
        logger.info("Loading chunks cache...")
        chunks_file = f"{self.config['data']['processed_dir']}chunks.json"
        with open(chunks_file, 'r', encoding='utf-8') as f:
            self.chunks_cache = json.load(f)
        
        logger.info("RAG pipeline initialized successfully")
    
    def retrieve_context(self, query: str, top_k: int = None) -> List[Dict[str, any]]:
        """Retrieve relevant context chunks for a query."""
        if top_k is None:
            top_k = self.rag_config['top_k']
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query], 
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Search FAISS index
        results = self.faiss_indexer.search(
            query_embedding, 
            top_k=top_k,
            score_threshold=self.rag_config.get('score_threshold')
        )
        
        # Add full text to results
        for result in results:
            idx = result['embedding_index']
            if idx < len(self.chunks_cache):
                result['text'] = self.chunks_cache[idx]['text']
        
        return results
    
    def format_context(self, retrieved_chunks: List[Dict[str, any]]) -> str:
        """Format retrieved chunks into context string."""
        context_parts = []
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            title = chunk.get('title', 'Unknown')
            text = chunk.get('text', '')
            score = chunk.get('similarity_score', 0.0)
            
            context_parts.append(
                f"[Source {i}: {title} (relevance: {score:.3f})]\n{text}\n"
            )
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate response using Ollama LLM with context."""
        
        # Build prompt
        system_prompt = """You are a helpful AI assistant with access to Wikipedia knowledge. 
Answer questions based on the provided context from Wikipedia articles. 
If the context doesn't contain relevant information, say so clearly.
Be concise but informative."""
        
        user_prompt = f"""Context from Wikipedia:
{context}

Question: {query}

Answer based on the context above:"""
        
        # Call Ollama API
        try:
            response = requests.post(
                f"{self.llm_config['base_url']}/api/generate",
                json={
                    "model": self.llm_config['model'],
                    "prompt": f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:",
                    "stream": False,
                    "options": {
                        "temperature": self.llm_config['temperature'],
                        "num_predict": self.llm_config['max_tokens']
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return f"Error generating response: {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return f"Error: Could not connect to Ollama. Is it running? (brew services start ollama)"
    
    def query(self, question: str, verbose: bool = False) -> Dict[str, any]:
        """Complete RAG query: retrieve context and generate answer."""
        
        if verbose:
            print(f"\n🔍 Searching Wikipedia for: '{question}'")
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve_context(question)
        
        if verbose:
            print(f"📚 Found {len(retrieved_chunks)} relevant sources:")
            for i, chunk in enumerate(retrieved_chunks, 1):
                print(f"   {i}. {chunk['title']} (score: {chunk['similarity_score']:.3f})")
        
        # Format context
        context = self.format_context(retrieved_chunks)
        
        # Trim context if too long
        max_context_length = self.rag_config['max_context_length']
        if len(context) > max_context_length:
            context = context[:max_context_length] + "\n[...context truncated...]"
        
        if verbose:
            print(f"\n🤖 Generating answer using {self.llm_config['model']}...")
        
        # Generate response
        answer = self.generate_response(question, context)
        
        return {
            'question': question,
            'answer': answer,
            'sources': [
                {
                    'title': chunk['title'],
                    'score': chunk['similarity_score'],
                    'text_preview': chunk.get('text', '')[:200] + '...'
                }
                for chunk in retrieved_chunks
            ],
            'num_sources': len(retrieved_chunks)
        }


def test_rag_pipeline():
    """Test the complete RAG pipeline."""
    import yaml
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("🚀 Initializing RAG Pipeline...")
    rag = RAGPipeline(config)
    rag.initialize()
    
    print("\n✅ RAG Pipeline ready!\n")
    
    # Test queries
    test_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain quantum computing",
        "What is the theory of relativity?"
    ]
    
    for query in test_queries:
        print("=" * 80)
        result = rag.query(query, verbose=True)
        print(f"\n💬 Answer:\n{result['answer']}\n")
    
    # Interactive mode
    print("=" * 80)
    print("\n🎮 Interactive Mode - Ask anything about Wikipedia!")
    print("(Type 'quit' to exit)\n")
    
    while True:
        try:
            user_query = input("You: ").strip()
            
            if not user_query:
                continue
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            result = rag.query(user_query, verbose=True)
            print(f"\n💬 Assistant:\n{result['answer']}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    test_rag_pipeline()