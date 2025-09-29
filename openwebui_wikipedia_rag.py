"""
title: Wikipedia RAG Function
author: Your Name
description: Enhances responses with Wikipedia knowledge using RAG
version: 1.0.0
"""

from pydantic import BaseModel, Field
from typing import Optional, Generator
import json


class Pipe:
    """OpenWebUI Pipe for Wikipedia RAG integration."""
    
    class Valves(BaseModel):
        """Configuration valves for the pipe."""
        embeddings_dir: str = Field(
            default="data/embeddings/",
            description="Directory containing FAISS index and embeddings"
        )
        chunks_file: str = Field(
            default="data/processed/chunks.json",
            description="Path to chunks JSON file"
        )
        top_k: int = Field(
            default=5,
            description="Number of chunks to retrieve"
        )
        ollama_model: str = Field(
            default="mistral:latest",
            description="Ollama model to use"
        )
        enable_rag: bool = Field(
            default=True,
            description="Enable Wikipedia RAG enhancement"
        )
    
    def __init__(self):
        self.valves = self.Valves()
        self.rag_pipeline = None
        self.initialized = False
    
    def _initialize_rag(self):
        """Lazy initialization of RAG pipeline."""
        if self.initialized:
            return
        
        try:
            import sys
            from pathlib import Path
            
            # Add project root to path
            project_root = Path(__file__).parent.parent
            sys.path.append(str(project_root))
            
            from src.llm_integration.rag_pipeline import RAGPipeline
            import yaml
            
            # Load config
            config_path = project_root / "config" / "config.yaml"
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Override with valve settings
            config['llm']['model'] = self.valves.ollama_model
            config['rag']['top_k'] = self.valves.top_k
            config['data']['embeddings_dir'] = self.valves.embeddings_dir
            config['data']['processed_dir'] = str(Path(self.valves.chunks_file).parent) + "/"
            
            # Initialize pipeline
            self.rag_pipeline = RAGPipeline(config)
            self.rag_pipeline.initialize()
            self.initialized = True
            
            print("✅ Wikipedia RAG pipeline initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize RAG pipeline: {e}")
            self.initialized = False
    
    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: list,
        body: dict
    ) -> str:
        """Process messages through the RAG pipeline."""
        
        # Initialize if needed
        if not self.initialized and self.valves.enable_rag:
            self._initialize_rag()
        
        # If RAG is disabled or failed to initialize, pass through
        if not self.valves.enable_rag or not self.initialized:
            return user_message
        
        try:
            # Get RAG response
            result = self.rag_pipeline.query(user_message, verbose=False)
            
            # Format response with sources
            response = result['answer']
            
            if result['sources']:
                response += "\n\n---\n**Sources:**\n"
                for i, source in enumerate(result['sources'][:3], 1):
                    response += f"{i}. {source['title']} (relevance: {source['score']:.2f})\n"
            
            return response
            
        except Exception as e:
            print(f"❌ RAG query error: {e}")
            return f"I encountered an error accessing Wikipedia: {e}\n\nOriginal question: {user_message}"