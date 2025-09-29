"""Simple web interface for Wikipedia RAG using Flask."""

from flask import Flask, render_template, request, jsonify, stream_with_context, Response
import yaml
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from llm_integration.rag_pipeline import RAGPipeline

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Global RAG pipeline
rag_pipeline = None


def initialize_rag():
    """Initialize the RAG pipeline."""
    global rag_pipeline
    
    if rag_pipeline is not None:
        return
    
    print("🚀 Initializing Wikipedia RAG Pipeline...")
    
    # Load config
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    rag_pipeline = RAGPipeline(config)
    rag_pipeline.initialize()
    
    print("✅ RAG Pipeline ready!")


@app.route('/')
def index():
    """Serve the main chat interface."""
    return render_template('chat.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests."""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get RAG response
        result = rag_pipeline.query(user_message, verbose=False)
        
        # Format response
        response = {
            'answer': result['answer'],
            'sources': [
                {
                    'title': source['title'],
                    'score': round(source['score'], 3),
                    'preview': source['text_preview']
                }
                for source in result['sources']
            ]
        }
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Error processing chat request: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'rag_initialized': rag_pipeline is not None
    })


if __name__ == '__main__':
    initialize_rag()
    print("\n" + "="*60)
    print("🌐 Wikipedia RAG Web Interface")
    print("="*60)
    print("📍 Open your browser to: http://localhost:5000")
    print("💬 Start chatting with Wikipedia!")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)