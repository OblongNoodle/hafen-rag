"""
Flask API wrapper for Haven & Hearth RAG
Deploy to cPanel with hafen_chroma_db folder
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

app = Flask(__name__)
CORS(app)

# Configuration
CHROMA_DIR = "./hafen_chroma_db"
vectorstore = None
embeddings = None

def init_vectorstore():
    """Load vectorstore on startup"""
    global vectorstore, embeddings
    
    if not os.path.exists(CHROMA_DIR):
        print(f"ERROR: Vector store not found at {CHROMA_DIR}")
        return False
    
    try:
        print("Loading embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        print("Loading vector database...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        print("Vector store loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        return False

@app.route('/api/search', methods=['POST'])
def search():
    """Search endpoint"""
    if not vectorstore:
        return jsonify({'error': 'Vector store not initialized'}), 500
    
    try:
        data = request.json
        query = data.get('query', '').strip()
        top_k = data.get('top_k', 10)
        max_files = data.get('max_files', 5)
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Search
        results = vectorstore.similarity_search_with_score(query, k=top_k)
        
        # Group by source file
        grouped = {}
        for doc, score in results:
            source = doc.metadata['source']
            if source not in grouped:
                grouped[source] = {
                    'source': source,
                    'filename': doc.metadata.get('filename', ''),
                    'chunks': [],
                    'best_score': score
                }
            grouped[source]['chunks'].append({
                'content': doc.page_content,
                'score': float(score)
            })
        
        # Sort by best score and limit files
        sorted_results = sorted(grouped.values(), key=lambda x: x['best_score'])[:max_files]
        
        return jsonify({
            'query': query,
            'results': sorted_results,
            'count': len(sorted_results)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'vectorstore_loaded': vectorstore is not None
    })

@app.route('/')
def index():
    """Serve frontend"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('.', filename)

if __name__ == '__main__':
    print("Initializing Haven & Hearth RAG API...")
    if init_vectorstore():
        print("Starting Flask server...")
        app.run(debug=False, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize vectorstore")
        sys.exit(1)