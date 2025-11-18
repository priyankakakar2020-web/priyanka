#!/usr/bin/env python3
"""
Flask web server for the mutual fund FAQ chatbot.
Provides a REST API and serves the frontend UI.

Usage:
    py -3 app.py
    
Then open: http://localhost:5000
"""

from __future__ import annotations

import sys
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add scripts directory to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from rag_query import retrieve, compose_answer

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

@app.route('/')
def index():
    """Serve the main chatbot UI"""
    return send_from_directory('static', 'index.html')

@app.route('/api/query', methods=['POST'])
def query():
    """
    Handle chatbot queries.
    
    Request JSON:
        {
            "question": "What is the expense ratio of JM Value Fund?"
        }
    
    Response JSON:
        {
            "success": true,
            "question": "What is the expense ratio of JM Value Fund?",
            "answer": "JM Value Fund Direct Plan Growth - Expense Ratio: 0.98%...",
            "source": "https://groww.in/mutual-funds/jm-basic-fund-direct-growth"
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing question in request'
            }), 400
        
        question = data['question'].strip()
        
        if not question:
            return jsonify({
                'success': False,
                'error': 'Question cannot be empty'
            }), 400
        
        # Retrieve relevant documents
        hits = retrieve(question, top_k=3)
        
        # Compose answer
        answer_text = compose_answer(question, hits)
        
        # Extract source URL from the top hit
        source_url = None
        if hits and len(hits) > 0:
            source_url = hits[0]['metadata'].get('url')
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': answer_text,
            'source': source_url
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Mutual Fund FAQ Chatbot'
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    
    print("="*80)
    print("🚀 Starting Mutual Fund FAQ Chatbot Server")
    print("="*80)
    print(f"\n📍 Server running on port: {port}")
    print("🔧 API endpoint: /api/query")
    print("\n✨ Visit the app in your browser\n")
    print("="*80)
    
    app.run(host='0.0.0.0', port=port, debug=False)
