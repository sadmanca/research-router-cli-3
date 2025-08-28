#!/usr/bin/env python3
"""
Research Router Web App - Simple web interface for nano-graphrag document chat
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import tempfile
import uuid

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from werkzeug.utils import secure_filename

# Import existing CLI components
from research_router_cli.commands.session import SessionManager
from research_router_cli.commands.insert import InsertCommand
from research_router_cli.commands.query import QueryCommand
from research_router_cli.utils.config import Config

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Initialize CLI components
session_manager = SessionManager()
insert_command = InsertCommand(session_manager)
query_command = QueryCommand(session_manager)
config = Config()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'md', 'doc', 'docx'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class WebSessionManager:
    """Web-specific session management"""
    
    @staticmethod
    def get_current_session():
        return session.get('current_session')
    
    @staticmethod
    def set_current_session(session_name: str):
        session['current_session'] = session_name
    
    @staticmethod
    def clear_session():
        session.pop('current_session', None)

@app.route('/')
def index():
    """Main page - session selection or creation"""
    current_session = WebSessionManager.get_current_session()
    sessions = session_manager.sessions
    
    # If no sessions exist, redirect to create session
    if not sessions:
        return render_template('create_session.html')
    
    return render_template('index.html', 
                         sessions=sessions, 
                         current_session=current_session)

@app.route('/create_session')
def create_session_page():
    """Session creation page"""
    return render_template('create_session.html')

@app.route('/api/sessions', methods=['POST'])
def api_create_session():
    """API: Create a new session"""
    data = request.get_json()
    session_name = data.get('name', '').strip()
    
    if not session_name:
        return jsonify({'error': 'Session name is required'}), 400
    
    if session_name in session_manager.sessions:
        return jsonify({'error': f'Session "{session_name}" already exists'}), 400
    
    # Create session
    success = session_manager.create_session(session_name)
    if success:
        WebSessionManager.set_current_session(session_name)
        # Reset command instances for new session
        insert_command.reset_instance()
        query_command.reset_instance()
        return jsonify({'success': True, 'session': session_name})
    else:
        return jsonify({'error': 'Failed to create session'}), 500

@app.route('/api/sessions/<session_name>/switch', methods=['POST'])
def api_switch_session(session_name):
    """API: Switch to a session"""
    if session_name not in session_manager.sessions:
        return jsonify({'error': f'Session "{session_name}" does not exist'}), 404
    
    # Switch session
    session_manager.switch_session(session_name)
    WebSessionManager.set_current_session(session_name)
    
    # Reset command instances for switched session
    insert_command.reset_instance()
    query_command.reset_instance()
    
    return jsonify({'success': True, 'session': session_name})

@app.route('/chat')
def chat():
    """Main chat interface"""
    current_session = WebSessionManager.get_current_session()
    
    if not current_session:
        flash('Please select or create a session first', 'warning')
        return redirect(url_for('index'))
    
    # Get session history
    history_manager = session_manager.get_session_history_manager(current_session)
    recent_history = []
    if history_manager and history_manager.has_history():
        recent_history = history_manager.get_history(limit=10)
    
    return render_template('chat.html', 
                         session_name=current_session,
                         history=recent_history)

@app.route('/upload')
def upload_page():
    """Document upload page"""
    current_session = WebSessionManager.get_current_session()
    
    if not current_session:
        flash('Please select or create a session first', 'warning')
        return redirect(url_for('index'))
    
    return render_template('upload.html', session_name=current_session)

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """API: Upload and process documents"""
    current_session = WebSessionManager.get_current_session()
    
    if not current_session:
        return jsonify({'error': 'No active session'}), 400
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400
    
    results = []
    temp_files = []
    
    try:
        # Save uploaded files temporarily
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                
                # Create temp file with original extension
                suffix = Path(filename).suffix
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                temp_file.close()
                
                file.save(temp_file.name)
                temp_files.append((temp_file.name, filename))
                results.append({'filename': filename, 'status': 'uploaded'})
            else:
                results.append({'filename': file.filename, 'status': 'invalid', 'error': 'File type not allowed'})
        
        if not temp_files:
            return jsonify({'error': 'No valid files to process', 'results': results}), 400
        
        # Process files with insert command
        async def process_files():
            file_paths = [temp_path for temp_path, _ in temp_files]
            await insert_command.insert_multiple_files(file_paths)
        
        # Run async processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(process_files())
        loop.close()
        
        # Update results with success
        for i, (_, filename) in enumerate(temp_files):
            if i < len(results):
                results[i]['status'] = 'processed'
        
        return jsonify({'success': True, 'results': results})
    
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}', 'results': results}), 500
    
    finally:
        # Clean up temp files
        for temp_path, _ in temp_files:
            try:
                os.unlink(temp_path)
            except:
                pass

@app.route('/api/query', methods=['POST'])
def api_query():
    """API: Query the knowledge graph"""
    current_session = WebSessionManager.get_current_session()
    
    if not current_session:
        return jsonify({'error': 'No active session'}), 400
    
    data = request.get_json()
    question = data.get('question', '').strip()
    mode = data.get('mode', 'global').lower()
    
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    
    if mode not in ['local', 'global', 'naive']:
        mode = 'global'
    
    try:
        # Run query asynchronously
        async def run_query():
            # Switch to the current session
            session_manager.switch_session(current_session)
            
            # Get GraphRAG instance
            graphrag = await query_command._get_graphrag_instance()
            if not graphrag:
                return None, "No knowledge graph found. Please upload documents first."
            
            # Import QueryParam
            from nano_graphrag import QueryParam
            
            # Perform query
            result = await graphrag.aquery(
                question, 
                param=QueryParam(mode=mode)
            )
            
            # Save to history
            query_command._save_to_history(question, result, mode)
            
            return result, None
        
        # Run async query
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result, error = loop.run_until_complete(run_query())
        loop.close()
        
        if error:
            return jsonify({'error': error}), 400
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': result,
            'mode': mode
        })
    
    except Exception as e:
        return jsonify({'error': f'Query failed: {str(e)}'}), 500

@app.route('/api/sessions/<session_name>/status')
def api_session_status(session_name):
    """API: Get session status (documents, knowledge graph info)"""
    if session_name not in session_manager.sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    working_dir = Path(session_manager.sessions[session_name])
    
    # Check if knowledge graph exists
    graph_files = [
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json", 
        "vdb_entities.json"
    ]
    
    has_knowledge_graph = any((working_dir / filename).exists() for filename in graph_files)
    
    # Get basic stats
    stats = {}
    if has_knowledge_graph:
        try:
            # Count documents
            docs_file = working_dir / "kv_store_full_docs.json"
            if docs_file.exists():
                with open(docs_file, 'r', encoding='utf-8') as f:
                    docs = json.load(f)
                    stats['documents'] = len(docs)
            
            # Count text chunks  
            chunks_file = working_dir / "kv_store_text_chunks.json"
            if chunks_file.exists():
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                    stats['chunks'] = len(chunks)
        except:
            pass
    
    return jsonify({
        'session': session_name,
        'has_knowledge_graph': has_knowledge_graph,
        'stats': stats
    })

@app.route('/api/sessions/<session_name>/history')
def api_session_history(session_name):
    """API: Get session query history"""
    if session_name not in session_manager.sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    history_manager = session_manager.get_session_history_manager(session_name)
    if not history_manager or not history_manager.has_history():
        return jsonify({'history': []})
    
    limit = request.args.get('limit', 20, type=int)
    history = history_manager.get_history(limit=limit)
    
    return jsonify({'history': history})

if __name__ == '__main__':
    # Check config
    if not config.has_openai_config and not config.has_azure_openai_config:
        print("Warning: No OpenAI API configuration found.")
        print("Please set OPENAI_API_KEY in your environment or .env file.")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)