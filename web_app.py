#!/usr/bin/env python3
"""
Research Router Web App - Simple web interface for nano-graphrag document chat
"""

import asyncio
import json
import os
import logging
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import tempfile
import uuid

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, Response
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

# Chat message streaming setup
chat_streams = {}  # operation_id -> {'messages': [], 'complete': False}
chat_handlers = {}  # operation_id -> handler

class ChatLogHandler(logging.Handler):
    """Custom log handler that formats logs as chat messages"""
    def __init__(self, operation_id):
        super().__init__()
        self.operation_id = operation_id
        
    def emit(self, record):
        try:
            msg = self.format(record)
            if self.operation_id in chat_streams:
                message = {
                    'type': 'log',
                    'content': msg,
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'level': record.levelname
                }
                chat_streams[self.operation_id]['messages'].append(message)
        except:
            pass  # Ignore handler errors

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
    """API: Upload and process documents (returns immediately with operation_id)"""
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
    
    # Generate operation ID for this upload
    operation_id = str(uuid.uuid4())
    
    # Setup chat streaming for this operation
    setup_chat_streaming(operation_id)
    
    # Start background processing
    def process_files_background():
        success = False
        try:
            # Add initial message
            add_chat_message(operation_id, 'system', f'📁 Starting upload of {len(temp_files)} files...')
            
            # Switch to the current session before processing
            session_manager.switch_session(current_session)
            
            # Process files with insert command
            async def process_files():
                file_paths = [temp_path for temp_path, _ in temp_files]
                try:
                    await insert_command.insert_multiple_files(file_paths, skip_confirmation=True)
                    return True
                except Exception as e:
                    add_chat_message(operation_id, 'error', f'Insert operation failed: {str(e)}')
                    return False
            
            # Run async processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            success = loop.run_until_complete(process_files())
            loop.close()
            
            if success:
                # Signal successful completion
                add_chat_message(operation_id, 'system', '✅ Upload completed successfully!')
            else:
                add_chat_message(operation_id, 'system', '❌ Upload completed with errors. Check logs above.')
            mark_chat_complete(operation_id)
                
        except Exception as e:
            add_chat_message(operation_id, 'error', f'Upload process failed: {str(e)}')
            mark_chat_complete(operation_id)
        finally:
            # Clean up temp files
            for temp_path, _ in temp_files:
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    # Start processing in background thread
    threading.Thread(target=process_files_background, daemon=True).start()
    
    return jsonify({
        'success': True,
        'operation_id': operation_id,
        'results': results,
        'chat_endpoint': f'/api/chat/{operation_id}'
    })

@app.route('/api/query', methods=['POST'])
def api_query():
    """API: Query the knowledge graph (returns immediately with operation_id)"""
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
    
    # Generate operation ID for this query
    operation_id = str(uuid.uuid4())
    
    # Setup chat streaming for this operation
    setup_chat_streaming(operation_id)
    
    # Start background processing
    def run_query_background():
        try:
            # Add initial message
            add_chat_message(operation_id, 'system', f'🔍 Searching knowledge graph ({mode} mode)...')
            
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
                add_chat_message(operation_id, 'error', error)
            else:
                add_chat_message(operation_id, 'assistant', result, {'mode': mode})
                add_chat_message(operation_id, 'system', '✅ Query completed successfully!')
            mark_chat_complete(operation_id)
                
        except Exception as e:
            add_chat_message(operation_id, 'error', f'Query failed: {str(e)}')
            mark_chat_complete(operation_id)
    
    # Start processing in background thread
    threading.Thread(target=run_query_background, daemon=True).start()
    
    return jsonify({
        'success': True,
        'operation_id': operation_id,
        'question': question,
        'mode': mode,
        'chat_endpoint': f'/api/chat/{operation_id}'
    })

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
    stats = {'documents': 0, 'chunks': 0}
    debug_info = {'working_dir': str(working_dir), 'files_found': []}
    
    try:
        # List all files in working directory for debugging
        if working_dir.exists():
            debug_info['files_found'] = [f.name for f in working_dir.iterdir() if f.is_file()]
        
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
                
    except Exception as e:
        debug_info['error'] = str(e)
    
    return jsonify({
        'session': session_name,
        'has_knowledge_graph': has_knowledge_graph,
        'stats': stats,
        'debug': debug_info  # Temporary debug info
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

def setup_chat_streaming(operation_id: str):
    """Setup chat streaming for a specific operation"""
    # Initialize chat stream for this operation
    chat_streams[operation_id] = {'messages': [], 'complete': False}
    
    # Create and configure handler
    handler = ChatLogHandler(operation_id)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(name)s] %(message)s')
    handler.setFormatter(formatter)
    chat_handlers[operation_id] = handler
    
    # Add handler to nano-graphrag loggers
    loggers_to_capture = [
        'nano-graphrag',
        'nano-vectordb', 
        'google_genai._api_client'
    ]
    
    for logger_name in loggers_to_capture:
        logger = logging.getLogger(logger_name)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

def add_chat_message(operation_id: str, message_type: str, content: str, metadata: dict = None):
    """Add a message to the chat stream"""
    if operation_id in chat_streams:
        message = {
            'type': message_type,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        chat_streams[operation_id]['messages'].append(message)

def mark_chat_complete(operation_id: str):
    """Mark chat stream as complete"""
    if operation_id in chat_streams:
        chat_streams[operation_id]['complete'] = True
        cleanup_chat_streaming(operation_id)

@app.route('/api/chat/<operation_id>')
def get_chat_messages(operation_id: str):
    """Get chat messages for an operation"""
    if operation_id not in chat_streams:
        return jsonify({'error': 'Operation not found'}), 404
    
    stream_data = chat_streams[operation_id]
    return jsonify({
        'messages': stream_data['messages'],
        'complete': stream_data['complete']
    })

def cleanup_chat_streaming(operation_id: str):
    """Clean up chat streaming resources"""
    if operation_id in chat_handlers:
        handler = chat_handlers[operation_id]
        
        # Remove handler from all loggers
        loggers_to_cleanup = [
            'nano-graphrag',
            'nano-vectordb',
            'google_genai._api_client'
        ]
        
        for logger_name in loggers_to_cleanup:
            logger = logging.getLogger(logger_name)
            logger.removeHandler(handler)
        
        del chat_handlers[operation_id]
    
    # Keep chat_streams data for a while for retrieval
    # It will be cleaned up naturally or by periodic cleanup

# Health check endpoint for Cloud Run
@app.route('/health')
def health_check():
    """Health check endpoint for Cloud Run"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    # Check config
    if not config.has_openai_config and not config.has_azure_openai_config:
        print("Warning: No OpenAI API configuration found.")
        print("Please set OPENAI_API_KEY in your environment or .env file.")
    
    # Configure port for Cloud Run (uses PORT env variable)
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV', 'development') == 'development'
    
    # Run the app
    app.run(debug=debug_mode, host='0.0.0.0', port=port)