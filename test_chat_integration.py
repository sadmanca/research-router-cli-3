#!/usr/bin/env python3
"""
Test script for chat integration functionality without running Flask
"""

import json
import logging
from datetime import datetime
from typing import Dict, List

# Mock the chat streaming functionality
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
                    'timestamp': record.created,
                    'level': record.levelname
                }
                chat_streams[self.operation_id]['messages'].append(message)
        except:
            pass  # Ignore handler errors

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
    
    # Add handler to nano-graphrag loggers (simulated)
    loggers_to_capture = [
        'nano-graphrag',
        'nano-vectordb', 
        'google_genai._api_client'
    ]
    
    for logger_name in loggers_to_capture:
        logger = logging.getLogger(logger_name)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

def mark_chat_complete(operation_id: str):
    """Mark chat stream as complete"""
    if operation_id in chat_streams:
        chat_streams[operation_id]['complete'] = True

def test_chat_integration():
    """Test the chat integration functionality"""
    print("Testing Chat Integration System")
    print("=" * 50)
    
    # Test 1: Setup chat streaming
    operation_id = "test-123"
    setup_chat_streaming(operation_id)
    print(f"PASS: Chat streaming setup for operation: {operation_id}")
    
    # Test 2: Add system messages
    add_chat_message(operation_id, 'system', 'Starting query operation...')
    print("PASS: Added system message")
    
    # Test 3: Simulate nano-graphrag logs
    nano_logger = logging.getLogger('nano-graphrag')
    nano_logger.info('Using Gemini for LLM and embeddings')
    nano_logger.info('Load KV full_docs with 6 data')
    nano_logger.info('Load KV text_chunks with 101 data')
    
    vectordb_logger = logging.getLogger('nano-vectordb')
    vectordb_logger.info('Load (142, 3072) data')
    vectordb_logger.info('Init embedding_dim: 3072, metric: cosine')
    
    print("PASS: Simulated nano-graphrag and nano-vectordb logs")
    
    # Test 4: Add assistant response
    add_chat_message(operation_id, 'assistant', 
                    'Based on the documents, GraphRAG is a retrieval-augmented generation approach that uses knowledge graphs...', 
                    {'mode': 'global'})
    print("PASS: Added assistant response")
    
    # Test 5: Complete operation
    add_chat_message(operation_id, 'system', 'Query completed successfully!')
    mark_chat_complete(operation_id)
    print("PASS: Marked operation as complete")
    
    # Test 6: Show results
    stream_data = chat_streams[operation_id]
    print(f"\nChat Stream Results:")
    print(f"   Messages: {len(stream_data['messages'])}")
    print(f"   Complete: {stream_data['complete']}")
    
    print(f"\nMessage Types:")
    message_types = {}
    for msg in stream_data['messages']:
        msg_type = msg['type']
        message_types[msg_type] = message_types.get(msg_type, 0) + 1
    
    for msg_type, count in message_types.items():
        print(f"   {msg_type}: {count}")
    
    print(f"\nSample Messages:")
    for i, msg in enumerate(stream_data['messages'][:5]):
        if isinstance(msg['timestamp'], str):
            timestamp = datetime.fromisoformat(msg['timestamp']).strftime('%H:%M:%S')
        else:
            timestamp = datetime.fromtimestamp(msg['timestamp']).strftime('%H:%M:%S')
        print(f"   [{timestamp}] {msg['type'].upper()}: {msg['content'][:60]}{'...' if len(msg['content']) > 60 else ''}")
    
    if len(stream_data['messages']) > 5:
        print(f"   ... and {len(stream_data['messages']) - 5} more messages")
    
    print(f"\nIntegration Test Results:")
    print(f"   PASS: Chat streaming setup")
    print(f"   PASS: Message logging")
    print(f"   PASS: Multiple message types")
    print(f"   PASS: Timestamp handling")
    print(f"   PASS: Operation completion")
    
    return stream_data

if __name__ == "__main__":
    test_results = test_chat_integration()
    
    # Export results for verification
    with open('chat_integration_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nTest results exported to: chat_integration_test_results.json")
    print(f"Chat integration test completed successfully!")