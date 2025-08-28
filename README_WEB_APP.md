# Research Router Web App

A simple web interface for nano-graphrag document chat sessions. Upload documents, create knowledge graphs, and chat with your research materials through an intuitive web interface.

## Features

- **Session Management**: Create and switch between different research sessions
- **Document Upload**: Support for PDF, TXT, MD, DOC, and DOCX files
- **Interactive Chat**: Query your documents using different search modes (Global, Local, Naive)
- **Knowledge Graph**: Automatic knowledge graph generation from your documents
- **Session History**: View previous queries and responses
- **Duplicate Detection**: Automatically skip duplicate files
- **Progress Tracking**: Real-time upload and processing progress

## Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# or with uv (recommended)
uv pip install -r requirements.txt
```

### 2. Set Up API Configuration

Create a `.env` file in the project root:

```bash
# OpenAI API (required)
OPENAI_API_KEY=your-openai-api-key-here

# Optional: Flask secret key for sessions
SECRET_KEY=your-secret-key-here
```

### 3. Run the Web App

```bash
# Simple way
python run.py

# Or directly
python web_app.py
```

The web app will be available at: http://localhost:5000

## How to Use

### 1. Create a Session
- Visit the home page
- Enter a name for your research session
- Click "Create Session"

### 2. Upload Documents
- Click "Upload Documents" in the navigation
- Drag and drop files or click to browse
- Supported formats: PDF, TXT, MD, DOC, DOCX
- Click "Upload and Process" to add them to your knowledge graph

### 3. Start Chatting
- Go to the Chat page
- Ask questions about your uploaded documents
- Choose between different query modes:
  - **Global**: Searches across all documents and community summaries
  - **Local**: Focuses on specific text chunks and entities  
  - **Naive**: Simple similarity-based retrieval

## Web App Structure

```
├── web_app.py              # Main Flask application
├── run.py                  # Simple launcher script
├── templates/              # HTML templates
│   ├── base.html          # Base template with navigation
│   ├── index.html         # Home page (session management)
│   ├── create_session.html # Session creation page
│   ├── chat.html          # Chat interface
│   └── upload.html        # Document upload page
├── static/                # Static assets
│   ├── css/
│   │   └── style.css      # Custom styles
│   └── js/
│       └── main.js        # JavaScript utilities
└── sessions/              # Session data (auto-created)
    ├── sessions.json      # Session registry
    └── [session-name]/    # Individual session folders
        ├── *.json         # Knowledge graph files
        └── downloads/     # Downloaded papers (ArXiv)
```

## API Endpoints

The web app provides REST API endpoints:

- `POST /api/sessions` - Create a new session
- `POST /api/sessions/{name}/switch` - Switch to a session
- `GET /api/sessions/{name}/status` - Get session status
- `GET /api/sessions/{name}/history` - Get query history
- `POST /api/upload` - Upload and process documents
- `POST /api/query` - Query the knowledge graph

## Configuration

### Environment Variables

- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `SECRET_KEY` - Flask session secret key (optional, auto-generated)
- `MAX_CONTENT_LENGTH` - Maximum file upload size (default: 50MB)

### Supported File Types

- **PDF**: `.pdf`
- **Text**: `.txt`
- **Markdown**: `.md`
- **Word**: `.doc`, `.docx`

## Deployment

### Development
```bash
python run.py
```

### Production
For production deployment, consider using a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 web_app:app
```

## Differences from CLI

The web app provides the same core functionality as the CLI but with a more user-friendly interface:

| Feature | CLI | Web App |
|---------|-----|---------|
| Session Management | ✅ | ✅ |
| Document Upload | ✅ | ✅ (with drag & drop) |
| Query Interface | ✅ | ✅ (real-time chat) |
| Progress Tracking | ✅ | ✅ (visual progress bars) |
| History | ✅ | ✅ (persistent in browser) |
| ArXiv Integration | ✅ | ❌ (not implemented) |
| File Browser | ✅ | ❌ (web upload instead) |

## Troubleshooting

### Common Issues

1. **No API Key Error**
   - Make sure you have set `OPENAI_API_KEY` in your `.env` file
   - Restart the web app after adding the key

2. **Upload Fails**
   - Check file size (max 50MB)
   - Ensure file format is supported
   - Check browser console for errors

3. **Session Not Found**
   - Make sure you have created a session first
   - Check that the session directory exists

4. **Query Returns No Results**
   - Ensure documents have been uploaded and processed
   - Try different query modes
   - Check that the knowledge graph files exist in the session folder

### Logs and Debugging

- Enable Flask debug mode by setting `debug=True` in `web_app.py`
- Check browser developer console for JavaScript errors
- Monitor the terminal/console for Python errors

## Contributing

The web app reuses existing CLI components from:
- `research_router_cli/commands/session.py` - Session management
- `research_router_cli/commands/insert.py` - Document processing  
- `research_router_cli/commands/query.py` - Knowledge graph querying
- `research_router_cli/utils/` - Various utilities

To add new features, you can extend these existing modules or create new web-specific components.