# Research Router CLI

An interactive command-line interface for creating and querying knowledge graphs from PDF documents using nano-graphrag.

## Features

- 📄 **PDF Processing**: Extract text from PDFs and build knowledge graphs
- 🧠 **Multiple Query Modes**: Local, global, and naive RAG queries  
- 💾 **Session Management**: Multiple isolated research sessions
- 🔄 **Interactive Mode**: Persistent CLI that stays running
- ⚡ **Fast Setup**: Uses nano-graphrag for lightweight GraphRAG implementation

## Quick Start

### 1. Install Dependencies

```bash
uv pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=your-api-key-here
```

### 3. Run the CLI

```bash
uv run python main.py
```

## Usage

### Basic Workflow

1. **Create a session**:
   ```
   research-router> session create my_research
   ```

2. **Add PDF documents**:
   ```
   research-router (my_research)> insert document.pdf
   research-router (my_research)> insert papers/
   ```

3. **Query the knowledge graph**:
   ```
   research-router (my_research)> query "What are the main findings?"
   research-router (my_research)> query --mode local "Tell me about methodology"
   ```

### Commands Reference

#### Session Management
- `session create <name>` - Create a new research session
- `session list` - List all sessions
- `session switch <name>` - Switch to a different session
- `session delete <name>` - Delete a session
- `status` - Show current session status

#### Document Management
- `insert <pdf_path>` - Insert a single PDF
- `insert <directory>` - Insert all PDFs from directory
- `insert *.pdf` - Insert PDFs matching pattern

#### Querying
- `query <question>` - Query using global mode (default)
- `query --mode local <question>` - Local search within documents
- `query --mode global <question>` - Global thematic analysis
- `query --mode naive <question>` - Simple RAG without graph
- `iquery` - Start interactive query mode

#### Utilities
- `config` - Show configuration status
- `help` - Show available commands
- `exit` - Exit the CLI

### Query Modes Explained

- **Global Mode**: Best for high-level questions about themes, trends, and overall insights across all documents
- **Local Mode**: Best for specific questions about particular documents or sections
- **Naive Mode**: Traditional RAG without knowledge graph structure

## Session Management

Each session maintains its own:
- Working directory for knowledge graph files
- Isolated knowledge graph data
- Independent document collections

Sessions are stored in `./sessions/` and persist between CLI runs.

## Configuration

The CLI supports both OpenAI and Azure OpenAI:

**OpenAI** (recommended):
```
OPENAI_API_KEY=sk-...
```

**Azure OpenAI**:
```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-key
```

## File Structure

```
research-router-cli/
├── main.py                     # CLI entry point
├── research_router_cli/
│   ├── commands/
│   │   ├── session.py          # Session management
│   │   ├── insert.py           # PDF insertion
│   │   └── query.py            # Knowledge graph querying
│   └── utils/
│       ├── config.py           # Configuration management
│       └── pdf_processor.py    # PDF text extraction
├── sessions/                   # Session data (auto-created)
├── requirements.txt            # Dependencies
└── .env.example               # Environment template
```

## Troubleshooting

**No API key configured**:
- Ensure `.env` file exists with `OPENAI_API_KEY`
- Check `config` command output

**PDF extraction fails**:
- Ensure PDF files are not corrupted or password-protected
- Try with a different PDF to test

**Knowledge graph not found**:
- Use `insert` command to add documents first
- Check `status` command for session info

**Session issues**:
- Use `session list` to see all sessions
- Use `session create` to create a new session

## Dependencies

- **nano-graphrag**: Lightweight GraphRAG implementation
- **typer**: CLI framework
- **rich**: Enhanced terminal output
- **pdfplumber**: PDF text extraction
- **python-dotenv**: Environment variable management