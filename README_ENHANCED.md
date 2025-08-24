# Research Router CLI - Enhanced 🚀

An **enhanced** interactive command-line interface for creating and querying knowledge graphs from PDF documents using nano-graphrag, now with **dramatically improved user experience**.

## ✨ Enhanced Features (NEW!)

### 🎯 **Smart Command System**
- **Tab completion** for commands, file paths, and session names
- **Fuzzy command matching** - type partial commands (e.g., `quer` → `query`)
- **Command aliases** for faster typing (`q` for query, `s` for session, `i` for insert)
- **Command history** with ↑↓ arrow key navigation
- **Smart error recovery** with helpful suggestions

### 🗂️ **Interactive File Browser**
- **No more typing exact paths!** Use `insert browse` for visual file selection
- Navigate directories with simple number keys
- Multi-select files with visual confirmation
- Shows file sizes, dates, and types
- Quick path completion and common folder shortcuts

### 🧙 **ArXiv Search Wizard**
- **Interactive search builder** with `arxiv wizard`
- **Smart paper ranking** based on relevance
- **Enhanced search results** with better formatting
- **Bulk download** with progress tracking
- **Interactive paper selection** with previews

### 🎨 **Better Visual Experience**
- **Rich status display** with session health indicators
- **Enhanced progress bars** with time estimates
- **Contextual suggestions** based on your current state
- **Smart help system** that shows relevant commands
- **Color-coded feedback** for different types of messages

### 🚀 **Workflow Improvements**
- **First-time setup wizard** for new users
- **Smart defaults** and session-aware suggestions
- **Better error messages** with specific solutions
- **Quick confirmation prompts** for destructive operations
- **Session indicators** showing knowledge graph status (🧠 vs 📄)

## 🆚 Before vs After

### Before (Original):
```bash
research-router (no session)> insert /long/path/to/my/document.pdf
research-router (no session)> arxiv search "machine learning transformers"
research-router (my_session)> query what are the main findings
```

### After (Enhanced):
```bash
research-router (no session)> i browse    # Interactive file selection!
research-router (no session)> arx wizard  # Guided search experience!
research-router (🧠 my_session)> q main findings  # Smart completion & aliases!
```

## 📦 Quick Setup

### Option 1: Automated Setup (Recommended)
```bash
# Run the enhanced setup script
python setup_enhanced.py
```

### Option 2: Manual Setup
```bash
# Backup original and install enhanced version
cp main.py main_original.py
cp main_enhanced.py main.py

# Install enhanced dependencies
pip install -r requirements.txt
```

## 🎮 Getting Started with Enhanced Features

### 1. **Smart Session Management**
```bash
research-router (no session)> s create    # Prompts for session name
research-router (no session)> s list      # Shows all sessions with status
research-router (no session)> s switch my_research
```

### 2. **Easy File Management**
```bash
# Interactive file browser - no more path typing!
research-router (📄 my_session)> insert browse

# Or use tab completion for paths
research-router (📄 my_session)> insert ~/Downloads/[TAB]

# Quick selection from common folders
research-router (📄 my_session)> insert
# Shows: 1. Desktop  2. Downloads  3. Documents...
```

### 3. **Enhanced ArXiv Integration**
```bash
# Interactive search wizard
research-router (📄 my_session)> arxiv wizard
# Guides you through: topic, keywords, time filters, paper types

# Smart search with ranking
research-router (📄 my_session)> arxiv search "graph neural networks"
# Shows papers ranked by relevance with 🔥 for recent papers
```

### 4. **Intelligent Querying**
```bash
# Query with smart suggestions
research-router (🧠 my_session)> q methodology    # Fuzzy matching
research-router (🧠 my_session)> query --mode local findings  # Tab completion

# Interactive query mode with better UX
research-router (🧠 my_session)> iquery
```

## 🎯 Command Aliases & Shortcuts

| Original | Aliases | Description |
|----------|---------|-------------|
| `session` | `s`, `sess` | Session management |
| `insert` | `i`, `add` | Add documents |
| `query` | `q`, `search` | Search knowledge |
| `iquery` | `iq` | Interactive queries |
| `arxiv` | `arx`, `paper` | ArXiv operations |
| `history` | `hist`, `h` | File history |
| `duplicates` | `dups`, `dup` | Find duplicates |
| `status` | `stat`, `info` | Session status |
| `config` | `cfg`, `conf` | Configuration |
| `help` | `?`, `h` | Get help |
| `exit` | `quit`, `bye` | Exit CLI |

## 💡 Pro Tips for Enhanced Workflow

### 🔄 **Command History Power-User Tips**
- Use ↑↓ arrows to navigate command history
- History persists between sessions
- Recent commands are suggested contextually

### 📁 **File Selection Mastery**
```bash
# Quick paths without typing
insert                    # Shows common folders
insert ?                  # Opens file browser
insert ~/Doc[TAB]        # Auto-completes to ~/Documents/
```

### 🔍 **ArXiv Search Pro Mode**
```bash
# Build complex searches easily
arxiv wizard
# Creates queries like: "graph neural networks" AND (survey OR review) AND recent

# Quick relevance-ranked search
arxiv search gnn attention    # Shows papers ranked by relevance
```

### ❓ **Smart Help System**
```bash
# Context-aware help
help query                # Shows query-specific help
help                      # Shows general help with current context
status                    # Smart suggestions based on your session state
```

### 🎨 **Visual Indicators**
- `🧠` in prompt = Knowledge graph ready
- `📄` in prompt = Session active, no knowledge graph yet
- `🔥` in search results = Recent papers
- Color-coded status messages (green=success, yellow=warning, red=error)

## 📊 Enhanced Status Dashboard

The new `status` command shows much more information:

```bash
research-router (🧠 my_session)> status

Research Router Status
┌─────────────────────┬─────────────────────────────────┐
│ Current Session     │ my_session                      │
│ Working Directory   │ ./sessions/my_session           │
│ Knowledge Graph     │ ✓ Available                     │
│ Graph Statistics    │ 15 documents, 342 chunks       │
│ OpenAI API         │ ✓ Configured                    │
│ Total Sessions     │ 3                               │
└─────────────────────┴─────────────────────────────────┘

💡 Suggestions:
  💡 Query your knowledge graph: query 'your question'
  💡 Try interactive mode: iquery
```

## 🔧 Configuration & Troubleshooting

### **Enhanced Error Messages**
The CLI now provides specific solutions for common issues:

```bash
research-router (no session)> query test
❌ Command 'query' requires an active session. Use 'session create <name>' first.

research-router (📄 my_session)> quer test
⚠️  Did you mean 'query'?

research-router (🧠 my_session)> insert /nonexistent/file.pdf
❌ File not found: /nonexistent/file.pdf
💡 Try: insert browse (for interactive selection)
```

### **Session Health Indicators**
- `(no session)` - Need to create/switch to a session
- `(📄 session_name)` - Session active, no knowledge graph
- `(🧠 session_name)` - Session with working knowledge graph

## 📈 Performance & Reliability

### **Enhanced Progress Tracking**
- Real-time progress bars with time estimates
- Network-aware download progress
- Better error recovery during batch operations
- Processing statistics and summaries

### **Smart Caching**
- ArXiv search results cached for faster access
- Command history with intelligent suggestions
- Session state awareness for better performance

## 🔄 Migration & Compatibility

### **Seamless Migration**
- All existing sessions and data are preserved
- Original functionality remains unchanged
- Can revert to original version anytime: `mv main_original.py main.py`

### **Backward Compatibility**
- All original commands work exactly the same
- Enhanced features are additive, not breaking
- Existing workflows continue to work

## 🐛 Troubleshooting Enhanced Features

### **Tab Completion Not Working**
```bash
# On Windows, install readline alternative
pip install prompt-toolkit

# On Unix/Linux/Mac, readline should work out of the box
```

### **File Browser Issues**
```bash
# If file browser doesn't work, fall back to manual paths
insert /path/to/file.pdf          # Still works!
insert ?                          # Alternative browser trigger
```

### **Command History Problems**
```bash
# History file location: ~/.research_router_history
# Delete it to reset: rm ~/.research_router_history
```

## 🆘 Getting Help

### **Contextual Help System**
- `help` - General help with current context
- `help <command>` - Specific command help
- `status` - Current state and suggestions
- `?` - Quick help alias

### **Error Recovery**
The enhanced CLI is designed to be forgiving:
- Typos are automatically corrected
- Missing arguments prompt for input
- Failed operations suggest alternatives
- Context-aware error messages

## 🎯 What's Next?

The enhanced CLI provides a **dramatically better user experience** while maintaining all the power of the original. Key improvements:

✅ **No more typing exact file paths** - use the file browser  
✅ **No more memorizing commands** - fuzzy matching and aliases  
✅ **No more guessing parameters** - smart suggestions and help  
✅ **No more waiting without feedback** - enhanced progress tracking  
✅ **No more cryptic errors** - helpful error messages with solutions  

**Try it now:** `python setup_enhanced.py` and experience the difference! 🚀