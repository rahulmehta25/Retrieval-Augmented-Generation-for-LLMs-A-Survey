# RAG Application - Full Stack Setup

This document explains how to run the complete RAG application with both backend and frontend.

## Prerequisites

1. **Backend Requirements:**
   - Python 3.8+
   - Ollama running with `gemma:2b` model
   - All dependencies from `requirements.txt` and `requirements_api.txt`

2. **Frontend Requirements:**
   - Node.js 16+
   - npm or yarn

## Installation

### Backend Setup
```bash
cd rag-from-scratch

# Install RAG system dependencies
pip install -r requirements.txt

# Install API server dependencies
pip install -r requirements_api.txt

# Make sure Ollama is running
ollama serve

# Pull the model if not already available
ollama pull gemma:2b
```

### Frontend Setup
```bash
cd glass-scroll-scribe

# Install dependencies
npm install
```

## Running the Application

### Option 1: Using the run script (Recommended)
```bash
cd rag-from-scratch
./run_app.sh
```

### Option 2: Manual startup

**Terminal 1 - Backend:**
```bash
cd rag-from-scratch
python api_server.py
```

**Terminal 2 - Frontend:**
```bash
cd glass-scroll-scribe
npm run dev
```

## Access the Application

- Frontend: http://localhost:5173
- Backend API: http://localhost:5000
- API Documentation: http://localhost:5000/docs

## Default Credentials

- Username: `demo`
- Password: `password123`

## Features

1. **Document Upload**: Upload PDF, TXT, or Markdown files
2. **Chat Interface**: Ask questions about uploaded documents
3. **Document Management**: View and delete uploaded documents
4. **Authentication**: Login/Register functionality

## Troubleshooting

1. **Port conflicts**: If ports 5000 or 5173 are in use, modify:
   - Backend: Edit `api_server.py` line with `uvicorn.run`
   - Frontend: Edit `vite.config.ts` to change the port

2. **CORS issues**: The backend is configured to accept requests from localhost:5173. If using a different port, update the CORS settings in `api_server.py`.

3. **Ollama not running**: Make sure Ollama is running with `ollama serve` before starting the backend.

4. **Model not found**: Pull the required model with `ollama pull gemma:2b`