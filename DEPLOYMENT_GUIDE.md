# RAG Application Deployment Guide

## Quick Start

1. **Start the application:**
   ```bash
   ./start_app.sh
   ```

2. **Access the application:**
   - Frontend: http://localhost:5173
   - API Docs: http://localhost:5000/docs

3. **Login with default credentials:**
   - Username: `demo`
   - Password: `password123`

## Features Overview

### 1. Authentication
- Login/Register functionality
- Token-based authentication
- Persistent sessions

### 2. Document Management
- Upload PDF, TXT, and Markdown files
- View uploaded documents with metadata
- Delete documents
- Automatic text chunking and indexing

### 3. Chat Interface
- Ask questions about uploaded documents
- Get AI-generated answers with source citations
- Context-aware responses using RAG

### 4. System Status
- View model information
- Check document count
- Monitor system readiness

## Testing the Application

### Step 1: Authentication
1. Open http://localhost:5173
2. Click "Login" and use credentials: demo/password123
3. Or register a new account (min 3 chars username, 6 chars password)

### Step 2: Upload Documents
1. Navigate to "Documents" section
2. Click "Upload Document"
3. Select a PDF, TXT, or MD file
4. Wait for processing confirmation

### Step 3: Ask Questions
1. Go to "Chat" section
2. Type questions about your uploaded documents
3. View AI responses with source references

## Troubleshooting

### Common Issues

1. **"Ollama not running" error:**
   ```bash
   ollama serve
   ```

2. **Model not found:**
   ```bash
   ollama pull gemma:2b
   ```

3. **Port already in use:**
   - Backend (5000): Edit `api_server.py` last line
   - Frontend (5173): Edit `vite.config.ts`

4. **CORS errors:**
   - Check that frontend URL matches CORS config in `api_server.py`

5. **Import errors:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements_api.txt
   ```

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   React UI      │────▶│  FastAPI Backend │────▶│   RAG System    │
│  (Port 5173)    │     │   (Port 5000)   │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │                          │
                                ▼                          ▼
                        ┌──────────────┐          ┌──────────────┐
                        │   Storage    │          │    Ollama    │
                        │  Documents   │          │  (gemma:2b)  │
                        └──────────────┘          └──────────────┘
```

## API Endpoints

- `POST /api/auth/login` - User login
- `POST /api/auth/register` - User registration
- `POST /api/documents/upload` - Upload document
- `GET /api/documents` - List documents
- `DELETE /api/documents/{id}` - Delete document
- `POST /api/chat/query` - Query RAG system
- `GET /api/status` - System status
- `GET /api/health` - Health check

## Performance Tips

1. **Document Processing:**
   - Smaller chunk sizes = more precise answers
   - Larger chunk sizes = better context understanding
   - Default: 500 chars with 50 char overlap

2. **Model Selection:**
   - `gemma:2b` - Fast, lightweight (1.7GB RAM)
   - `gemma:7b` - Better quality (5GB RAM)
   - `qwen2.5-coder:7b` - Best for code (4.7GB RAM)

3. **Vector Store:**
   - ChromaDB for development
   - Consider FAISS for production scale

## Security Notes

- Default auth is for demo only
- Add proper JWT validation for production
- Implement rate limiting
- Sanitize file uploads
- Use HTTPS in production