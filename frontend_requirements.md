# Frontend Requirements for RAG System

## Overview
Create a modern web frontend for a Retrieval-Augmented Generation (RAG) system that processes documents and answers questions using local LLMs via Ollama.

## Backend API Endpoints (Flask)
The frontend should connect to these API endpoints:

### 1. POST /api/upload
- Accepts: multipart/form-data with file(s)
- Returns: `{ "status": "success", "message": "Processed X documents", "document_count": X }`

### 2. POST /api/query
- Accepts: `{ "question": "string", "stream": boolean }`
- Returns: 
  - If stream=false: `{ "answer": "string", "sources": ["doc1.pdf:page3", "doc2.txt:line45"] }`
  - If stream=true: Server-Sent Events with chunks

### 3. GET /api/documents
- Returns: `{ "documents": [{"name": "doc1.pdf", "chunks": 25, "upload_date": "2024-01-20"}] }`

### 4. DELETE /api/documents/:id
- Returns: `{ "status": "success" }`

### 5. GET /api/status
- Returns: `{ "model": "gemma:2b", "documents_loaded": 10, "ready": true }`

## UI Components Needed

### 1. Header
- App title: "RAG Knowledge Assistant"
- Status indicator showing model (gemma:2b) and readiness

### 2. Document Management Panel (Left Sidebar)
- Upload button/drag-drop area
- List of uploaded documents with delete option
- Show chunk count per document
- Clear all documents button

### 3. Chat Interface (Main Area)
- Chat history display
- Input field for questions
- Send button
- Streaming response support with typing indicator
- Show sources for each answer (collapsible)

### 4. Settings Panel (Right Sidebar - Collapsible)
- Model selection (show current: gemma:2b)
- Temperature slider (0.0 - 1.0, default 0.7)
- Max tokens input (default 500)
- Stream responses toggle

## Technical Requirements

### Frontend Stack
- React or Vue.js
- Tailwind CSS for styling
- Axios or Fetch API for HTTP requests
- React/Vue component for markdown rendering
- File upload with progress indicator

### Key Features
1. **Responsive Design**: Mobile-friendly
2. **Dark Mode**: Toggle between light/dark themes
3. **Loading States**: Spinners for all async operations
4. **Error Handling**: Toast notifications for errors
5. **Keyboard Shortcuts**: Ctrl/Cmd+Enter to send query

### API Integration Code Structure
```javascript
// Example API service structure
const API_BASE = 'http://localhost:5000/api';

const apiService = {
  uploadDocument: (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return fetch(`${API_BASE}/upload`, {
      method: 'POST',
      body: formData
    });
  },
  
  queryRAG: async (question, stream = true) => {
    if (stream) {
      // Use EventSource for SSE
      return new EventSource(`${API_BASE}/query?question=${encodeURIComponent(question)}`);
    } else {
      return fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, stream: false })
      });
    }
  }
};
```

## UI/UX Guidelines
1. Clean, minimalist design
2. Purple accent color (#8B5CF6) for primary actions
3. Smooth transitions and animations
4. Clear visual hierarchy
5. Accessibility compliant (ARIA labels, keyboard navigation)

## File Structure Suggestion
```
frontend/
├── index.html
├── src/
│   ├── App.vue/jsx
│   ├── components/
│   │   ├── ChatInterface.vue/jsx
│   │   ├── DocumentPanel.vue/jsx
│   │   ├── SettingsPanel.vue/jsx
│   │   └── StatusBar.vue/jsx
│   ├── services/
│   │   └── api.js
│   └── styles/
│       └── main.css
```

## Environment Variables
```
VITE_API_URL=http://localhost:5000/api
```

## Important Notes
- The backend uses Ollama for LLM inference (currently gemma:2b model)
- Documents are chunked and stored in ChromaDB vector database
- The system uses sentence-transformers for embeddings
- Support both file upload and text paste options
- Consider rate limiting on the frontend to prevent API spam

## Bonus Features (Optional)
1. Export chat history as markdown
2. Code syntax highlighting in responses
3. Citation preview on hover
4. Voice input for questions
5. Shareable chat links