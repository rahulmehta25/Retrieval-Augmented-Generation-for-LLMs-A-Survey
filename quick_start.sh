#!/bin/bash

echo "🚀 RAG Application Quick Start"
echo "=============================="

# Start backend in background
echo "Starting backend server..."
cd "/Users/rahulmehta/Desktop/RAG for LLMs-  A Survey/rag-from-scratch"
python3 api_server.py > backend.log 2>&1 &
BACKEND_PID=$!

# Give backend time to start
sleep 5

# Start frontend in background  
echo "Starting frontend..."
cd "/Users/rahulmehta/Desktop/RAG for LLMs-  A Survey/glass-scroll-scribe"
npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!

# Give frontend time to start
sleep 5

echo ""
echo "✅ Application should be running!"
echo "================================="
echo "📱 Frontend: http://localhost:5173"
echo "🔧 Backend: http://localhost:5000"
echo "📚 API Docs: http://localhost:5000/docs"
echo ""
echo "Default login: demo / password123"
echo ""
echo "PIDs: Backend=$BACKEND_PID, Frontend=$FRONTEND_PID"
echo ""
echo "To stop: kill $BACKEND_PID $FRONTEND_PID"
echo ""
echo "Logs: Check backend.log and frontend.log if there are issues"