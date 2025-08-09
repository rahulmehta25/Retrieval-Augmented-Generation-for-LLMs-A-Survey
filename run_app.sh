#!/bin/bash

echo "Starting RAG Application..."

# Start the backend server
echo "Starting backend server..."
cd /Users/rahulmehta/Desktop/RAG\ for\ LLMs-\ \ A\ Survey/rag-from-scratch
python api_server.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start the frontend
echo "Starting frontend..."
cd /Users/rahulmehta/Desktop/RAG\ for\ LLMs-\ \ A\ Survey/glass-scroll-scribe
npm run dev &
FRONTEND_PID=$!

echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"

echo ""
echo "RAG Application is running!"
echo "Frontend: http://localhost:5173"
echo "Backend API: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for user to press Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait