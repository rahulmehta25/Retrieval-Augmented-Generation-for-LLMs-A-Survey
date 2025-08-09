#!/bin/bash

echo "🚀 RAG Application Complete Deployment"
echo "====================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Kill any existing processes on our ports
echo -e "${YELLOW}Cleaning up existing processes...${NC}"
lsof -ti:5000 | xargs kill -9 2>/dev/null || true
lsof -ti:5173 | xargs kill -9 2>/dev/null || true
lsof -ti:8083 | xargs kill -9 2>/dev/null || true

# Backend setup
echo -e "\n${GREEN}Setting up backend...${NC}"
cd "/Users/rahulmehta/Desktop/RAG for LLMs-  A Survey/rag-from-scratch"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment and install dependencies
echo -e "${YELLOW}Installing backend dependencies...${NC}"
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_api.txt

# Create necessary directories
mkdir -p uploaded_documents
mkdir -p chroma_db
mkdir -p embedding_cache

# Start backend
echo -e "${GREEN}Starting backend server...${NC}"
python api_server.py > backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to start
echo -e "${YELLOW}Waiting for backend to start...${NC}"
for i in {1..20}; do
    if curl -s http://localhost:5000/api/health >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Backend is ready!${NC}"
        break
    fi
    sleep 1
done

# Frontend setup
echo -e "\n${GREEN}Setting up frontend...${NC}"
cd "/Users/rahulmehta/Desktop/RAG for LLMs-  A Survey/glass-scroll-scribe"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    npm install
fi

# Update the API URL in the frontend to use the correct port
echo -e "${YELLOW}Configuring frontend...${NC}"
echo "VITE_API_URL=http://localhost:5000/api" > .env

# Kill any process using port 5173
lsof -ti:5173 | xargs kill -9 2>/dev/null || true

# Start frontend on the correct port
echo -e "${GREEN}Starting frontend...${NC}"
npm run dev -- --port 5173 > frontend.log 2>&1 &
FRONTEND_PID=$!

# Wait for frontend to start
echo -e "${YELLOW}Waiting for frontend to start...${NC}"
sleep 5

# Final check
if curl -s http://localhost:5000/api/health >/dev/null 2>&1; then
    BACKEND_STATUS="${GREEN}✅ Running${NC}"
else
    BACKEND_STATUS="${RED}❌ Not running${NC}"
fi

if curl -s http://localhost:5173 >/dev/null 2>&1; then
    FRONTEND_STATUS="${GREEN}✅ Running${NC}"
else
    FRONTEND_STATUS="${RED}❌ Not running (check port 8083)${NC}"
fi

echo -e "\n${GREEN}=====================================_${NC}"
echo -e "${GREEN}🎉 Deployment Complete!${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "Backend Status: $BACKEND_STATUS"
echo -e "Frontend Status: $FRONTEND_STATUS"
echo -e "\n${GREEN}Access Points:${NC}"
echo -e "📱 Frontend: http://localhost:5173 (or http://localhost:8083)"
echo -e "🔧 Backend API: http://localhost:5000"
echo -e "📚 API Docs: http://localhost:5000/docs"
echo -e "\n${GREEN}Credentials:${NC}"
echo -e "Username: demo"
echo -e "Password: password123"
echo -e "\n${YELLOW}Process IDs:${NC}"
echo -e "Backend PID: $BACKEND_PID"
echo -e "Frontend PID: $FRONTEND_PID"
echo -e "\n${YELLOW}To stop all services:${NC}"
echo -e "kill $BACKEND_PID $FRONTEND_PID"
echo -e "\n${YELLOW}Check logs if needed:${NC}"
echo -e "Backend: tail -f backend.log"
echo -e "Frontend: tail -f glass-scroll-scribe/frontend.log"