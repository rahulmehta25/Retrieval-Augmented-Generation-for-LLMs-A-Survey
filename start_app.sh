#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting RAG Application...${NC}"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command_exists python3; then
    echo -e "${RED}Error: Python is not installed${NC}"
    exit 1
fi

if ! command_exists node; then
    echo -e "${RED}Error: Node.js is not installed${NC}"
    exit 1
fi

if ! command_exists ollama; then
    echo -e "${RED}Error: Ollama is not installed${NC}"
    exit 1
fi

# Check if Ollama is running
echo -e "${YELLOW}Checking Ollama status...${NC}"
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo -e "${YELLOW}Ollama is not running. Please start it with: ollama serve${NC}"
    echo -e "${YELLOW}Starting Ollama in background...${NC}"
    ollama serve >/dev/null 2>&1 &
    sleep 3
fi

# Check if gemma:2b model exists
if ! ollama list | grep -q "gemma:2b"; then
    echo -e "${YELLOW}Pulling gemma:2b model...${NC}"
    ollama pull gemma:2b
fi

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down servers...${NC}"
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
    fi
    exit 0
}

# Set trap for cleanup
trap cleanup INT TERM

# Start backend
echo -e "${GREEN}Starting backend server...${NC}"
cd "/Users/rahulmehta/Desktop/RAG for LLMs-  A Survey/rag-from-scratch"

# Create uploaded_documents directory if it doesn't exist
mkdir -p uploaded_documents

# Start the backend
python3 api_server.py &
BACKEND_PID=$!

# Wait for backend to start
echo -e "${YELLOW}Waiting for backend to start...${NC}"
for i in {1..10}; do
    if curl -s http://localhost:5000/api/health >/dev/null 2>&1; then
        echo -e "${GREEN}Backend is ready!${NC}"
        break
    fi
    sleep 1
done

# Start frontend
echo -e "${GREEN}Starting frontend...${NC}"
cd "/Users/rahulmehta/Desktop/RAG for LLMs-  A Survey/glass-scroll-scribe"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    npm install
fi

# Start the frontend
npm run dev &
FRONTEND_PID=$!

# Wait for frontend to start
echo -e "${YELLOW}Waiting for frontend to start...${NC}"
sleep 5

echo -e "\n${GREEN}✅ RAG Application is running!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📱 Frontend: http://localhost:5173${NC}"
echo -e "${GREEN}🔧 Backend API: http://localhost:5000${NC}"
echo -e "${GREEN}📚 API Docs: http://localhost:5000/docs${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Default credentials:${NC}"
echo -e "  Username: demo"
echo -e "  Password: password123"
echo -e "\n${YELLOW}Press Ctrl+C to stop all servers${NC}\n"

# Keep the script running
wait