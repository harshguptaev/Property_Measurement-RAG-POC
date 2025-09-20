#!/bin/bash

# Property Measurement RAG System - Assistant UI Startup Script

echo "🏠 Starting Property Measurement RAG System with Assistant UI..."

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: Please run this script from the root directory of the Property_Measurement-RAG-POC project"
    exit 1
fi

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to kill process on port
kill_port() {
    if check_port $1; then
        echo "🔄 Killing existing process on port $1..."
        lsof -ti:$1 | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
}

# Kill existing processes
kill_port 8000  # FastAPI backend
kill_port 3000  # Next.js frontend

echo ""
echo "📋 System Requirements Check:"

# Check Python
if command -v python3 &> /dev/null; then
    echo "✅ Python3 found: $(python3 --version)"
else
    echo "❌ Python3 not found. Please install Python 3.9+"
    exit 1
fi

# Check Node.js
if command -v node &> /dev/null; then
    echo "✅ Node.js found: $(node --version)"
else
    echo "❌ Node.js not found. Please install Node.js 18+"
    exit 1
fi

# Check npm
if command -v npm &> /dev/null; then
    echo "✅ npm found: $(npm --version)"
else
    echo "❌ npm not found. Please install npm"
    exit 1
fi

echo ""
echo "🔧 Installing Dependencies..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt -q
pip install fastapi uvicorn pydantic python-multipart -q

# Install Node.js dependencies for frontend
echo "📦 Installing Node.js dependencies..."
cd assistant-ui
npm install -q
cd ..

echo ""
echo "🚀 Starting Services..."

# Start FastAPI backend
echo "🔧 Starting FastAPI backend on port 8000..."
python3 api_server.py &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 5

# Check if backend is running
if ! check_port 8000; then
    echo "❌ Backend failed to start on port 8000"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

echo "✅ Backend started successfully"

# Start Next.js frontend
echo "🌐 Starting Next.js frontend on port 3000..."
cd assistant-ui
npm run dev &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
echo "⏳ Waiting for frontend to start..."
sleep 10

# Check if frontend is running
if ! check_port 3000; then
    echo "❌ Frontend failed to start on port 3000"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 1
fi

echo ""
echo "🎉 Property Measurement RAG System is ready!"
echo ""
echo "📱 Frontend (Assistant UI): http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo ""
echo "💡 Features available:"
echo "   • Modern chat interface with Assistant UI"
echo "   • Real-time property document analysis"
echo "   • Interactive conversation threads"
echo "   • Property-specific analysis tools"
echo "   • Document statistics dashboard"
echo ""
echo "🛑 Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    kill_port 8000
    kill_port 3000
    echo "👋 Services stopped. Goodbye!"
    exit 0
}

# Set trap to cleanup on exit
trap cleanup SIGINT SIGTERM

# Wait for user to stop
while true; do
    if ! check_port 8000; then
        echo "❌ Backend stopped unexpectedly"
        break
    fi
    if ! check_port 3000; then
        echo "❌ Frontend stopped unexpectedly"
        break
    fi
    sleep 5
done

cleanup