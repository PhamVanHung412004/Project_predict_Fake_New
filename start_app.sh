#!/bin/bash

echo "🚀 Starting Fake News Detection Application"
echo "=========================================="

# Kiểm tra virtual environment
if [ ! -d "env_fake_new" ]; then
    echo "❌ Virtual environment not found. Please run the training first."
    exit 1
fi

# Kích hoạt virtual environment
echo "📦 Activating virtual environment..."
source env_fake_new/bin/activate

# Cài đặt dependencies cho backend
echo "📥 Installing backend dependencies..."
pip install -r backend/requirements.txt

# Cài đặt dependencies cho frontend
echo "📥 Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo ""
echo "🎯 Starting services..."
echo ""

# Chạy backend trong background
echo "🔧 Starting Backend (Flask API)..."
cd backend
python app.py &
BACKEND_PID=$!
cd ..

# Đợi backend khởi động
echo "⏳ Waiting for backend to start..."
sleep 5

# Chạy frontend
echo "🌐 Starting Frontend (React)..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

echo ""
echo "🎉 Application is starting!"
echo ""
echo "📱 Access your application:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:5000"
echo ""
echo "🛑 To stop the application, press Ctrl+C"
echo ""

# Function để cleanup khi thoát
cleanup() {
    echo ""
    echo "🛑 Stopping application..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Application stopped."
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT

# Đợi cho đến khi user nhấn Ctrl+C
wait
