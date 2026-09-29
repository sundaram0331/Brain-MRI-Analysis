#!/bin/bash
# NeonCortrex AI - Launch Mobile Servers
LOCAL_IP="10.176.98.158"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================================="
echo "⚡ NEONCORTREX AI — MOBILE ANDROID DEV RUNNER"
echo "=========================================================="
echo "Detected Mac Wi-Fi IP: http://$LOCAL_IP:8000"
echo ""

echo "Option 1: Launch Backend + Instant Android PWA Web App"
echo "Option 2: Launch Dedicated React Native (Expo) Mobile App"
echo "=========================================================="

read -p "Select option (1 or 2, default is 1): " choice
choice=${choice:-1}

if [ "$choice" = "1" ]; then
    echo "Starting Backend on 0.0.0.0:8000..."
    cd "$DIR/backend" && python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    
    echo "Starting Frontend on 0.0.0.0:5173..."
    cd "$DIR/frontend" && npm run dev -- --host 0.0.0.0 &
    FRONT_PID=$!
    
    echo ""
    echo ">> Ready! On your Android phone (connected to same Wi-Fi):"
    echo ">> Open Chrome and visit: http://$LOCAL_IP:5173"
    echo ">> Tap Chrome Menu (3 dots) -> 'Install App'"
    echo ""
    wait $BACKEND_PID $FRONT_PID
else
    echo "Starting Backend on 0.0.0.0:8000..."
    cd "$DIR/backend" && python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    
    echo "Starting Expo Server..."
    cd "$DIR/mobile" && npx expo start &
    EXPO_PID=$!
    
    wait $BACKEND_PID $EXPO_PID
fi
