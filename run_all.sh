#!/bin/bash
# Start FastAPI backend
uvicorn backend:app --reload &
BACKEND_PID=$!
# Wait a bit for backend to start
sleep 3
# Start frontend (assume using npm or yarn)
cd front-simulation
npm run dev &
FRONTEND_PID=$!
# Wait a bit for frontend to start
sleep 3
# Open frontend in default browser
xdg-open http://localhost:5173 &
cd .. 