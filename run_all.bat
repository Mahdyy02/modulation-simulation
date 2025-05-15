@echo off
REM Start FastAPI backend
start cmd /k "uvicorn backend:app --reload"
REM Wait a bit for backend to start
ping 127.0.0.1 -n 3 > nul
REM Start frontend (assume using npm or yarn)
cd front-simulation
start cmd /k "npm run dev"
REM Wait a bit for frontend to start
ping 127.0.0.1 -n 3 > nul
REM Open frontend in default browser
start http://localhost:5173
cd .. 