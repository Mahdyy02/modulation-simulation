# Digital Modulation Simulator with SRRC Filter

This project is an interactive digital communication system simulator that allows users to analyze and visualize different digital modulation schemes (BPSK, QPSK, and QAM) with Square Root Raised Cosine (SRRC) pulse shaping. The simulator provides detailed visualizations of constellation diagrams, eye diagrams, and BER performance curves.

## Features

- Support for multiple modulation schemes:
  - BPSK (Binary Phase Shift Keying)
  - QPSK (Quadrature Phase Shift Keying)
  - QAM (Quadrature Amplitude Modulation) with configurable orders (4, 16, 64, 256, 1024)
- Interactive parameter adjustment:
  - Number of symbols
  - Eb/N0 ratio
  - Roll-off factor (α)
  - Samples per symbol
  - Filter span
- Comprehensive visualizations:
  - SRRC filter response
  - Constellation diagrams at different stages
  - Eye diagrams
  - BER vs Eb/N0 curves
- Real-time performance metrics
- Detailed bit comparison analysis

## Prerequisites

### Python Requirements
- Python 3.8 or higher
- Required Python packages:
  ```
  fastapi
  uvicorn
  numpy
  scipy
  matplotlib
  pydantic
  ```

### Node.js Requirements
- Node.js 14.0 or higher
- npm (Node Package Manager)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd digital-modulation-simulator
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Node.js dependencies**
   ```bash
   cd front-simulation
   npm install
   ```

## Running the Application

### Windows
1. Run the backend server:
   ```bash
   run_backend.bat
   ```
2. In a new terminal, run the frontend:
   ```bash
   run_frontend.bat
   ```

### Linux/Mac
1. Run the backend server:
   ```bash
   ./run_backend.sh
   ```
2. In a new terminal, run the frontend:
   ```bash
   ./run_frontend.sh
   ```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

## Usage

1. Select the desired modulation scheme (BPSK, QPSK, or QAM)
2. Adjust the simulation parameters:
   - Number of symbols (K)
   - Eb/N0 ratio in dB
   - Roll-off factor (α)
   - Samples per symbol
   - Filter span
3. Click "Run Simulation" to see the results
4. Use the "Calculate BER Curve" button to generate BER vs Eb/N0 performance curves

## Technical Details

The simulator implements:
- Square Root Raised Cosine (SRRC) pulse shaping
- Matched filtering at the receiver
- AWGN channel simulation
- Theoretical BER calculations for each modulation scheme
- Real-time visualization of signal processing stages

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 