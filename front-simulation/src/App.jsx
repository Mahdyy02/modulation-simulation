import { useState } from 'react';
import { ArrowRight, RefreshCcw, Play, ChevronDown, ChevronUp } from 'lucide-react';

export default function DigitalModulationSimulator() {
  const [params, setParams] = useState({
    modulation_type: 'bpsk',
    K: 100000,
    EbN0_dB: 4.0,
    alpha: 0.22,
    sps: 8,
    span: 10,
    seed: null,
    display_length: 20,
    qam_order: 16 // Default QAM order
  });

  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [expandedSection, setExpandedSection] = useState('all');

  const [berCurveResults, setBerCurveResults] = useState(null);
  const [isLoadingBerCurve, setIsLoadingBerCurve] = useState(false);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    const parsedValue = name === 'seed' && value === '' ? null :
      ['K', 'sps', 'span', 'display_length', 'seed', 'qam_order'].includes(name) ?
        parseInt(value, 10) : parseFloat(value);

    setParams({
      ...params,
      [name]: parsedValue
    });
  };

  const handleModulationChange = (e) => {
    setParams({
      ...params,
      modulation_type: e.target.value
    });
    // Clear results when switching modulation types
    setResults(null);
    setError(null);
  };

  const handleRandomSeed = () => {
    const randomSeed = Math.floor(Math.random() * 1000000);
    setParams({
      ...params,
      seed: randomSeed
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setBerCurveResults(null); // Clear BER curve results when running normal simulation
    
    try {
      // Display loading animation with subtle transition
      document.body.classList.add('cursor-progress');
      
      // Select the appropriate endpoint based on modulation type
      let endpoint;
      switch(params.modulation_type) {
        case 'bpsk':
          endpoint = 'http://localhost:8000/advanced-simulate-bpsk';
          break;
        case 'qpsk':
          endpoint = 'http://localhost:8000/advanced-simulate-qpsk';
          break;
        case 'qam':
          endpoint = 'http://localhost:8000/advanced-simulate-qam';
          break;
        default:
          throw new Error('Invalid modulation type');
      }
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify(params),
        signal: AbortSignal.timeout(120000), // Increased timeout to 120 seconds
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `Server responded with status: ${response.status}`);
      }

      const data = await response.json();
      
      // Validate numerical values before setting results
      if (isNaN(data.results.ber) || isNaN(data.results.theoretical_ber)) {
        throw new Error('Invalid numerical results received from server');
      }
      
      // Smooth transition for results display
      setResults(data);
      setExpandedSection('all');
      
      // Scroll to results if needed
      setTimeout(() => {
        document.querySelector('.lg\\:col-span-8')?.scrollIntoView({ 
          behavior: 'smooth',
          block: 'start'
        });
      }, 100);
    } catch (err) {
      console.error('Simulation error:', err);
      if (err.name === 'TimeoutError') {
        setError('Simulation timed out. Please try with fewer symbols or lower Eb/N0 values.');
      } else {
        setError(err.message || 'Failed to run simulation. Please try again.');
      }
    } finally {
      setIsLoading(false);
      document.body.classList.remove('cursor-progress');
    }
  };

  const handleBerCurveSubmit = async (e) => {
    e.preventDefault();
    setIsLoadingBerCurve(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:8000/ber-curve', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({
          modulation_type: params.modulation_type,
          K: params.K,
          alpha: params.alpha,
          sps: params.sps,
          span: params.span,
          seed: params.seed,
          qam_order: params.qam_order,
          ebno_range: [0, 20],
          num_points: 41  // Match backend's default of 41 points for better resolution
        }),
        signal: AbortSignal.timeout(60000), // Increased timeout to 60 seconds
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `Server responded with status: ${response.status}`);
      }

      const data = await response.json();
      setBerCurveResults(data);
    } catch (err) {
      console.error('BER curve error:', err);
      setError(err.message || 'Failed to calculate BER curve. Please try again.');
    } finally {
      setIsLoadingBerCurve(false);
    }
  };

  const toggleSection = (section) => {
    if (expandedSection === section) {
      setExpandedSection('all');
    } else {
      setExpandedSection(section);
    }
  };

  const isExpanded = (section) => {
    return expandedSection === 'all' || expandedSection === section;
  };

  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <header className="bg-gradient-to-r from-blue-700 to-indigo-900 text-white p-6 lg:p-8 shadow-xl">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl md:text-4xl font-bold tracking-tight">Digital Modulation with SRRC Filter Simulation</h1>
          <p className="opacity-90 mt-2 text-lg font-light">Interactive digital communication system analyzer for BPSK, QPSK, and QAM</p>
        </div>
      </header>
  
      <main className="flex-1 max-w-7xl mx-auto w-full p-4 md:p-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8">
          {/* Parameters Panel */}
          <div className="lg:col-span-4 space-y-6">
            <div className="bg-white rounded-2xl border border-blue-100 shadow-md overflow-hidden transition-all duration-300 hover:shadow-lg">
              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-5 border-b border-blue-100">
                <h2 className="text-xl font-semibold text-blue-800">Simulation Parameters</h2>
                <p className="text-sm text-blue-600">Adjust the values to configure your simulation</p>
              </div>
  
              <form onSubmit={handleSubmit} className="p-6">
                <div className="space-y-6">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">Modulation Type</label>
                    <select
                      name="modulation_type"
                      value={params.modulation_type}
                      onChange={handleModulationChange}
                      className="w-full rounded-lg border border-slate-300 px-4 py-3 bg-white text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition shadow-sm hover:border-blue-300"
                    >
                      <option value="bpsk">BPSK</option>
                      <option value="qpsk">QPSK</option>
                      <option value="qam">QAM</option>
                    </select>
                  </div>

                  {params.modulation_type === 'qam' && (
                    <div>
                      <label className="block text-sm font-medium text-slate-700 mb-2">QAM Order (2^l)</label>
                      <select
                        name="qam_order"
                        value={params.qam_order}
                        onChange={handleInputChange}
                        className="w-full rounded-lg border border-slate-300 px-4 py-3 bg-white text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition shadow-sm hover:border-blue-300"
                      >
                        <option value="4">4-QAM</option>
                        <option value="16">16-QAM</option>
                        <option value="64">64-QAM</option>
                        <option value="256">256-QAM</option>
                        <option value="1024">1024-QAM</option>
                      </select>
                      <p className="mt-2 text-xs text-slate-500">Select the QAM constellation size (2^l, where l is bits per symbol)</p>
                    </div>
                  )}
  
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">Number of Symbols (K)</label>
                    <input
                      type="number"
                      name="K"
                      min="1000"
                      max="10000000"
                      value={params.K}
                      onChange={handleInputChange}
                      className="w-full rounded-lg border border-slate-300 px-4 py-3 bg-white text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition shadow-sm hover:border-blue-300"
                    />
                    <p className="mt-2 text-xs text-slate-500">Range: 1,000 - 10,000,000</p>
                  </div>
  
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">Eb/N0 (dB)</label>
                    <div className="relative">
                      <input
                        type="range"
                        name="EbN0_dB"
                        min="0"
                        max="1000"
                        step="0.1"
                        value={params.EbN0_dB}
                        onChange={handleInputChange}
                        className="w-full h-2 bg-gradient-to-r from-blue-200 to-indigo-300 rounded-lg appearance-none cursor-pointer"
                      />
                      <div className="flex justify-between text-xs text-slate-600 mt-2">
                        <span>0 dB</span>
                        <span className="px-2 py-1 bg-blue-100 rounded-md text-blue-800 font-medium">{params.EbN0_dB} dB</span>
                        <span>1000 dB</span>
                      </div>
                    </div>
                  </div>
  
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">Roll-off Factor (α)</label>
                    <div className="relative">
                      <input
                        type="range"
                        name="alpha"
                        min="0"
                        max="1"
                        step="0.01"
                        value={params.alpha}
                        onChange={handleInputChange}
                        className="w-full h-2 bg-gradient-to-r from-indigo-200 to-purple-300 rounded-lg appearance-none cursor-pointer"
                      />
                      <div className="flex justify-between text-xs text-slate-600 mt-2">
                        <span>0.0</span>
                        <span className="px-2 py-1 bg-indigo-100 rounded-md text-indigo-800 font-medium">{params.alpha.toFixed(2)}</span>
                        <span>1.0</span>
                      </div>
                    </div>
                  </div>
  
                  <div className="grid grid-cols-2 gap-5">
                    <div>
                      <label className="block text-sm font-medium text-slate-700 mb-2">Samples/Symbol</label>
                      <input
                        type="number"
                        name="sps"
                        min="1"
                        max="500"
                        value={params.sps}
                        onChange={handleInputChange}
                        className="w-full rounded-lg border border-slate-300 px-4 py-3 bg-white text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition shadow-sm hover:border-blue-300"
                      />
                    </div>
  
                    <div>
                      <label className="block text-sm font-medium text-slate-700 mb-2">Filter Span</label>
                      <input
                        type="number"
                        name="span"
                        min="2"
                        max="20"
                        value={params.span}
                        onChange={handleInputChange}
                        className="w-full rounded-lg border border-slate-300 px-4 py-3 bg-white text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition shadow-sm hover:border-blue-300"
                      />
                    </div>
                  </div>
  
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">Display Length</label>
                    <input
                      type="number"
                      name="display_length"
                      min="5"
                      max="100"
                      value={params.display_length}
                      onChange={handleInputChange}
                      className="w-full rounded-lg border border-slate-300 px-4 py-3 bg-white text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition shadow-sm hover:border-blue-300"
                    />
                    <p className="mt-2 text-xs text-slate-500">Number of bits to show in graphics</p>
                  </div>
  
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">Random Seed</label>
                    <div className="flex">
                      <input
                        type="number"
                        name="seed"
                        value={params.seed === null ? '' : params.seed}
                        onChange={handleInputChange}
                        className="w-full rounded-l-lg border border-slate-300 px-4 py-3 bg-white text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition shadow-sm hover:border-blue-300"
                        placeholder="Optional"
                      />
                      <button
                        type="button"
                        onClick={handleRandomSeed}
                        className="flex items-center justify-center px-4 rounded-r-lg bg-slate-100 border border-l-0 border-slate-300 hover:bg-slate-200 text-slate-600 transition duration-200"
                        title="Generate random seed"
                      >
                        <RefreshCcw size={18} />
                      </button>
                    </div>
                    <p className="mt-2 text-xs text-slate-500">For reproducible results</p>
                  </div>
  
                  <button
                    type="submit"
                    disabled={isLoading}
                    className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-blue-600 to-indigo-700 hover:from-blue-700 hover:to-indigo-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 text-white font-medium py-3 px-5 rounded-lg shadow-md transition duration-200 disabled:opacity-70 disabled:cursor-not-allowed"
                  >
                    {isLoading ? (
                      <>
                        <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                        <span>Simulating...</span>
                      </>
                    ) : (
                      <>
                        <Play size={18} />
                        <span>Run Simulation</span>
                      </>
                    )}
                  </button>
                </div>
              </form>
            </div>
  
            {results && (
              <div className="bg-white rounded-2xl border border-green-100 shadow-md overflow-hidden transition-all duration-300 hover:shadow-lg">
                <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-5 border-b border-green-100">
                  <h2 className="text-xl font-semibold text-emerald-800">Performance Metrics</h2>
                </div>
                <div className="p-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                    <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-5 border border-blue-100 shadow-sm hover:shadow transition duration-300">
                      <div className="text-sm font-medium text-blue-700">Measured BER</div>
                      <div className="text-3xl font-bold mt-2 text-blue-800">
                        {(results.results.ber * 100).toFixed(2)}%
                      </div>
                      <div className="text-xs text-blue-600 mt-2 font-medium">
                        {results.results.errors} errors / {results.results.total_bits} bits
                      </div>
                    </div>
  
                    <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-xl p-5 border border-indigo-100 shadow-sm hover:shadow transition duration-300">
                      <div className="text-sm font-medium text-indigo-700">Theoretical BER</div>
                      <div className="text-3xl font-bold mt-2 text-indigo-800">
                        {(results.results.theoretical_ber * 100).toFixed(2)}%
                      </div>
                      <div className="text-xs text-indigo-600 mt-2 font-medium">
                        BPSK in AWGN channel
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
  
          {/* Results Panel */}
          <div className="lg:col-span-8 space-y-6">
            {error && (
              <div className="bg-red-50 border-l-4 border-red-500 p-5 rounded-lg shadow animate-pulse">
                <div className="flex">
                  <div className="ml-3">
                    <h3 className="text-lg font-medium text-red-800">Simulation Error</h3>
                    <div className="mt-2 text-sm text-red-700">{error}</div>
                  </div>
                </div>
              </div>
            )}
  
            {isLoading && !results && (
              <div className="bg-white rounded-2xl border border-blue-100 shadow-md p-16">
                <div className="flex flex-col items-center justify-center">
                  <div className="w-20 h-20 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                  <h3 className="mt-6 text-xl font-medium text-blue-800">Running Simulation</h3>
                  <p className="mt-3 text-blue-600">This may take a few moments...</p>
                </div>
              </div>
            )}
  
            {!results && !isLoading && !error && (
              <div className="bg-white rounded-2xl border border-blue-100 shadow-md p-16">
                <div className="flex flex-col items-center justify-center text-center">
                  <div className="w-20 h-20 flex items-center justify-center bg-blue-100 text-blue-600 rounded-full">
                    <ArrowRight size={40} />
                  </div>
                  <h3 className="mt-6 text-xl font-medium text-blue-800">Ready to Simulate</h3>
                  <p className="mt-3 text-blue-600 max-w-md">
                    Adjust parameters on the left panel and click "Run Simulation" to see the results
                  </p>
                </div>
              </div>
            )}
  
            {results && !isLoading && (
              <div className="space-y-8">
                {/* SRRC Filter */}
                <div className="bg-white rounded-2xl border border-blue-100 shadow-md overflow-hidden transition-all duration-300 hover:shadow-lg">
                  <div 
                    className="bg-gradient-to-r from-blue-50 to-indigo-50 p-5 border-b border-blue-100 flex justify-between items-center cursor-pointer hover:bg-gradient-to-r hover:from-blue-100 hover:to-indigo-100 transition duration-300"
                    onClick={() => toggleSection('filter')}
                  >
                    <h2 className="text-xl font-semibold text-blue-800">SRRC Filter Response</h2>
                    {isExpanded('filter') ? <ChevronUp size={22} className="text-blue-600" /> : <ChevronDown size={22} className="text-blue-600" />}
                  </div>
  
                  {isExpanded('filter') && (
                    <div className="p-6">
                      <div className="bg-white border rounded-xl overflow-hidden shadow-sm hover:shadow-md transition duration-300">
                        <img
                          src={`data:image/png;base64,${results.plots.srrc_filter}`}
                          alt="SRRC Filter"
                          className="w-full"
                        />
                      </div>
                      <div className="mt-4 text-sm text-slate-600 bg-slate-50 p-4 rounded-lg border border-slate-100">
                        Square Root Raised Cosine filter with roll-off factor α = {params.alpha}
                      </div>
                    </div>
                  )}
                </div>
                
                {/* BPSK Constellation at Transmitter */}
                <div className="bg-white rounded-2xl border border-blue-100 shadow-md overflow-hidden transition-all duration-300 hover:shadow-lg">
                  <div 
                    className="bg-gradient-to-r from-blue-50 to-indigo-50 p-5 border-b border-blue-100 flex justify-between items-center cursor-pointer hover:bg-gradient-to-r hover:from-blue-100 hover:to-indigo-100 transition duration-300"
                    onClick={() => toggleSection('constellation_tx')}
                  >
                    <h2 className="text-xl font-semibold text-blue-800">BPSK Constellation at Transmitter</h2>
                    {isExpanded('constellation_tx') ? <ChevronUp size={22} className="text-blue-600" /> : <ChevronDown size={22} className="text-blue-600" />}
                  </div>
  
                  {isExpanded('constellation_tx') && (
                    <div className="p-6">
                      <div className="bg-white border rounded-xl overflow-hidden shadow-sm hover:shadow-md transition duration-300">
                        <img
                          src={`data:image/png;base64,${results.plots.constellation_tx}`}
                          alt={`${params.modulation_type.toUpperCase()} Constellation at Transmitter`}
                          className="w-full"
                        />
                      </div>
                      <div className="mt-4 text-sm text-slate-600 bg-slate-50 p-4 rounded-lg border border-slate-100">
                        {params.modulation_type.toUpperCase()} constellation diagram showing the symbols at the transmitter
                      </div>
                    </div>
                  )}
                </div>
                
                {/* Constellation of Filtered Signal at TX */}
                <div className="bg-white rounded-2xl border border-blue-100 shadow-md overflow-hidden transition-all duration-300 hover:shadow-lg">
                  <div 
                    className="bg-gradient-to-r from-blue-50 to-indigo-50 p-5 border-b border-blue-100 flex justify-between items-center cursor-pointer hover:bg-gradient-to-r hover:from-blue-100 hover:to-indigo-100 transition duration-300"
                    onClick={() => toggleSection('constellation_tx_filtered')}
                  >
                    <h2 className="text-xl font-semibold text-blue-800">Filtered Signal Constellation (Tx)</h2>
                    {isExpanded('constellation_tx_filtered') ? <ChevronUp size={22} className="text-blue-600" /> : <ChevronDown size={22} className="text-blue-600" />}
                  </div>
  
                  {isExpanded('constellation_tx_filtered') && (
                    <div className="p-6">
                      <div className="bg-white border rounded-xl overflow-hidden shadow-sm hover:shadow-md transition duration-300">
                        <img
                          src={`data:image/png;base64,${results.plots.constellation_tx_filtered}`}
                          alt={`Filtered ${params.modulation_type.toUpperCase()} Signal Constellation (Tx)`}
                          className="w-full"
                        />
                      </div>
                      <div className="mt-4 text-sm text-slate-600 bg-slate-50 p-4 rounded-lg border border-slate-100">
                        Constellation diagram of the signal after pulse shaping with SRRC filter at the transmitter
                      </div>
                    </div>
                  )}
                </div>
                
                {/* Eye Diagram at Transmitter */}
                <div className="bg-white rounded-2xl border border-blue-100 shadow-md overflow-hidden transition-all duration-300 hover:shadow-lg">
                  <div 
                    className="bg-gradient-to-r from-blue-50 to-indigo-50 p-5 border-b border-blue-100 flex justify-between items-center cursor-pointer hover:bg-gradient-to-r hover:from-blue-100 hover:to-indigo-100 transition duration-300"
                    onClick={() => toggleSection('eye_tx')}
                  >
                    <h2 className="text-xl font-semibold text-blue-800">Eye Diagram at Transmitter</h2>
                    {isExpanded('eye_tx') ? <ChevronUp size={22} className="text-blue-600" /> : <ChevronDown size={22} className="text-blue-600" />}
                  </div>
  
                  {isExpanded('eye_tx') && (
                    <div className="p-6">
                      <div className="bg-white border rounded-xl overflow-hidden shadow-sm hover:shadow-md transition duration-300">
                        <img
                          src={`data:image/png;base64,${results.plots.eye_diagram_tx}`}
                          alt="Eye Diagram at Transmitter"
                          className="w-full"
                        />
                      </div>
                      <div className="mt-4 text-sm text-slate-600 bg-slate-50 p-4 rounded-lg border border-slate-100">
                        Eye diagram showing signal transitions at the transmitter after pulse shaping
                      </div>
                    </div>
                  )}
                </div>
                
                {/* Constellation of Received Signal with Noise */}
                <div className="bg-white rounded-2xl border border-blue-100 shadow-md overflow-hidden transition-all duration-300 hover:shadow-lg">
                  <div 
                    className="bg-gradient-to-r from-blue-50 to-indigo-50 p-5 border-b border-blue-100 flex justify-between items-center cursor-pointer hover:bg-gradient-to-r hover:from-blue-100 hover:to-indigo-100 transition duration-300"
                    onClick={() => toggleSection('constellation_rx')}
                  >
                    <h2 className="text-xl font-semibold text-blue-800">Received Signal Constellation (with Noise)</h2>
                    {isExpanded('constellation_rx') ? <ChevronUp size={22} className="text-blue-600" /> : <ChevronDown size={22} className="text-blue-600" />}
                  </div>
  
                  {isExpanded('constellation_rx') && (
                    <div className="p-6">
                      <div className="bg-white border rounded-xl overflow-hidden shadow-sm hover:shadow-md transition duration-300">
                        <img
                          src={`data:image/png;base64,${results.plots.constellation_rx}`}
                          alt={`Received ${params.modulation_type.toUpperCase()} Signal Constellation (with Noise)`}
                          className="w-full"
                        />
                      </div>
                      <div className="mt-4 text-sm text-slate-600 bg-slate-50 p-4 rounded-lg border border-slate-100">
                        Constellation diagram of the received signal with AWGN noise (Eb/N0 = {params.EbN0_dB} dB)
                      </div>
                    </div>
                  )}
                </div>
                
                {/* Eye Diagram at Receiver (before filtering) */}
                <div className="bg-white rounded-2xl border border-blue-100 shadow-md overflow-hidden transition-all duration-300 hover:shadow-lg">
                  <div 
                    className="bg-gradient-to-r from-blue-50 to-indigo-50 p-5 border-b border-blue-100 flex justify-between items-center cursor-pointer hover:bg-gradient-to-r hover:from-blue-100 hover:to-indigo-100 transition duration-300"
                    onClick={() => toggleSection('eye_rx')}
                  >
                    <h2 className="text-xl font-semibold text-blue-800">Eye Diagram at Receiver (Before Filtering)</h2>
                    {isExpanded('eye_rx') ? <ChevronUp size={22} className="text-blue-600" /> : <ChevronDown size={22} className="text-blue-600" />}
                  </div>
  
                  {isExpanded('eye_rx') && (
                    <div className="p-6">
                      <div className="bg-white border rounded-xl overflow-hidden shadow-sm hover:shadow-md transition duration-300">
                        <img
                          src={`data:image/png;base64,${results.plots.eye_diagram_rx}`}
                          alt="Eye Diagram at Receiver (Before Filtering)"
                          className="w-full"
                        />
                      </div>
                      <div className="mt-4 text-sm text-slate-600 bg-slate-50 p-4 rounded-lg border border-slate-100">
                        Eye diagram of the received signal with noise, before matched filtering
                      </div>
                    </div>
                  )}
                </div>
                
                {/* Constellation after Matched Filtering */}
                <div className="bg-white rounded-2xl border border-blue-100 shadow-md overflow-hidden transition-all duration-300 hover:shadow-lg">
                  <div 
                    className="bg-gradient-to-r from-blue-50 to-indigo-50 p-5 border-b border-blue-100 flex justify-between items-center cursor-pointer hover:bg-gradient-to-r hover:from-blue-100 hover:to-indigo-100 transition duration-300"
                    onClick={() => toggleSection('constellation_rx_filtered')}
                  >
                    <h2 className="text-xl font-semibold text-blue-800">Constellation after Matched Filtering</h2>
                    {isExpanded('constellation_rx_filtered') ? <ChevronUp size={22} className="text-blue-600" /> : <ChevronDown size={22} className="text-blue-600" />}
                  </div>
  
                  {isExpanded('constellation_rx_filtered') && (
                    <div className="p-6">
                      <div className="bg-white border rounded-xl overflow-hidden shadow-sm hover:shadow-md transition duration-300">
                        <img
                          src={`data:image/png;base64,${results.plots.constellation_rx_filtered}`}
                          alt={`${params.modulation_type.toUpperCase()} Constellation after Matched Filtering`}
                          className="w-full"
                        />
                      </div>
                      <div className="mt-4 text-sm text-slate-600 bg-slate-50 p-4 rounded-lg border border-slate-100">
                        Constellation diagram of the received signal after matched filtering, showing the decision points
                      </div>
                    </div>
                  )}
                </div>
                
                {/* Eye Diagram after Matched Filtering */}
                <div className="bg-white rounded-2xl border border-blue-100 shadow-md overflow-hidden transition-all duration-300 hover:shadow-lg">
                  <div 
                    className="bg-gradient-to-r from-blue-50 to-indigo-50 p-5 border-b border-blue-100 flex justify-between items-center cursor-pointer hover:bg-gradient-to-r hover:from-blue-100 hover:to-indigo-100 transition duration-300"
                    onClick={() => toggleSection('eye_rx_filtered')}
                  >
                    <h2 className="text-xl font-semibold text-blue-800">Eye Diagram after Matched Filtering</h2>
                    {isExpanded('eye_rx_filtered') ? <ChevronUp size={22} className="text-blue-600" /> : <ChevronDown size={22} className="text-blue-600" />}
                  </div>
  
                  {isExpanded('eye_rx_filtered') && (
                    <div className="p-6">
                      <div className="bg-white border rounded-xl overflow-hidden shadow-sm hover:shadow-md transition duration-300">
                        <img
                          src={`data:image/png;base64,${results.plots.eye_diagram_rx_filtered}`}
                          alt="Eye Diagram after Matched Filtering"
                          className="w-full"
                        />
                      </div>
                      <div className="mt-4 text-sm text-slate-600 bg-slate-50 p-4 rounded-lg border border-slate-100">
                        Eye diagram after matched filtering, with sampling point indicated by the vertical red line
                      </div>
                    </div>
                  )}
                </div>
                {/* BER Curve */}
                <div className="bg-white rounded-2xl border border-purple-100 shadow-md overflow-hidden transition-all duration-300 hover:shadow-lg">
                  <div 
                    className="bg-gradient-to-r from-purple-50 to-blue-50 p-5 border-b border-purple-100 flex justify-between items-center cursor-pointer hover:bg-gradient-to-r hover:from-purple-100 hover:to-blue-100 transition duration-300"
                    onClick={() => toggleSection('ber_curve')}
                  >
                    <h2 className="text-xl font-semibold text-purple-800">BER vs Eb/N0 Curve</h2>
                    {isExpanded('ber_curve') ? <ChevronUp size={22} className="text-purple-600" /> : <ChevronDown size={22} className="text-purple-600" />}
                  </div>
  
                  {isExpanded('ber_curve') && (
                    <div className="p-6">
                      <div className="space-y-6">
                        <div className="flex justify-center">
                          <button
                            onClick={handleBerCurveSubmit}
                            disabled={isLoadingBerCurve}
                            className="flex items-center justify-center gap-2 bg-gradient-to-r from-purple-600 to-blue-700 hover:from-purple-700 hover:to-blue-800 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 text-white font-medium py-3 px-5 rounded-lg shadow-md transition duration-200 disabled:opacity-70 disabled:cursor-not-allowed"
                          >
                            {isLoadingBerCurve ? (
                              <>
                                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                                <span>Calculating...</span>
                              </>
                            ) : (
                              <>
                                <Play size={18} />
                                <span>Calculate BER Curve</span>
                              </>
                            )}
                          </button>
                        </div>

                        {isLoadingBerCurve && (
                          <div className="flex flex-col items-center justify-center py-8">
                            <div className="w-20 h-20 border-4 border-purple-600 border-t-transparent rounded-full animate-spin"></div>
                            <h3 className="mt-6 text-xl font-medium text-purple-800">Calculating BER Curve</h3>
                            <p className="mt-3 text-purple-600">This may take a few moments...</p>
                          </div>
                        )}

                        {berCurveResults && !isLoadingBerCurve && (
                          <div className="space-y-6">
                            <div className="bg-white border rounded-xl overflow-hidden shadow-sm hover:shadow-md transition duration-300">
                              <img
                                src={`data:image/png;base64,${berCurveResults.ber_curve_plot}`}
                                alt="BER vs Eb/N0 Curve"
                                className="w-full"
                              />
                            </div>
                            
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              <div className="bg-purple-50 p-4 rounded-lg border border-purple-100">
                                <div className="text-sm font-medium text-purple-700">Simulated BER</div>
                                <div className="text-2xl font-bold mt-1 text-purple-900">
                                  {berCurveResults.ber_simulated[berCurveResults.ber_simulated.length - 1].toExponential(4)}
                                </div>
                                <div className="text-xs text-purple-600 mt-2">
                                  at Eb/N0 = {berCurveResults.ebno_points[berCurveResults.ebno_points.length - 1]} dB
                                </div>
                              </div>
                              
                              <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
                                <div className="text-sm font-medium text-blue-700">Theoretical BER</div>
                                <div className="text-2xl font-bold mt-1 text-blue-900">
                                  {berCurveResults.ber_theoretical[berCurveResults.ber_theoretical.length - 1].toExponential(4)}
                                </div>
                                <div className="text-xs text-blue-600 mt-2">
                                  at Eb/N0 = {berCurveResults.ebno_points[berCurveResults.ebno_points.length - 1]} dB
                                </div>
                              </div>
                            </div>

                            <div className="mt-4 text-sm bg-slate-50 p-4 rounded-lg border border-slate-100">
                              <p className="mb-2 text-slate-700 font-medium">About BER Curves</p>
                              <p className="text-slate-600">
                                The BER curve shows how the Bit Error Rate varies with the signal-to-noise ratio (Eb/N0).
                                The blue line represents the simulated BER, while the red dashed line shows the theoretical BER.
                                As Eb/N0 increases, the BER typically decreases exponentially.
                              </p>
                              <p className="mt-2 text-slate-600">
                                The theoretical BER for BPSK is given by: P<sub>e</sub> = 0.5 × erfc(√(E<sub>b</sub>/N<sub>0</sub>))
                              </p>
                              <p className="mt-2 text-slate-600">
                                For QPSK, the theoretical BER is: P<sub>e</sub> = erfc(√(E<sub>b</sub>/N<sub>0</sub>)) - 0.25 × erfc²(√(E<sub>b</sub>/N<sub>0</sub>))
                              </p>
                              <p className="mt-2 text-slate-600">
                                For M-QAM, the theoretical BER is approximately: P<sub>e</sub> ≈ (4/log2(M)) × (1-1/√M) × Q(√(3×log2(M)×E<sub>b</sub>/N<sub>0</sub>/(M-1)))
                              </p>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
                
                {/* Bits Comparison */}
                <div className="bg-white rounded-2xl border border-green-100 shadow-md overflow-hidden transition-all duration-300 hover:shadow-lg">
                  <div 
                    className="bg-gradient-to-r from-green-50 to-teal-50 p-5 border-b border-green-100 flex justify-between items-center cursor-pointer hover:bg-gradient-to-r hover:from-green-100 hover:to-teal-100 transition duration-300"
                    onClick={() => toggleSection('bits_comparison')}
                  >
                    <h2 className="text-xl font-semibold text-green-800">Transmitted vs Received Bits</h2>
                    {isExpanded('bits_comparison') ? <ChevronUp size={22} className="text-green-600" /> : <ChevronDown size={22} className="text-green-600" />}
                  </div>
  
                  {isExpanded('bits_comparison') && (
                    <div className="p-6">
                      <div className="overflow-x-auto">
                        <table className="w-full border-collapse">
                          <thead>
                            <tr>
                              <th className="bg-green-50 text-green-800 px-4 py-3 text-left border border-green-100">Index</th>
                              <th className="bg-green-50 text-green-800 px-4 py-3 text-left border border-green-100">Transmitted Bit</th>
                              <th className="bg-green-50 text-green-800 px-4 py-3 text-left border border-green-100">Received Bit</th>
                              <th className="bg-green-50 text-green-800 px-4 py-3 text-left border border-green-100">Sampled Value</th>
                              <th className="bg-green-50 text-green-800 px-4 py-3 text-left border border-green-100">Result</th>
                            </tr>
                          </thead>
                          <tbody>
                            {results.results.input_bits.map((bit, index) => {
                              const received = results.results.decoded_bits[index];
                              const sampled = results.results.sampled_signal[index];
                              const isError = bit !== received;
                              return (
                                <tr key={index} className={isError ? "bg-red-50" : "hover:bg-gray-50"}>
                                  <td className="px-4 py-3 border border-gray-100">{index}</td>
                                  <td className="px-4 py-3 border border-gray-100">{bit}</td>
                                  <td className="px-4 py-3 border border-gray-100">{received}</td>
                                  <td className="px-4 py-3 border border-gray-100">
                                    {params.modulation_type === 'bpsk' 
                                      ? sampled.toFixed(3)
                                      : sampled.real.toFixed(3)
                                    }
                                  </td>
                                  <td className={`px-4 py-3 border border-gray-100 font-medium ${isError ? "text-red-600" : "text-green-600"}`}>
                                    {isError ? "Error" : "Correct"}
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                      <div className="mt-4 text-sm text-slate-600 bg-slate-50 p-4 rounded-lg border border-slate-100">
                        The decision boundary for BPSK is at 0. Positive sampled values are decoded as bit 1, negative values as bit 0.
                      </div>
                    </div>
                  )}
                </div>
                
                {/* Technical Details */}
                <div className="bg-white rounded-2xl border border-gray-100 shadow-md overflow-hidden transition-all duration-300 hover:shadow-lg">
                  <div 
                    className="bg-gradient-to-r from-gray-50 to-slate-50 p-5 border-b border-gray-100 flex justify-between items-center cursor-pointer hover:bg-gradient-to-r hover:from-gray-100 hover:to-slate-100 transition duration-300"
                    onClick={() => toggleSection('technical')}
                  >
                    <h2 className="text-xl font-semibold text-gray-800">Technical Details</h2>
                    {isExpanded('technical') ? <ChevronUp size={22} className="text-gray-600" /> : <ChevronDown size={22} className="text-gray-600" />}
                  </div>
  
                  {isExpanded('technical') && (
                    <div className="p-6">
                      <div className="space-y-6">
                        <div>
                          <h3 className="text-lg font-medium text-gray-800 mb-3">System Parameters</h3>
                          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            <div className="bg-gray-50 p-4 rounded-lg border border-gray-100">
                              <div className="text-sm font-medium text-gray-600">Number of Symbols</div>
                              <div className="text-lg font-medium mt-1 text-gray-900">{params.K.toLocaleString()}</div>
                            </div>
                            <div className="bg-gray-50 p-4 rounded-lg border border-gray-100">
                              <div className="text-sm font-medium text-gray-600">Eb/N0</div>
                              <div className="text-lg font-medium mt-1 text-gray-900">{params.EbN0_dB} dB</div>
                            </div>
                            <div className="bg-gray-50 p-4 rounded-lg border border-gray-100">
                              <div className="text-sm font-medium text-gray-600">Roll-off Factor (α)</div>
                              <div className="text-lg font-medium mt-1 text-gray-900">{params.alpha}</div>
                            </div>
                            <div className="bg-gray-50 p-4 rounded-lg border border-gray-100">
                              <div className="text-sm font-medium text-gray-600">Samples per Symbol</div>
                              <div className="text-lg font-medium mt-1 text-gray-900">{params.sps}</div>
                            </div>
                            <div className="bg-gray-50 p-4 rounded-lg border border-gray-100">
                              <div className="text-sm font-medium text-gray-600">Filter Span</div>
                              <div className="text-lg font-medium mt-1 text-gray-900">{params.span} symbols</div>
                            </div>
                            <div className="bg-gray-50 p-4 rounded-lg border border-gray-100">
                              <div className="text-sm font-medium text-gray-600">Random Seed</div>
                              <div className="text-lg font-medium mt-1 text-gray-900">{params.seed === null ? "None (Random)" : params.seed}</div>
                            </div>
                          </div>
                        </div>
                        
                        <div>
                          <h3 className="text-lg font-medium text-gray-800 mb-3">About {params.modulation_type.toUpperCase()} with SRRC</h3>
                          <div className="bg-gray-50 p-4 rounded-lg border border-gray-100 text-gray-700">
                            <p>
                              {params.modulation_type.toUpperCase()} ({params.modulation_type === 'bpsk' ? 'Binary' : 'Quadrature'} Phase Shift Keying) is a digital modulation scheme where 
                              {params.modulation_type === 'bpsk' ? ' binary data is represented using two phase states (typically 0° and 180°).' : 
                              ' two bits are transmitted per symbol using four phase states (typically 45°, 135°, 225°, and 315°).'}
                            </p>
                            <p className="mt-2">
                              The Square Root Raised Cosine (SRRC) filter is used for pulse shaping to reduce 
                              intersymbol interference (ISI) while maintaining bandwidth efficiency. When an SRRC 
                              filter is used at both transmitter and receiver (matched filtering), the overall 
                              response satisfies the Nyquist criterion for zero ISI.
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
  
      <footer className="bg-gradient-to-r from-gray-800 to-gray-900 text-gray-300 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="md:flex md:items-center md:justify-between">
            <div className="flex justify-center md:justify-start space-x-6">
              <div>© {new Date().getFullYear()} BPSK Simulator</div>
              <div>|</div>
              <div>Digital Communications Educational Tool</div>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}