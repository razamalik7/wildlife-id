"use client";

import "./globals.css";
import { useState, useRef } from 'react';
import { Leaf, Upload, X, Image as ImageIcon, Scan, CheckCircle2, RefreshCw, AlertTriangle } from 'lucide-react';

export default function Home() {
  // --- STATE ---
  const [view, setView] = useState<'initial' | 'analyzing' | 'result'>('initial');
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null); // NEW: We need to remember the actual file
  const [result, setResult] = useState<any>(null); // NEW: Store the answer from Python
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // --- LOGIC ---
  
  const handleFile = (file: File) => {
    if (!file.type.startsWith('image/')) return alert('Please upload an image file.');
    
    // Save the file in state so we can send it later
    setSelectedFile(file);

    // Create a fake URL for preview
    const objectUrl = URL.createObjectURL(file);
    setImagePreview(objectUrl);
  };

  // THIS IS THE NEW FUNCTION THAT CONNECTS TO PYTHON
  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setView('analyzing'); // Show the loading spinner

    const formData = new FormData();
    formData.append('file', selectedFile); 

    try {
      // 1. Send to FastAPI (Port 8000)
      const response = await fetch('http://127.0.0.1:8000/api/analyze', {
        method: 'POST',
        body: formData,
      });

      // 2. Get the result
      const data = await response.json();

      if (data.success) {
        setResult(data); // Save the Python data
        setView('result'); // Switch to result screen
      } else {
        alert("Server error: " + data.error);
        setView('initial');
      }

    } catch (error) {
      console.error(error);
      alert("Could not connect to Python! Is the backend running?");
      setView('initial');
    }
  };

  const resetApp = () => {
    setImagePreview(null);
    setSelectedFile(null);
    setResult(null);
    setView('initial');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Drag and Drop Handlers
  const handleDragOver = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = () => setIsDragging(false);
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
  };

  // --- RENDER ---
  return (
    <div className="max-w-[900px] mx-auto p-8 min-h-screen flex flex-col">
      
      {/* Navbar */}
      <nav className="flex justify-between items-center mb-12">
        <div className="font-serif font-bold text-2xl text-[#1a2f1a]">Wildlife Dex</div>
        <div className="text-sm font-medium text-[#4f824f]">v0.1.0 Beta</div>
      </nav>

      <main className="flex-1 flex flex-col items-center gap-8 w-full">
        
        {/* SECTION 1: HERO & UPLOAD */}
        {view === 'initial' && (
          <>
            <section className="text-center w-full max-w-[600px] animate-fade-in">
              <div className="bg-[#e0f0e0] p-3 rounded-full inline-flex mb-4">
                <Leaf className="w-8 h-8 text-[#4f824f]" />
              </div>
              <h1 className="font-serif text-5xl mb-4">Wildlife Dex</h1>
              <p className="text-[#2d4a2d] text-lg">
                Your digital field guide. Upload a photo to identify species instantly.
              </p>
            </section>

            <section className="w-full max-w-[600px] animate-fade-in">
              <div 
                className={`border-2 border-dashed rounded-2xl p-8 transition-all cursor-pointer relative
                  ${isDragging ? 'border-[#609e60] bg-[#f0f7f0]' : 'border-[#c2e0c2] bg-white/50 hover:bg-[#f0f7f0]'}
                `}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => !imagePreview && fileInputRef.current?.click()}
              >
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  className="hidden" 
                  accept="image/*"
                  onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} 
                />

                {!imagePreview ? (
                  // UPLOAD PROMPT
                  <div className="text-center">
                    <div className="bg-[#e0f0e0] p-3 rounded-full inline-flex mb-4">
                      <Upload className="w-6 h-6 text-[#4f824f]" />
                    </div>
                    <h3 className="text-lg font-semibold mb-1">Click or drag image to upload</h3>
                    <p className="text-sm text-gray-500">Supports JPG, PNG, WEBP</p>
                  </div>
                ) : (
                  // IMAGE PREVIEW
                  <div className="animate-fade-in">
                    <div className="relative rounded-xl overflow-hidden bg-[#e0f0e0] aspect-video mb-6">
                      <img src={imagePreview} alt="Preview" className="w-full h-full object-contain" />
                      <button 
                        onClick={(e) => { e.stopPropagation(); resetApp(); }}
                        className="absolute top-2 right-2 bg-black/50 text-white p-1 rounded-full hover:bg-black/70"
                      >
                        <X className="w-5 h-5" />
                      </button>
                    </div>
                    {/* UPDATED BUTTON: Calls handleAnalyze now */}
                    <button 
                      onClick={(e) => { e.stopPropagation(); handleAnalyze(); }}
                      className="w-full py-3 px-6 bg-[#4f824f] text-white rounded-xl font-medium flex items-center justify-center gap-2 hover:bg-[#609e60] shadow-lg shadow-[#4f824f]/20 transition-colors"
                    >
                      <ImageIcon className="w-5 h-5" />
                      Identify Animal
                    </button>
                  </div>
                )}
              </div>
            </section>
          </>
        )}

        {/* SECTION 2: ANALYZING */}
        {view === 'analyzing' && (
          <section className="text-center w-full max-w-[600px] animate-fade-in">
            <div className="relative w-32 h-32 mx-auto mb-8">
              <div className="absolute inset-0 border-4 border-[#c2e0c2] rounded-full"></div>
              <div className="absolute inset-0 border-4 border-[#609e60] rounded-full border-t-transparent border-l-transparent animate-[spin_1.5s_linear_infinite]"></div>
              <div className="absolute inset-4 bg-[#e0f0e0] rounded-full flex items-center justify-center animate-[pulse_2s_infinite]">
                <Scan className="w-12 h-12 text-[#4f824f]" />
              </div>
            </div>
            <h2 className="text-2xl font-bold mb-2">Analyzing Features...</h2>
            <p className="text-[#2d4a2d]">Sending data to backend...</p>
          </section>
        )}

        {/* SECTION 3: RESULT */}
        {view === 'result' && result && (
          <section className="w-full max-w-[600px] animate-fade-in">
            <div className="bg-white/80 backdrop-blur-md border border-white/20 rounded-2xl p-8 shadow-xl text-center">
              
              <div className="inline-flex items-center gap-2 bg-green-100 text-green-800 px-4 py-1.5 rounded-full text-sm font-medium mb-6">
                <CheckCircle2 className="w-4 h-4" />
                Identification Complete
              </div>

              <div className="rounded-xl overflow-hidden shadow-sm border-4 border-white mb-6">
                <img src={imagePreview || ''} alt="Result" className="w-full h-auto block" />
              </div>

              <div className="mb-8">
                {/* DYNAMIC RESULT FROM PYTHON */}
                <h2 className="text-3xl font-serif font-bold mb-2 text-[#1a2f1a]">
                  {result.result}
                </h2>
                <p className="text-[#4f824f] font-medium mb-4">Confidence: {result.confidence}</p>
                <p className="text-gray-600 mb-6">{result.info}</p>

                {/* INVASIVE STATUS BADGE */}
                {result.invasive_status ? (
                   <div className="inline-flex items-center gap-2 bg-red-100 text-red-700 px-4 py-2 rounded-lg text-sm font-bold border border-red-200">
                     <AlertTriangle className="w-4 h-4" />
                     INVASIVE SPECIES
                   </div>
                ) : (
                  <div className="inline-flex items-center gap-2 bg-blue-50 text-blue-700 px-4 py-2 rounded-lg text-sm font-medium border border-blue-100">
                    <CheckCircle2 className="w-4 h-4" />
                    Native Species
                  </div>
                )}
              </div>

              <button 
                onClick={resetApp}
                className="text-[#4f824f] font-medium inline-flex items-center gap-2 hover:text-[#1a2f1a] transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
                Identify Another Animal
              </button>
            </div>
          </section>
        )}

      </main>

      <footer className="text-center text-[#85b885] text-sm mt-16">
        <p>&copy; 2024 Wildlife Dex. Built for nature lovers.</p>
      </footer>
    </div>
  );
}