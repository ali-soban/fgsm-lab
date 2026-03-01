"use client";
import React, { useState } from 'react';
import { Upload, Zap, ShieldCheck } from 'lucide-react';

export default function AdversarialLab() {
  const [file, setFile] = useState<File | null>(null);
  const [epsilon, setEpsilon] = useState(0.1);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleAttack = async () => {
    if (!file) return alert("Please upload an image first!");
    setLoading(true);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('epsilon', epsilon.toString());

    try {
      // Replace with your AWS URL later for Part 3
      const response = await fetch('https://c2pr5n76ms5f234jfqfe3xnvvu0muuhi.lambda-url.ap-southeast-2.on.aws/attack', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (err) {
      alert("Error connecting to backend. Is FastAPI running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-50 p-8 font-sans">
      <div className="max-w-4xl mx-auto space-y-8">
        <header className="text-center">
          <h1 className="text-4xl font-bold text-gray-900">DevNeuron Adversarial Lab</h1>
          <p className="text-gray-600 mt-2">Testing Model Robustness with FGSM Attacks</p>
        </header>

        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Controls Section */}
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">1. Upload MNIST Digit</label>
              <input 
                type="file" 
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">2. Epsilon (Noise Level): {epsilon}</label>
              <input 
                type="range" min="0" max="0.5" step="0.01" value={epsilon} 
                onChange={(e) => setEpsilon(parseFloat(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
              />
            </div>

            <button 
              onClick={handleAttack}
              disabled={loading}
              className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold flex items-center justify-center gap-2 hover:bg-blue-700 transition-colors disabled:bg-gray-400"
            >
              {loading ? "Calculating..." : <><Zap size={18}/> Execute FGSM Attack</>}
            </button>
          </div>

          {/* Result Section */}
          <div className="flex flex-col items-center justify-center border-l border-gray-100 pl-8">
            {!result ? (
              <div className="text-gray-400 text-center">
                <ShieldCheck size={48} className="mx-auto mb-2 opacity-20" />
                <p>Upload and attack to see results</p>
              </div>
            ) : (
              <div className="space-y-6 w-full text-center">
                <div className="bg-red-50 p-4 rounded-lg">
                 <p className="text-sm text-red-600 font-bold uppercase tracking-wider">Adversarial Result</p>
                  <img src={result.adversarial_image} alt="Adversarial" className="w-32 h-32 mx-auto my-4 border-2 border-red-200 rounded shadow-inner" />
  
                   {/* Updated this line for high contrast */}
                    <p className="text-2xl font-mono text-gray-900">
                       Prediction: <span className="font-black text-red-700">{result.adversarial_prediction}</span>
                    </p>
                </div>
                <div className="flex justify-between text-xs font-medium text-gray-500">
                  <span>Original: {result.original_prediction}</span>
                  <span>Status: {result.success ? "✅ Success" : "❌ Failed"}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}