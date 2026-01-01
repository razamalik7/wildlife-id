'use client';

import { useState, useRef } from 'react';
import axios from 'axios';
import { Camera, MapPin, Loader2, Search, Check } from 'lucide-react';
import dynamic from 'next/dynamic';
import 'leaflet/dist/leaflet.css';
import AppHeader from '@/components/AppHeader';
import { useApp } from '@/lib/AppContext';
import { supabase } from '@/lib/supabase';
import { useRouter } from 'next/navigation';

const ObservationLocationPicker = dynamic(() => import('@/components/ObservationLocationPicker'), { ssr: false });

export default function IdentifyPage() {
  const router = useRouter();
  const { user, allSpecies, anidex, setAnidex, location } = useApp();

  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [observationLocation, setObservationLocation] = useState<{ lat: number; lng: number } | null>(null);
  const [isLocationConfirmed, setIsLocationConfirmed] = useState(false);

  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [selectedCandidate, setSelectedCandidate] = useState<any>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setObservationLocation(null);
      setIsLocationConfirmed(false);
      setPredictionResult(null);
      setSelectedCandidate(null);
    }
  };

  const handleLocationConfirm = (lat: number, lng: number) => {
    setObservationLocation({ lat, lng });
    setIsLocationConfirmed(true);
  };

  const handleLocationChange = () => {
    setIsLocationConfirmed(false);
  };

  const resetFlow = () => {
    setImage(null);
    setPreview(null);
    setObservationLocation(null);
    setIsLocationConfirmed(false);
    setPredictionResult(null);
    setSelectedCandidate(null);
  };

  const identifyAnimal = async () => {
    if (!image || !observationLocation) return;

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', image);

      const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
      const aiRes = await axios.post(`${API_BASE}/predict`, formData);
      const candidates = aiRes.data.candidates;

      if (!candidates || candidates.length === 0) {
        alert('AI could not identify this animal.');
        setLoading(false);
        return;
      }

      const finalPrediction = candidates[0].name;
      const topScore = candidates[0].score;
      let bestTaxonId = 0;

      try {
        const taxRes = await axios.get(`https://api.inaturalist.org/v1/taxa?q=${finalPrediction}&per_page=1`);
        if (taxRes.data.results.length > 0) {
          bestTaxonId = taxRes.data.results[0].id;
        }
      } catch (e) {
        console.log('Could not fetch taxon ID');
      }

      const newEntry = {
        id: Date.now().toString(),
        name: finalPrediction,
        timestamp: new Date(),
        imageUrl: preview || '',
        verified: true,
        taxonId: bestTaxonId,
        lat: observationLocation.lat,
        lng: observationLocation.lng,
        isLocked: false,
        candidates: candidates.map((c: any) => c.name)
      };

      setAnidex((prev: any) => [newEntry, ...prev]);

      // Save to Supabase if logged in
      if (user) {
        try {
          const speciesInfo = allSpecies.find(s => s.name === finalPrediction);
          await supabase.from('observations').insert({
            user_id: user.id,
            image_url: preview || '',
            species: finalPrediction,
            confidence: topScore / 100, // Convert back to 0-1 for DB if needed, or keep as is. Usually DB expects 0-1 float.
            family: candidates[0].taxonomy?.family || speciesInfo?.taxonomy?.family || null,
            class: candidates[0].taxonomy?.class || speciesInfo?.taxonomy?.class || null,
            latitude: observationLocation.lat,
            longitude: observationLocation.lng,
            observed_at: new Date().toISOString()
          });
        } catch (e) {
          console.error('Supabase error:', e);
        }
      }

      // SET RESULT instead of redirecting
      setPredictionResult({
        candidates,
        bestTaxonId,
        taxonId: bestTaxonId // Just to be safe
      });
      setSelectedCandidate(candidates[0]);

    } catch (error) {
      alert('Backend Error. Is Python running?');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-stone-100 text-stone-900 pb-24">
      <AppHeader />

      <div className="max-w-5xl mx-auto p-4">
        {/* Main Content */}
        <div className="transition-all duration-500">

          {/* 1. RESULT VIEW */}
          {predictionResult && selectedCandidate && preview ? (
            <div className="max-w-2xl mx-auto bg-white rounded-3xl shadow-xl overflow-hidden animate-in zoom-in-95 duration-300">
              <div className="relative h-64 bg-stone-900 group">
                <img src={preview} alt="Identified" className="w-full h-full object-cover opacity-90 group-hover:scale-105 transition-transform duration-700" />
                <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/20 to-transparent" />
                <div className="absolute bottom-0 left-0 p-6 text-white w-full">
                  {selectedCandidate.score < 40 && (
                    <div className="mb-2 inline-flex items-center gap-2 px-3 py-1 bg-amber-500/90 text-amber-50 rounded-full text-xs font-bold uppercase tracking-wider backdrop-blur-md">
                      <span>⚠️ Low Confidence</span>
                    </div>
                  )}
                  <h2 className="text-4xl font-black tracking-tighter mb-1">{selectedCandidate.name}</h2>
                  <p className="opacity-80 font-mono text-sm uppercase tracking-widest">{selectedCandidate.scientific_name}</p>
                </div>
              </div>

              <div className="p-6 space-y-8">
                {/* Confidence Bar */}
                <div className="space-y-3">
                  <div className="flex justify-between items-end text-stone-600">
                    <span className="text-sm font-bold uppercase tracking-wider opacity-70">Confidence Match</span>
                    <span className="text-2xl font-black text-emerald-600">{selectedCandidate.score.toFixed(1)}%</span>
                  </div>
                  <div className="h-5 bg-stone-100 rounded-full overflow-hidden border border-stone-200 shadow-inner">
                    <div
                      className={`h-full rounded-full transition-all duration-1000 ${selectedCandidate.score > 80 ? 'bg-gradient-to-r from-emerald-500 to-emerald-400' :
                          selectedCandidate.score > 40 ? 'bg-gradient-to-r from-amber-400 to-amber-300' : 'bg-red-400'
                        }`}
                      style={{ width: `${selectedCandidate.score}%` }}
                    />
                  </div>
                  {selectedCandidate.score < 40 && (
                    <p className="text-sm text-amber-800 bg-amber-50 p-4 rounded-xl border border-amber-100 flex items-start gap-3">
                      <span className="text-xl">🤔</span>
                      <span>Our model is unsure about this identification. It might be a species we haven't trained on yet, or the image quality is low.</span>
                    </p>
                  )}
                </div>

                {/* Taxonomy Grid */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-5 bg-stone-50 rounded-2xl border border-stone-100">
                    <p className="text-xs uppercase tracking-wider text-stone-400 font-bold mb-2">My Family</p>
                    <p className="font-bold text-xl text-stone-700">{selectedCandidate.taxonomy.family || 'Unknown'}</p>
                  </div>
                  <div className="p-5 bg-stone-50 rounded-2xl border border-stone-100">
                    <p className="text-xs uppercase tracking-wider text-stone-400 font-bold mb-2">My Class</p>
                    <p className="font-bold text-xl text-stone-700">{selectedCandidate.taxonomy.class || 'Unknown'}</p>
                  </div>
                </div>

                {/* Alternative Candidates - Interactive Cards */}
                {predictionResult.candidates.length > 1 && (
                  <div className="pt-4 border-t border-stone-100">
                    <p className="text-xs font-bold text-stone-400 uppercase tracking-widest mb-4">Other Potential Matches</p>
                    <div className="space-y-3">
                      {predictionResult.candidates
                        .filter((c: any) => c.name !== selectedCandidate.name)
                        .map((candidate: any, idx: number) => (
                          <div
                            key={idx}
                            onClick={() => setSelectedCandidate(candidate)}
                            className="flex justify-between items-center p-4 rounded-xl border-2 border-transparent bg-stone-50 hover:bg-white hover:border-emerald-500 hover:shadow-md cursor-pointer transition-all group"
                          >
                            <div>
                              <h4 className="font-bold text-stone-700 group-hover:text-emerald-700 transition-colors">{candidate.name}</h4>
                              <p className="text-xs text-stone-400 uppercase tracking-wide">{candidate.scientific_name}</p>
                            </div>
                            <div className="text-right">
                              <span className={`font-mono font-bold ${candidate.score > 50 ? 'text-emerald-600' : 'text-stone-400'}`}>
                                {candidate.score.toFixed(1)}%
                              </span>
                              <p className="text-[10px] text-stone-300 font-bold uppercase tracking-widest mt-1 group-hover:text-emerald-500">View This</p>
                            </div>
                          </div>
                        ))}
                    </div>
                  </div>
                )}

                {/* Actions */}
                <div className="flex gap-4 pt-4 border-t border-stone-100">
                  <button
                    onClick={resetFlow}
                    className="flex-1 py-4 px-6 bg-stone-100 hover:bg-stone-200 text-stone-600 font-bold rounded-2xl transition-colors text-lg"
                  >
                    Identify Another
                  </button>
                  <button
                    onClick={() => router.push(`/anidex/${encodeURIComponent(selectedCandidate.name)}`)}
                    className="flex-[2] py-4 px-6 bg-emerald-600 hover:bg-emerald-700 text-white font-bold rounded-2xl shadow-xl hover:shadow-2xl hover:scale-[1.02] transition-all text-lg flex items-center justify-center gap-2"
                  >
                    <span>Learn More</span>
                    <Check size={20} strokeWidth={3} />
                  </button>
                </div>
              </div>
            </div>
          ) : (

            /* 2. UPLOAD & MAP FLOW */
            <>
              {/* Photo Upload Panel - Only shows when NO photo */}
              {!preview && (
                <div className="max-w-4xl mx-auto w-full">
                  <div
                    onClick={() => fileInputRef.current?.click()}
                    className="aspect-[4/3] w-full bg-stone-200 rounded-3xl border-4 border-dashed border-stone-300 flex flex-col items-center justify-center cursor-pointer hover:border-emerald-500 hover:bg-stone-100 transition-colors overflow-hidden relative shadow-inner"
                  >
                    <div className="text-stone-400 flex flex-col items-center">
                      <div className="bg-white p-6 rounded-full shadow-sm mb-4">
                        <Camera size={48} className="text-emerald-600" />
                      </div>
                      <p className="font-bold text-lg text-stone-600">Tap to Upload Photo</p>
                      <p className="text-sm opacity-70">JPG or PNG</p>
                    </div>
                  </div>
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    className="hidden"
                    accept="image/*"
                  />
                </div>
              )}

              {/* Map Panel - Full width after photo upload */}
              {preview && !predictionResult && (
                <div className="w-full flex flex-col animate-in fade-in duration-500">
                  <div className="rounded-3xl overflow-hidden shadow-lg h-[500px]">
                    <ObservationLocationPicker
                      initialLat={location?.lat}
                      initialLng={location?.lng}
                      photoPreview={preview}
                      isConfirmed={isLocationConfirmed}
                      onConfirm={handleLocationConfirm}
                      onLocationChange={handleLocationChange}
                    />
                  </div>

                  {/* Identify Button */}
                  <button
                    onClick={identifyAnimal}
                    disabled={!isLocationConfirmed || loading}
                    className="mt-4 w-full py-4 bg-emerald-600 hover:bg-emerald-700 disabled:bg-stone-300 text-white font-bold text-lg rounded-2xl shadow-xl flex items-center justify-center gap-3 transition-all disabled:cursor-not-allowed"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="animate-spin" size={24} />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Search size={24} />
                        {!isLocationConfirmed ? 'Confirm Location First' : 'Identify Species'}
                      </>
                    )}
                  </button>
                </div>
              )}
            </>
          )}
        </div>

        {/* Oogway Branding - Only when no photo */}
        {!preview && (
          <div className="mt-8 text-center space-y-4">
            <div>
              <p className="text-xs font-bold text-emerald-800/60 tracking-[0.2em] uppercase">Powered by OOGWAY</p>
              <p className="text-xs text-emerald-600/70 mt-1 italic">Custom Architecture Originally Developed for aniML vision</p>
            </div>

            <div className="flex justify-center gap-8 text-xs text-emerald-700/80 bg-stone-200/50 py-3 rounded-xl max-w-lg mx-auto border border-stone-200">
              <div className="flex flex-col">
                <span className="font-bold text-lg">90.7%</span>
                <span className="uppercase tracking-wider opacity-70">Species Acc</span>
              </div>
              <div className="w-px bg-stone-300" />
              <div className="flex flex-col">
                <span className="font-bold text-lg">92.7%</span>
                <span className="uppercase tracking-wider opacity-70">Family Acc</span>
              </div>
              <div className="w-px bg-stone-300" />
              <div className="flex flex-col">
                <span className="font-bold text-lg">97.6%</span>
                <span className="uppercase tracking-wider opacity-70">Class Acc</span>
              </div>
              <div className="w-px bg-stone-300" />
              <div className="flex flex-col">
                <span className="font-bold text-lg">100</span>
                <span className="uppercase tracking-wider opacity-70">Species</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}