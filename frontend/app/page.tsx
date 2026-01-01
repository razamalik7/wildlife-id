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

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setObservationLocation(null);
      setIsLocationConfirmed(false);
      setPredictionResult(null);
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
          {predictionResult && preview ? (
            <div className="max-w-2xl mx-auto bg-white rounded-3xl shadow-xl overflow-hidden animate-in zoom-in-95 duration-300">
              <div className="relative h-64 bg-stone-900">
                <img src={preview} alt="Identified" className="w-full h-full object-cover opacity-90" />
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent" />
                <div className="absolute bottom-0 left-0 p-6 text-white w-full">
                  {predictionResult.candidates[0].score < 40 && (
                    <div className="mb-2 inline-flex items-center gap-2 px-3 py-1 bg-amber-500/90 text-amber-50 rounded-full text-xs font-bold uppercase tracking-wider backdrop-blur-md">
                      <span>⚠️ Low Confidence</span>
                    </div>
                  )}
                  <h2 className="text-3xl font-black tracking-tight">{predictionResult.candidates[0].name}</h2>
                  <p className="opacity-80 font-mono text-sm uppercase tracking-widest">{predictionResult.candidates[0].scientific_name}</p>
                </div>
              </div>

              <div className="p-6 space-y-6">
                {/* Confidence Bar */}
                <div className="space-y-2">
                  <div className="flex justify-between text-sm font-bold text-stone-600">
                    <span>Confidence Match</span>
                    <span>{predictionResult.candidates[0].score.toFixed(1)}%</span>
                  </div>
                  <div className="h-4 bg-stone-100 rounded-full overflow-hidden border border-stone-200">
                    <div
                      className={`h-full rounded-full transition-all duration-1000 ${predictionResult.candidates[0].score > 80 ? 'bg-emerald-500' :
                          predictionResult.candidates[0].score > 40 ? 'bg-amber-400' : 'bg-red-400'
                        }`}
                      style={{ width: `${predictionResult.candidates[0].score}%` }}
                    />
                  </div>
                  {predictionResult.candidates[0].score < 40 && (
                    <p className="text-xs text-amber-700 bg-amber-50 p-3 rounded-lg border border-amber-100">
                      Our model is unsure about this identification. It might be a species we haven't trained on yet, or the image quality is low.
                    </p>
                  )}
                </div>

                {/* Taxonomy Grid */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 bg-stone-50 rounded-2xl border border-stone-100">
                    <p className="text-xs uppercase tracking-wider text-stone-400 font-bold mb-1">Family</p>
                    <p className="font-bold text-stone-700">{predictionResult.candidates[0].taxonomy.family || 'Unknown'}</p>
                  </div>
                  <div className="p-4 bg-stone-50 rounded-2xl border border-stone-100">
                    <p className="text-xs uppercase tracking-wider text-stone-400 font-bold mb-1">Class</p>
                    <p className="font-bold text-stone-700">{predictionResult.candidates[0].taxonomy.class || 'Unknown'}</p>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-3 pt-2">
                  <button
                    onClick={resetFlow}
                    className="flex-1 py-3 px-4 bg-stone-100 hover:bg-stone-200 text-stone-600 font-bold rounded-xl transition-colors"
                  >
                    Identify Another
                  </button>
                  <button
                    onClick={() => router.push(`/anidex/${encodeURIComponent(predictionResult.candidates[0].name)}`)}
                    className="flex-[2] py-3 px-4 bg-emerald-600 hover:bg-emerald-700 text-white font-bold rounded-xl shadow-lg transition-transform active:scale-95"
                  >
                    View Full Details
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