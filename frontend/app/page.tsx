'use client';

import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Camera, MapPin, Loader2, Search, Check } from 'lucide-react';
import dynamic from 'next/dynamic';
import 'leaflet/dist/leaflet.css';
import AppHeader from '@/components/AppHeader';
import { useApp } from '@/lib/AppContext';
import { supabase } from '@/lib/supabase';
import { useRouter } from 'next/navigation';

const ObservationLocationPicker = dynamic(() => import('@/components/ObservationLocationPicker'), { ssr: false });

// --- SIDEBAR RANGE BADGE (Live iNaturalist Check) ---
const SidebarRangeBadge = ({ taxonId, lat, lng }: { taxonId: number; lat: number; lng: number }) => {
  const [status, setStatus] = useState<'loading' | 'yes' | 'no' | null>(null);

  useEffect(() => {
    if (!taxonId || !lat || !lng) {
      setStatus(null);
      return;
    }

    setStatus('loading');
    let cancelled = false;

    const check = async () => {
      try {
        const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
        const res = await axios.get(`${API_BASE}/inat/observations`, {
          params: {
            taxon_id: taxonId,
            lat: lat,
            lng: lng,
            radius: 100,
            captive: 'false',
            per_page: 1
          }
        });
        if (!cancelled) {
          setStatus(res.data.total_results > 0 ? 'yes' : 'no');
        }
      } catch (e) {
        if (!cancelled) setStatus(null);
      }
    };

    check();
    return () => { cancelled = true; };
  }, [taxonId, lat, lng]);

  if (status === 'loading') {
    return <span className="text-[9px] px-1.5 py-0.5 rounded-md font-bold uppercase bg-stone-100 text-stone-400 animate-pulse">Checking...</span>;
  }
  if (status === 'yes') {
    return <span className="text-[9px] px-1.5 py-0.5 rounded-md font-bold uppercase bg-emerald-100 text-emerald-700">In Range</span>;
  }
  if (status === 'no') {
    return <span className="text-[9px] px-1.5 py-0.5 rounded-md font-bold uppercase bg-red-100 text-red-700">Out of Range</span>;
  }
  return null;
};


export default function IdentifyPage() {
  const router = useRouter();
  const { user, allSpecies, anidex, setAnidex, location } = useApp();

  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [observationLocation, setObservationLocation] = useState<{ lat: number; lng: number } | null>(null);
  const [isLocationConfirmed, setIsLocationConfirmed] = useState(false);
  const [showSignupNudge, setShowSignupNudge] = useState(false);

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

  const [boundingBox, setBoundingBox] = useState<any>(null);

  // New State for Live Range Check
  const [rangeStatus, setRangeStatus] = useState<'loading' | 'yes' | 'no' | null>(null);

  useEffect(() => {
    if (!selectedCandidate || !observationLocation || !selectedCandidate.taxon_id) {
      setRangeStatus(null);
      return;
    }

    // Don't re-fetch if we already have a specialized guess, but live valid is better.
    setRangeStatus('loading');

    // Debounce slightly to avoid rapid clicks
    const timer = setTimeout(async () => {
      try {
        const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
        const res = await axios.get(`${API_BASE}/inat/observations`, {
          params: {
            taxon_id: selectedCandidate.taxon_id,
            lat: observationLocation.lat,
            lng: observationLocation.lng,
            radius: 100, // 100km radius
            captive: 'false', // Exclude Zoo animals
            per_page: 1 // Optimization
          }
        });

        if (res.data.total_results > 0) {
          setRangeStatus('yes');
        } else {
          setRangeStatus('no');
        }
      } catch (e) {
        console.error("Range check failed", e);
        setRangeStatus(null);
      }
    }, 500);

    return () => clearTimeout(timer);
  }, [selectedCandidate, observationLocation]);

  const identifyAnimal = async () => {
    if (!image || !observationLocation) return;

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', image);
      if (observationLocation) {
        formData.append('lat', observationLocation.lat.toString());
        formData.append('lng', observationLocation.lng.toString());
      }

      const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
      const aiRes = await axios.post(`${API_BASE}/predict`, formData);
      const candidates = aiRes.data.candidates;

      if (aiRes.data.bbox) {
        setBoundingBox(aiRes.data.bbox);
      } else {
        setBoundingBox(null);
      }

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

      const speciesInfo = allSpecies.find(s => s.name === finalPrediction);

      // Save to Supabase if logged in
      if (user) {
        try {
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
      } else {
        // Save to localStorage for anonymous map
        const localObs = {
          id: Date.now(), // Fake ID
          user_id: 'anon',
          image_url: preview || '',
          common_name: finalPrediction, // Map uses common_name
          scientific_name: speciesInfo?.scientific_name || candidates[0].name,
          category: speciesInfo?.category || 'Native', // Default to Native locally if unknown
          family: candidates[0].taxonomy?.family || speciesInfo?.taxonomy?.family || null,
          class: candidates[0].taxonomy?.class || speciesInfo?.taxonomy?.class || null,
          lat: observationLocation.lat,
          lng: observationLocation.lng,
          country: 'Unknown',
          created_at: new Date().toISOString()
        };

        try {
          const existing = JSON.parse(localStorage.getItem('local_observations') || '[]');
          localStorage.setItem('local_observations', JSON.stringify([localObs, ...existing]));
        } catch (e) {
          console.error('LocalStorage error:', e);
        }
      }

      // Enrich candidates with client-side taxonomy if missing from API
      const enrichedCandidates = candidates.map((c: any) => {
        if (!c.taxonomy?.family || c.taxonomy.family === 'Unknown') {
          const match = allSpecies.find(s => s.name === c.name);
          if (match?.taxonomy) {
            return {
              ...c,
              taxonomy: {
                ...c.taxonomy,
                family: match.taxonomy.family || 'Unknown',
                class: match.taxonomy.class || 'Unknown'
              }
            };
          }
        }
        return c;
      });

      // SET RESULT instead of redirecting
      setPredictionResult({
        candidates: enrichedCandidates,
        bestTaxonId,
        taxonId: bestTaxonId // Just to be safe
      });
      setSelectedCandidate(enrichedCandidates[0]);

      // Show signup nudge for anonymous users after first identification
      if (!user) {
        setTimeout(() => setShowSignupNudge(true), 2000); // Delay to let them see the result first
      }

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
            <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6 animate-in zoom-in-95 duration-500">

              {/* PRIMARY CARD (Left, 2/3 width) */}
              <div className="lg:col-span-2 bg-white rounded-3xl shadow-xl overflow-hidden flex flex-col">
                <div className="relative h-72 sm:h-96 bg-stone-900 group">
                  <img src={preview} alt="Identified" className="w-full h-full object-cover opacity-90 group-hover:scale-105 transition-transform duration-700" />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/20 to-transparent" />

                  {/* Bounding Box Overlay */}
                  {boundingBox && (
                    <div
                      className="absolute border-2 border-white/80 bg-white/10 z-10 rounded-lg pointer-events-none shadow-[0_0_20px_rgba(255,255,255,0.3)] animate-in fade-in zoom-in duration-700"
                      style={{
                        left: `${boundingBox.x * 100}%`,
                        top: `${boundingBox.y * 100}%`,
                        width: `${boundingBox.w * 100}%`,
                        height: `${boundingBox.h * 100}%`
                      }}
                    >
                      {/* Corner Accents */}
                      <div className="absolute top-0 left-0 w-3 h-3 border-t-2 border-l-2 border-emerald-400 -mt-0.5 -ml-0.5"></div>
                      <div className="absolute top-0 right-0 w-3 h-3 border-t-2 border-r-2 border-emerald-400 -mt-0.5 -mr-0.5"></div>
                      <div className="absolute bottom-0 left-0 w-3 h-3 border-b-2 border-l-2 border-emerald-400 -mb-0.5 -ml-0.5"></div>
                      <div className="absolute bottom-0 right-0 w-3 h-3 border-b-2 border-r-2 border-emerald-400 -mb-0.5 -mr-0.5"></div>
                    </div>
                  )}

                  <div className="absolute bottom-0 left-0 p-8 text-white w-full">
                    {selectedCandidate.score < 40 && (
                      <div className="mb-3 inline-flex items-center gap-2 px-3 py-1 bg-amber-500/90 text-amber-50 rounded-full text-xs font-bold uppercase tracking-wider backdrop-blur-md">
                        <span>⚠️ Low Confidence</span>
                      </div>
                    )}
                    <h2 className="text-4xl sm:text-6xl font-black tracking-tighter mb-2 leading-none">{selectedCandidate.name}</h2>
                    <p className="opacity-80 font-mono text-sm sm:text-base uppercase tracking-widest flex items-center gap-2 flex-wrap">
                      {selectedCandidate.scientific_name}

                      {/* 1. Live Check Loading */}
                      {rangeStatus === 'loading' && (
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider border bg-stone-100 text-stone-500 border-stone-200 animate-pulse">
                          📡 Checking Location...
                        </span>
                      )}

                      {/* 2. Live Check Result (Priority) */}
                      {rangeStatus === 'yes' && (
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider border backdrop-blur-md bg-emerald-500/20 text-emerald-600 border-emerald-500/30">
                          ✅ Verified In Range
                        </span>
                      )}
                      {rangeStatus === 'no' && (
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider border backdrop-blur-md bg-amber-500/20 text-amber-600 border-amber-500/30">
                          ⚠️ Out of Range
                        </span>
                      )}

                      {/* 3. Static Fallback (Only if Live Check hasn't finished or failed) */}
                      {rangeStatus === null && selectedCandidate.in_range !== undefined && (
                        <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider border backdrop-blur-md ${selectedCandidate.in_range
                          ? 'bg-emerald-500/20 text-emerald-600 border-emerald-500/30'
                          : 'bg-red-500/20 text-red-600 border-red-500/30'
                          }`}>
                          {selectedCandidate.in_range ? 'Possibly In Range' : 'Likely Out of Range'}
                        </span>
                      )}
                    </p>
                  </div>
                </div>

                <div className="p-6 sm:p-8 flex-1 flex flex-col justify-between space-y-8">
                  {/* Taxonomy Grid */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 sm:p-5 bg-stone-50 rounded-2xl border border-stone-100">
                      <p className="text-xs uppercase tracking-wider text-stone-400 font-bold mb-2">Family</p>
                      <p className="font-bold text-xl sm:text-2xl text-stone-700">{selectedCandidate.taxonomy.family || 'Unknown'}</p>
                    </div>
                    <div className="p-4 sm:p-5 bg-stone-50 rounded-2xl border border-stone-100">
                      <p className="text-xs uppercase tracking-wider text-stone-400 font-bold mb-2">Class</p>
                      <p className="font-bold text-xl sm:text-2xl text-stone-700">{selectedCandidate.taxonomy.class || 'Unknown'}</p>
                    </div>
                  </div>

                  {/* Confidence Bar */}
                  <div className="space-y-3">
                    <div className="flex justify-between items-end text-stone-600">
                      <span className="text-sm font-bold uppercase tracking-wider opacity-70">Confidence</span>
                      <span className="text-3xl font-black text-emerald-600">{selectedCandidate.score.toFixed(1)}%</span>
                    </div>
                    <div className="h-4 bg-stone-100 rounded-full overflow-hidden border border-stone-200">
                      <div
                        className={`h-full rounded-full transition-all duration-1000 ${selectedCandidate.score > 80 ? 'bg-emerald-500' :
                          selectedCandidate.score > 40 ? 'bg-amber-400' : 'bg-red-400'
                          }`}
                        style={{ width: `${selectedCandidate.score}%` }}
                      />
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex flex-col sm:flex-row gap-4 pt-4 border-t border-stone-100">
                    <button
                      onClick={resetFlow}
                      className="w-full sm:flex-1 py-4 px-6 bg-stone-100 hover:bg-stone-200 text-stone-600 font-bold rounded-2xl transition-colors text-lg"
                    >
                      New Photo
                    </button>
                    <button
                      onClick={() => router.push(`/anidex/${encodeURIComponent(selectedCandidate.name)}`)}
                      className="w-full sm:flex-[2] py-4 px-6 bg-emerald-600 hover:bg-emerald-700 text-white font-bold rounded-2xl shadow-xl hover:shadow-2xl hover:scale-[1.02] transition-all text-lg flex items-center justify-center gap-2"
                    >
                      <span>Full Details</span>
                      <Check size={20} strokeWidth={3} />
                    </button>
                  </div>
                </div>
              </div>

              {/* SIDEBAR (Right, 1/3 width) - Alternatives */}
              <div className="space-y-4">
                <div className="bg-white/50 backdrop-blur-sm p-4 rounded-3xl border border-white/50 shadow-sm">
                  <h3 className="text-xs font-bold text-stone-500 uppercase tracking-widest mb-3 pl-2">Alternative Matches</h3>
                  <div className="space-y-3">
                    {/* Render Alternatives */}
                    {predictionResult.candidates
                      .filter((c: any) => c.name !== selectedCandidate.name)
                      .map((candidate: any, idx: number) => (
                        <div
                          key={idx}
                          onClick={() => setSelectedCandidate(candidate)}
                          className="group flex items-center justify-between p-3 bg-white border border-stone-100 rounded-2xl shadow-sm hover:shadow-md hover:border-emerald-500 cursor-pointer transition-all"
                        >
                          <div className="flex items-center gap-3">
                            <div className="w-12 h-12 rounded-xl bg-stone-100 flex items-center justify-center overflow-hidden border border-stone-200 group-hover:border-emerald-500 transition-colors">
                              <img
                                src={`/icons/${candidate.name.toLowerCase().replace(/ /g, '_')}.png`}
                                alt={candidate.name}
                                className="w-full h-full object-cover"
                                onError={(e) => {
                                  e.currentTarget.style.display = 'none';
                                  e.currentTarget.parentElement!.innerText = '🐾';
                                }}
                              />
                            </div>
                            <div>
                              <p className="font-bold text-stone-700 leading-tight group-hover:text-emerald-700">{candidate.name}</p>
                              <div className="flex items-center gap-2">
                                <p className="text-[10px] text-stone-400 font-mono tracking-wider">{candidate.score.toFixed(1)}%</p>
                                {observationLocation && candidate.taxon_id && (
                                  <SidebarRangeBadge
                                    taxonId={candidate.taxon_id}
                                    lat={observationLocation.lat}
                                    lng={observationLocation.lng}
                                  />
                                )}
                              </div>
                            </div>
                          </div>
                          <div className="opacity-0 group-hover:opacity-100 transition-opacity text-emerald-500">
                            <Search size={16} />
                          </div>
                        </div>
                      ))}

                    {/* If no alternatives, show placeholder */}
                    {predictionResult.candidates.length === 1 && (
                      <div className="p-4 text-center text-stone-400 text-sm italic">
                        No other close matches found.
                      </div>
                    )}
                  </div>
                </div>

                {/* Mini Stats or Info Tile */}
                <div className="bg-emerald-900 text-emerald-100 p-6 rounded-3xl shadow-lg relative overflow-hidden">
                  <div className="absolute top-0 right-0 p-4 opacity-10">
                    <Camera size={64} />
                  </div>
                  <p className="text-xs font-bold uppercase tracking-widest opacity-60 mb-2">Did you know?</p>
                  <p className="text-sm font-medium leading-relaxed">
                    {[
                      <>Our AI analyzes over <span className="text-white font-bold">1,000 unique features</span> to distinguish between similar species like the Gray Wolf and Coyote.</>,
                      <>The model was trained on <span className="text-white font-bold">50,000+ images</span> from verified wildlife photographers across North America.</>,
                      <>Location data helps boost accuracy by <span className="text-white font-bold">15%</span> — a Grizzly in Alaska is more likely than one in Florida!</>,
                      <>Our <span className="text-white font-bold">OOGWAY architecture</span> combines two neural networks for maximum precision.</>,
                      <>Black Bears can be <span className="text-white font-bold">brown, cinnamon, or blonde</span> — fur color doesn't always match the name!</>,
                      <>The AI checks <span className="text-white font-bold">iNaturalist's 50M+ observations</span> to verify if a species is in your area.</>,
                      <>We use <span className="text-white font-bold">YOLOv8</span> to automatically find and crop animals before identification.</>,
                    ][Math.floor(Math.random() * 7)]}
                  </p>
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

            <div className="flex justify-center gap-8 text-xs text-emerald-700/80 bg-stone-200/50 py-3 rounded-xl max-w-xs mx-auto border border-stone-200">
              <div className="flex flex-col">
                <span className="font-bold text-lg">90.7%</span>
                <span className="uppercase tracking-wider opacity-70">Accuracy</span>
              </div>
              <div className="w-px bg-stone-300" />
              <div className="flex flex-col">
                <span className="font-bold text-lg">100</span>
                <span className="uppercase tracking-wider opacity-70">Species</span>
              </div>
            </div>

            {!user && (
              <p className="text-xs text-emerald-700/60 italic">
                🎯 Sign in to start your <span className="font-bold">AniDex</span> — can you discover all 100?
              </p>
            )}
          </div>
        )}
      </div>

      {/* Signup Nudge Modal */}
      {showSignupNudge && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4 animate-fade-in">
          <div className="bg-white rounded-3xl shadow-2xl max-w-md w-full p-8 relative">
            <button
              onClick={() => setShowSignupNudge(false)}
              className="absolute top-4 right-4 text-stone-400 hover:text-stone-600 text-xl"
            >
              ✕
            </button>

            <div className="text-center space-y-4">
              <div className="text-5xl">🎯</div>
              <h2 className="text-2xl font-black text-stone-800">Nice Discovery!</h2>
              <p className="text-stone-600">
                Create a free account to save your observations and build your <span className="font-bold text-emerald-600">AniDex</span>!
              </p>
              <p className="text-sm text-stone-500 italic">
                Can you discover all <span className="font-bold">100 North American species</span>?
              </p>

              <div className="pt-4 space-y-3">
                <button
                  onClick={() => {
                    setShowSignupNudge(false);
                    router.push('/auth');
                  }}
                  className="w-full py-3 bg-emerald-600 hover:bg-emerald-700 text-white font-bold rounded-xl transition-colors"
                >
                  Create Free Account
                </button>
                <button
                  onClick={() => setShowSignupNudge(false)}
                  className="w-full py-2 text-stone-500 hover:text-stone-700 text-sm"
                >
                  Maybe later
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}