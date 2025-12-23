"use client";

import { useState, useRef, useEffect } from "react";
import axios from "axios";
import {
  Camera, MapPin, Loader2, AlertTriangle,
  Map as MapIcon, CheckCircle, Search, PawPrint, Share2, Leaf, Lock, X, Fingerprint, Calendar
} from "lucide-react";
import dynamic from "next/dynamic";
import "leaflet/dist/leaflet.css";

const SpeciesMap = dynamic(() => import("../components/SpeciesMap"), { ssr: false });
const LocationPicker = dynamic(() => import("../components/LocationPicker"), { ssr: false });

// --- TYPES ---
interface AnimalEntry {
  id: string;
  name: string;
  timestamp: Date;
  imageUrl?: string;
  verified?: boolean;
  taxonId?: number;
  lat?: number;
  lng?: number;
  isLocked?: boolean;
}

interface SpeciesInfo {
  name: string;
  scientific_name?: string;
  description?: string;
  origin?: string;
  category: "Native" | "Invasive";
}

export default function Home() {
  // --- STATE ---
  const [activeTab, setActiveTab] = useState<"camera" | "anidex">("camera");
  const [filter, setFilter] = useState<"All" | "Native" | "Invasive">("All");

  const [allSpecies, setAllSpecies] = useState<SpeciesInfo[]>([]);
  const [anidex, setAnidex] = useState<AnimalEntry[]>([]);
  const [selectedEntry, setSelectedEntry] = useState<AnimalEntry | null>(null);

  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [detailsLoading, setDetailsLoading] = useState(false);

  const [location, setLocation] = useState<{ lat: number; lng: number } | null>(null);
  const [showLocationPicker, setShowLocationPicker] = useState(false);
  const [locationMode, setLocationMode] = useState<"gps" | "map">("gps");

  const fileInputRef = useRef<HTMLInputElement>(null);

  // 1. INITIAL LOAD
  useEffect(() => {
    navigator.geolocation.getCurrentPosition(
      (pos) => setLocation({ lat: pos.coords.latitude, lng: pos.coords.longitude }),
      (err) => console.log("Auto-GPS failed", err)
    );

    const fetchSpecies = async () => {
      try {
        const res = await axios.get("http://127.0.0.1:8000/species");
        setAllSpecies(res.data);
      } catch (e) {
        console.error("Backend offline?", e);
      }
    };
    fetchSpecies();
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setImage(e.target.files[0]);
      setPreview(URL.createObjectURL(e.target.files[0]));
    }
  };

  const manualGetLocation = () => {
    setLocationMode("gps");
    navigator.geolocation.getCurrentPosition(
      (pos) => setLocation({ lat: pos.coords.latitude, lng: pos.coords.longitude }),
      (err) => alert("Could not get GPS. Please use the 'Set Map' button.")
    );
  };

  // 2. IDENTIFY LOGIC (Smart Location Validation)
  const identifyAnimal = async () => {
    if (!image) return;
    if (!location) {
      alert("Location is required for smart verification.");
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", image);

      // 1. Get Top 3 Guesses from AI
      const aiRes = await axios.post("http://127.0.0.1:8000/predict", formData);
      const candidates = aiRes.data.candidates; // Expects [{name: "Wolf", score: 90}, {name: "Coyote", score: 10}]

      if (!candidates || candidates.length === 0) {
        alert("AI could not identify this animal.");
        setLoading(false);
        return;
      }

      let confirmedMatch = null;
      let bestTaxonId = 0;

      // 2. Loop through candidates to find the first one that lives here
      for (const candidate of candidates) {
        try {
          // A. Get Taxon ID from iNat
          const taxRes = await axios.get(`https://api.inaturalist.org/v1/taxa?q=${candidate.name}&per_page=1`);

          if (taxRes.data.results.length > 0) {
            const taxonId = taxRes.data.results[0].id;

            // B. Check for LOCAL sightings (Wild & Verified only)
            const obsRes = await axios.get(
              `https://api.inaturalist.org/v1/observations?lat=${location.lat}&lng=${location.lng}&radius=100&taxon_id=${taxonId}&captive=false&quality_grade=research&per_page=1`
            );

            // C. If found locally, we have a winner!
            if (obsRes.data.total_results > 0) {
              confirmedMatch = candidate.name;
              bestTaxonId = taxonId;
              console.log(`✅ Confirmed Local Match: ${candidate.name}`);
              break; // Stop looking, we found the best local match
            } else {
              console.log(`❌ ${candidate.name} rejected: No local sightings.`);
            }
          }
        } catch (e) {
          console.error("Verification check failed for", candidate.name);
        }
      }

      // 3. Fallback: If AI is 99% sure but no local data, trust the AI (or if verify failed)
      // Otherwise, use the verified match we just found
      let finalPrediction = confirmedMatch || candidates[0].name;

      // If we fell back to the first choice, we still need its ID for the map
      if (!confirmedMatch) {
        try {
          const taxRes = await axios.get(`https://api.inaturalist.org/v1/taxa?q=${finalPrediction}&per_page=1`);
          if (taxRes.data.results.length > 0) bestTaxonId = taxRes.data.results[0].id;
        } catch (e) { }
      }

      // 4. Save to Collection
      const newEntry: AnimalEntry = {
        id: Date.now().toString(),
        name: finalPrediction,
        timestamp: new Date(),
        imageUrl: preview || "",
        verified: !!confirmedMatch, // It's verified if we found it in the loop
        taxonId: bestTaxonId,
        lat: location.lat,
        lng: location.lng,
        isLocked: false
      };

      setAnidex(prev => [newEntry, ...prev]);
      setSelectedEntry(newEntry);
      setActiveTab("anidex");

      // Clear camera
      setImage(null);
      setPreview(null);

    } catch (error) {
      alert("Backend Error. Is Python running?");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  // 3. HANDLE CLICKS
  const handleSpeciesClick = async (speciesName: string) => {
    const existingEntry = anidex.find(e => e.name === speciesName);

    if (existingEntry) {
      setSelectedEntry(existingEntry);
      return;
    }

    setDetailsLoading(true);
    try {
      // FIX: Use scientific name if available to avoid "Arctic Fox" -> "Red Fox" confusion
      const speciesInfo = allSpecies.find(s => s.name === speciesName);
      const queryName = speciesInfo?.scientific_name || speciesName;

      const taxRes = await axios.get(`https://api.inaturalist.org/v1/taxa?q=${queryName}&per_page=1`);
      let taxonId = 0;
      if (taxRes.data.results.length > 0) {
        taxonId = taxRes.data.results[0].id;
      }

      const lockedEntry: AnimalEntry = {
        id: "locked",
        name: speciesName,
        timestamp: new Date(),
        taxonId: taxonId,
        lat: location?.lat || 28.0,
        lng: location?.lng || -81.0,
        isLocked: true
      };

      setSelectedEntry(lockedEntry);

    } catch (e) { console.error(e); } finally { setDetailsLoading(false); }
  };

  const getFilteredList = () => {
    let list = [...allSpecies];
    if (filter === "Native") list = list.filter(s => s.category === "Native");
    if (filter === "Invasive") list = list.filter(s => s.category === "Invasive");
    return list.sort((a, b) => a.name.localeCompare(b.name));
  };

  const isUnlocked = (name: string) => anidex.find(e => e.name === name);
  const getObservationHistory = (name: string) => anidex.filter(e => e.name === name);
  const getStaticInfo = (name: string) => allSpecies.find(s => s.name === name);

  const getIconPath = (name: string) => {
    const slug = name.toLowerCase().replace(/ /g, "_");
    return `/icons/${slug}.png`;
  };

  const getContainerClass = () => {
    if (activeTab === "camera") return "max-w-5xl"; // HUGE for Laptop Camera
    if (selectedEntry) return "max-w-5xl";
    return "max-w-[1600px]";
  };

  return (
    <main className="min-h-screen bg-stone-100 text-stone-900 pb-24 font-sans">

      {/* HEADER */}
      {/* Changed background to bg-stone-900 (Dark Charcoal/Earth) for less color clashing */}
      <header className="bg-stone-900 text-stone-50 p-4 shadow-lg sticky top-0 z-50 border-b border-stone-800">
        <div className={`flex justify-between items-center mx-auto transition-all duration-300 ${getContainerClass()}`}>

          {/* NEW LOGO: AniML (No Box) */}
          <div className="flex items-center gap-4">

            {/* THE ICON CONTAINER */}
            <div className="relative w-14 h-14 flex items-center justify-center">
              <img
                src="/animl-logo.png"
                alt="AniML Logo"
                className="w-full h-full object-contain"
              />
            </div>

            {/* THE TEXT */}
            <div className="flex flex-col leading-none justify-center">
              <h1 className="text-3xl font-black tracking-tighter text-white">
                Ani<span className="text-emerald-400 font-mono">ML</span>
              </h1>
              {/* Subtext color tweaked slightly to pop against dark grey */}
              <span className="text-[10px] font-bold text-emerald-500 tracking-[0.2em] uppercase mt-0.5">
                Computer Vision
              </span>
            </div>
          </div>

          {/* Progress Counter - Updated colors to match dark grey theme */}
          {activeTab === "anidex" && (
            <div className="text-xs bg-stone-800 border border-stone-700 px-3 py-1.5 rounded-full font-mono text-stone-300 shadow-sm flex items-center gap-2">
              <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>
              <span className="text-white font-bold">{new Set(anidex.map(a => a.name)).size}</span> / {allSpecies.length} IDENTIFIED
            </div>
          )}
        </div>
      </header>

      {/* Main Container */}
      <div className={`mx-auto p-4 transition-all duration-300 ${getContainerClass()}`}>

        {/* --- CAMERA TAB --- */}
        {activeTab === "camera" && (
          <div className="flex flex-col gap-4 animate-in fade-in">
            <div className="flex justify-between items-center bg-white p-3 rounded-xl shadow-sm border border-stone-200">
              <div className="flex items-center gap-2">
                <MapPin size={18} className={location ? "text-emerald-600" : "text-red-400"} />
                <div className="text-xs">
                  <p className="font-bold text-stone-700">{location ? "Location Locked" : "No Location"}</p>
                  <p className="text-stone-400">{location ? `${location.lat.toFixed(2)}, ${location.lng.toFixed(2)}` : "Required for AI"}</p>
                </div>
              </div>
              <div className="flex gap-2">
                <button onClick={manualGetLocation} className={`text-xs px-3 py-2 rounded-lg font-bold transition-all ${locationMode === "gps" ? "bg-emerald-600 text-white shadow-md" : "bg-stone-100 text-stone-500 hover:bg-stone-200"}`}>GPS</button>
                <button onClick={() => { setShowLocationPicker(true); setLocationMode("map"); }} className={`text-xs px-3 py-2 rounded-lg font-bold transition-all ${locationMode === "map" ? "bg-emerald-600 text-white shadow-md" : "bg-stone-100 text-stone-500 hover:bg-stone-200"}`}>Set Map</button>
              </div>
            </div>

            {/* PREVIEW BOX */}
            <div
              onClick={() => fileInputRef.current?.click()}
              className="aspect-[4/5] md:aspect-video w-full bg-stone-200 rounded-3xl border-4 border-dashed border-stone-300 flex flex-col items-center justify-center cursor-pointer hover:border-emerald-500 hover:bg-stone-100 transition-colors overflow-hidden relative shadow-inner"
            >
              {preview ? (
                <img src={preview} className="w-full h-full object-contain bg-black/5" />
              ) : (
                <div className="text-stone-400 flex flex-col items-center">
                  <div className="bg-white p-6 rounded-full shadow-sm mb-4">
                    <Camera size={48} className="text-emerald-600" />
                  </div>
                  <p className="font-bold text-lg text-stone-600">Tap to Upload Photo</p>
                  <p className="text-sm opacity-70">JPG or PNG</p>
                </div>
              )}
            </div>
            <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept="image/*" />

            <button onClick={identifyAnimal} disabled={!image || loading} className="w-full py-4 bg-emerald-800 hover:bg-emerald-700 text-white rounded-xl font-bold text-lg shadow-xl flex justify-center items-center gap-2 disabled:opacity-50 transition-transform active:scale-95">
              {loading ? <Loader2 className="animate-spin" /> : <Search />} {loading ? "Analyzing..." : "Identify"}
            </button>
          </div>
        )}

        {/* --- ANIDEX GRID --- */}
        {activeTab === "anidex" && !selectedEntry && (
          <div className="animate-in slide-in-from-right">
            <div className="flex bg-stone-200 p-1 rounded-xl mb-6 max-w-md mx-auto">
              {(["All", "Native", "Invasive"] as const).map((f) => (
                <button key={f} onClick={() => setFilter(f)} className={`flex-1 py-2 text-xs font-bold rounded-lg transition-all ${filter === f ? "bg-white text-emerald-800 shadow-sm" : "text-stone-500 hover:text-stone-700"}`}>{f}</button>
              ))}
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 md:gap-6">
              {getFilteredList().map((species) => {
                const found = isUnlocked(species.name);
                return (
                  <div key={species.name} onClick={() => !detailsLoading && handleSpeciesClick(species.name)} className={`aspect-square rounded-2xl border flex flex-col items-center justify-between p-3 relative overflow-hidden transition-all cursor-pointer hover:scale-105 hover:shadow-lg active:scale-95 ${found ? "bg-white border-emerald-500 shadow-md" : "bg-stone-100 border-stone-200"}`}>

                    <div className="flex-1 w-full flex items-center justify-center">
                      {found ? (
                        <>
                          <img src={getIconPath(species.name)} className="w-[85%] h-[85%] object-contain drop-shadow-md" alt={species.name} />
                          {species.category === "Invasive" && <div className="absolute top-2 right-2 bg-red-500 w-3 h-3 rounded-full ring-2 ring-white shadow-sm"></div>}
                        </>
                      ) : (
                        <>
                          {detailsLoading ? <Loader2 className="animate-spin text-stone-400" size={24} /> :
                            <img src={getIconPath(species.name)} className="w-[75%] h-[75%] opacity-20 object-contain grayscale brightness-50" alt="locked" />
                          }
                        </>
                      )}
                    </div>

                    <p className={`text-[10px] md:text-xs font-bold text-center leading-tight w-full truncate ${found ? "text-emerald-900" : "text-stone-400"}`}>
                      {species.name}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* --- FIELD GUIDE DETAIL VIEW --- */}
        {selectedEntry && (
          <div className="animate-in slide-in-from-bottom pb-10 max-w-4xl mx-auto">
            <button onClick={() => setSelectedEntry(null)} className="text-sm font-bold text-stone-500 mb-4 flex items-center gap-1">← Back to Collection</button>

            <div className="bg-white rounded-3xl shadow-xl overflow-hidden border border-stone-100 flex flex-col md:flex-row h-auto md:h-[500px]">

              <div className="w-full md:w-1/2 p-6 flex flex-col relative border-b md:border-b-0 md:border-r border-stone-100">
                <div className="mb-4">
                  <h2 className="text-3xl font-bold text-emerald-950 leading-tight">{selectedEntry.name}</h2>
                  <p className="text-stone-500 text-sm italic">{getStaticInfo(selectedEntry.name)?.scientific_name || "Species Name"}</p>
                </div>

                <div className="flex gap-2 mb-6">
                  {!selectedEntry.isLocked ? (
                    selectedEntry.verified
                      ? <span className="text-green-700 bg-green-100 text-xs px-3 py-1 rounded-full font-bold flex items-center gap-1"><CheckCircle size={14} /> Verified Local</span>
                      : <span className="text-yellow-700 bg-yellow-100 text-xs px-3 py-1 rounded-full font-bold flex items-center gap-1"><AlertTriangle size={14} /> Out of Range</span>
                  ) : (
                    <span className="text-stone-500 bg-stone-100 text-xs px-3 py-1 rounded-full font-bold flex items-center gap-1">Not Yet Observed</span>
                  )}

                  <span className={`text-[10px] font-bold px-3 py-1 rounded-full border flex items-center ${getStaticInfo(selectedEntry.name)?.category === "Invasive" ? "bg-red-50 text-red-700 border-red-100" : "bg-emerald-50 text-emerald-700 border-emerald-100"}`}>
                    {getStaticInfo(selectedEntry.name)?.category.toUpperCase() || "NATIVE"}
                  </span>
                </div>

                <div className="flex-1 bg-stone-50 rounded-2xl flex items-center justify-center border border-stone-100 mb-6 min-h-[150px] p-6 relative overflow-hidden">
                  <img
                    src={getIconPath(selectedEntry.name)}
                    className={`w-full h-full object-contain drop-shadow-lg ${selectedEntry.isLocked ? "grayscale opacity-50" : ""}`}
                    alt={selectedEntry.name}
                  />
                </div>

                <div className="text-sm text-stone-600 leading-relaxed overflow-y-auto max-h-[100px] md:max-h-none">
                  {getStaticInfo(selectedEntry.name)?.description}
                </div>
              </div>

              <div className="w-full md:w-1/2 bg-stone-900 relative h-[300px] md:h-auto">
                {selectedEntry.taxonId ? (
                  <SpeciesMap
                    taxonId={selectedEntry.taxonId}
                    userLat={location?.lat || 28.0}
                    userLng={location?.lng || -81.0}
                    animalName={selectedEntry.name}
                    observations={getObservationHistory(selectedEntry.name).map(o => ({
                      lat: o.lat || 0, lng: o.lng || 0, date: o.timestamp
                    })).filter(o => o.lat !== 0)}
                  />
                ) : (
                  <div className="h-full flex items-center justify-center text-stone-500 text-xs">Map Unavailable</div>
                )}
              </div>
            </div>

            {!selectedEntry.isLocked && (
              <div className="mt-8">
                <h3 className="font-bold text-stone-700 mb-4 flex items-center gap-2 text-sm uppercase tracking-wider">
                  <Calendar size={18} /> Your Field Observations
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {getObservationHistory(selectedEntry.name).map((obs) => (
                    <div key={obs.id} className="bg-white p-3 rounded-xl border border-stone-200 shadow-sm flex gap-4 items-center hover:border-emerald-400 transition-colors cursor-pointer">
                      <img src={obs.imageUrl} className="w-16 h-16 rounded-lg object-cover bg-stone-200" />
                      <div className="flex-1">
                        <p className="font-bold text-stone-800">Observed in Field</p>
                        <p className="text-xs text-stone-500">{obs.timestamp.toLocaleString()}</p>
                        <div className="flex gap-2 mt-1">
                          <span className="text-[10px] bg-stone-100 text-stone-500 px-2 py-0.5 rounded-md">
                            {obs.lat?.toFixed(3)}, {obs.lng?.toFixed(3)}
                          </span>
                        </div>
                      </div>
                      {obs.verified && <CheckCircle size={20} className="text-emerald-500" />}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

      </div>

      {showLocationPicker && (
        <div className="fixed inset-0 z-[100] bg-black/80 flex items-center justify-center p-4">
          <div className="bg-white w-full max-w-md rounded-2xl overflow-hidden h-[500px] flex flex-col shadow-2xl">
            <div className="p-4 bg-emerald-900 text-white flex justify-between items-center shrink-0">
              <h3 className="font-bold flex items-center gap-2"><MapIcon size={18} /> Set Location</h3>
              <button onClick={() => setShowLocationPicker(false)} className="p-1 hover:bg-white/20 rounded-full"><X size={20} /></button>
            </div>
            <div className="flex-1 relative w-full bg-stone-100"><LocationPicker onLocationSelect={(lat, lng) => setLocation({ lat, lng })} /></div>
            <div className="p-3 bg-white border-t border-stone-200 flex justify-between items-center shrink-0">
              <p className="text-xs text-stone-500">Tap map to place pin</p>
              <button onClick={() => setShowLocationPicker(false)} className="bg-emerald-600 text-white text-xs font-bold px-4 py-2 rounded-lg">Confirm Location</button>
            </div>
          </div>
        </div>
      )}

      {!selectedEntry && (
        <nav className="fixed bottom-0 left-0 right-0 bg-white border-t border-stone-200 pb-safe pt-2 z-40">
          <div className="max-w-md mx-auto flex justify-around items-center px-6">
            <button onClick={() => setActiveTab("camera")} className={`p-3 rounded-xl flex flex-col items-center transition-all ${activeTab === "camera" ? "text-emerald-700 bg-emerald-50" : "text-stone-400 hover:text-stone-600"}`}><Camera size={24} /><span className="text-[10px] font-bold mt-1">IDENTIFY</span></button>
            <button onClick={() => setActiveTab("anidex")} className={`p-3 rounded-xl flex flex-col items-center transition-all ${activeTab === "anidex" ? "text-emerald-700 bg-emerald-50" : "text-stone-400 hover:text-stone-600"}`}><Search size={24} /><span className="text-[10px] font-bold mt-1">ANIDEX</span></button>
          </div>
        </nav>
      )}
    </main>
  );
}