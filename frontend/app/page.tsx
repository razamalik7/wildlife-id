"use client";

import { useState, useRef, useEffect } from "react";
import axios from "axios";
import {
  Camera, MapPin, Loader2, AlertTriangle,
  Map as MapIcon, CheckCircle, Search, PawPrint, Share2, Leaf, Lock, X, Fingerprint, Calendar,
  Globe, Dna, ExternalLink
} from "lucide-react";
import dynamic from "next/dynamic";
import "leaflet/dist/leaflet.css";

const SpeciesMap = dynamic(() => import("../components/SpeciesMap"), { ssr: false });
const LocationPicker = dynamic(() => import("../components/LocationPicker"), { ssr: false });
const ObservationLocationPicker = dynamic(() => import("../components/ObservationLocationPicker"), { ssr: false });

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
  candidates?: string[];
  isStumped?: boolean;
}

interface SpeciesInfo {
  name: string;
  scientific_name?: string;
  description?: string;
  origin?: string;
  category: "Native" | "Invasive";
  iucn?: string;
  hotspots?: string[];
  taxonomy?: {
    kingdom?: string;
    phylum?: string;
    class?: string;
    order?: string;
    family?: string;
    genus?: string;
    species?: string;
    taxon_id?: number;
  };
}

interface Park {
  id: number;
  name: string;
  state: string;
  country?: string; // "US", "CA", "MX", etc.
  lat: number;
  lng: number;
}

export default function Home() {
  // --- STATE ---
  const [activeTab, setActiveTab] = useState<"camera" | "anidex">("camera");
  const [filter, setFilter] = useState<"Total" | "Native" | "Invasive">("Total");

  const [allSpecies, setAllSpecies] = useState<SpeciesInfo[]>([]);
  const [anidex, setAnidex] = useState<AnimalEntry[]>([]);
  const [selectedEntry, setSelectedEntry] = useState<AnimalEntry | null>(null);

  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [detailsLoading, setDetailsLoading] = useState(false);

  const [location, setLocation] = useState<{ lat: number; lng: number } | null>(null);
  const [observationLocation, setObservationLocation] = useState<{ lat: number; lng: number } | null>(null);
  const [showLocationPicker, setShowLocationPicker] = useState(false);
  const [locationMode, setLocationMode] = useState<"gps" | "map">("gps");
  const [isLocating, setIsLocating] = useState(true);
  const [detailTab, setDetailTab] = useState<"map" | "where" | "taxonomy">("map");
  const [locationNames, setLocationNames] = useState<Record<string, string>>({});
  const [searchRadius, setSearchRadius] = useState<number>(50); // miles
  const [nearbyHotspots, setNearbyHotspots] = useState<Array<{ name: string; count: number; lat: number; lng: number; distance: number }>>([]);
  const [hotspotsLoading, setHotspotsLoading] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // 1. INITIAL LOAD
  useEffect(() => {
    setIsLocating(true);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setLocation({ lat: pos.coords.latitude, lng: pos.coords.longitude });
        setIsLocating(false);
      },
      (err) => {
        console.log("Auto-GPS failed", err);
        setIsLocating(false);
      },
      { timeout: 10000 } // Don't hang forever
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

  // Auto-fetch hotspots when "Where to See" tab is opened
  useEffect(() => {
    // Reset hotspots when changing species
    if (selectedEntry) {
      setNearbyHotspots([]);
    }
  }, [selectedEntry?.name]);

  useEffect(() => {
    if (detailTab === "where" && selectedEntry && allSpecies.length > 0 && !hotspotsLoading && location) {
      const taxonId = allSpecies.find(s => s.name === selectedEntry.name)?.taxonomy?.taxon_id;
      if (taxonId) {
        // Only fetch if empty or if radius changed (implied by dependency)
        // We rely on loading state to prevent dupes
        fetchNearbyHotspots(taxonId, searchRadius);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detailTab, selectedEntry?.name, allSpecies.length, location, searchRadius]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setImage(e.target.files[0]);
      setPreview(URL.createObjectURL(e.target.files[0]));
    }
  };

  const manualGetLocation = () => {
    setLocationMode("gps");
    setIsLocating(true);
    setLocation(null); // Clear old location to show we are trying
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setLocation({ lat: pos.coords.latitude, lng: pos.coords.longitude });
        setIsLocating(false);
      },
      (err) => {
        alert("Could not get GPS. Please use the 'Set Map' button.");
        setIsLocating(false);
      }
    );
  };

  // 2. IDENTIFY LOGIC (Smart Location Validation)
  const identifyAnimal = async () => {
    if (!image) return;
    if (!observationLocation) {
      alert("Please set the observation location on the map first.");
      return;
    }

    // Use observation location for verification
    const verifyLat = observationLocation.lat;
    const verifyLng = observationLocation.lng;

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", image);

      // 1. Get Top 3 Guesses from AI
      const aiRes = await axios.post("http://127.0.0.1:8000/predict", formData);
      const candidates: { name: string; score: number }[] = aiRes.data.candidates;

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
              `https://api.inaturalist.org/v1/observations?lat=${verifyLat}&lng=${verifyLng}&radius=100&taxon_id=${taxonId}&captive=false&quality_grade=research&per_page=1`
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
        lat: verifyLat,
        lng: verifyLng,
        isLocked: false,
        candidates: candidates.map(c => c.name) // Store all guesses for "Try Again" feature
      };

      setAnidex(prev => [newEntry, ...prev]);
      setSelectedEntry(newEntry);
      setActiveTab("anidex");

      // Clear camera
      setImage(null);
      setPreview(null);
      setObservationLocation(null);

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
      console.log(`[DEBUG] handleSpeciesClick: Querying iNat for: ${queryName}`);

      const taxRes = await axios.get(`https://api.inaturalist.org/v1/taxa?q=${queryName}&per_page=5`);
      let taxonId = 0;

      if (taxRes.data.results.length > 0) {
        // 1. Try to find EXACT math on Scientific Name + Species Rank
        const exactMatch = taxRes.data.results.find((t: any) =>
          (t.name.toLowerCase() === queryName.toLowerCase() || t.preferred_common_name?.toLowerCase() === queryName.toLowerCase())
          && t.rank === "species"
        );

        if (exactMatch) {
          taxonId = exactMatch.id;
          console.log(`[DEBUG] Found Exact Species Match: ${exactMatch.name} (${exactMatch.id})`);
        } else {
          // 2. Fallback to first result but warn
          taxonId = taxRes.data.results[0].id;
          console.warn(`[DEBUG] No Exact Species Match. Using Best Guess: ${taxRes.data.results[0].name} (${taxonId})`);
        }
      } else {
        console.warn(`[DEBUG] handleSpeciesClick: NO Taxon found for ${queryName}`);
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

  // Reverse geocoding to get location name from coordinates
  const getLocationName = async (lat: number, lng: number): Promise<string> => {
    const key = `${lat.toFixed(3)},${lng.toFixed(3)}`;

    // Check cache first
    if (locationNames[key]) return locationNames[key];

    try {
      const res = await fetch(
        `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lng}&format=json`,
        { headers: { "User-Agent": "AniML-App/1.0" } }
      );
      const data = await res.json();
      const addr = data.address || {};

      // Build location string: City/Town, State, Country
      const city = addr.city || addr.town || addr.village || addr.county || "";
      const state = addr.state || "";
      const country = addr.country_code?.toUpperCase() || "";

      const locationStr = [city, state, country].filter(Boolean).join(", ");

      // Cache the result
      setLocationNames(prev => ({ ...prev, [key]: locationStr }));
      return locationStr || `${lat.toFixed(2)}, ${lng.toFixed(2)}`;
    } catch {
      return `${lat.toFixed(2)}, ${lng.toFixed(2)}`;
    }
  };

  // Get related species from same family
  const getRelatedSpecies = (name: string) => {
    const info = getStaticInfo(name);
    const family = info?.taxonomy?.family;
    if (!family) return [];
    return allSpecies.filter(s => s.taxonomy?.family === family && s.name !== name);
  };

  // Calculate distance between two coordinates (Haversine formula)
  const getDistance = (lat1: number, lng1: number, lat2: number, lng2: number): number => {
    const R = 3959; // Earth radius in miles
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLng = (lng2 - lng1) * Math.PI / 180;
    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
      Math.sin(dLng / 2) * Math.sin(dLng / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  };

  // Fetch observation counts for curated major parks from iNaturalist
  const fetchNearbyHotspots = async (taxonId: number, radiusMiles: number) => {
    if (!location || !taxonId) return;

    setHotspotsLoading(true);
    setNearbyHotspots([]);

    try {
      // Fetch parks from external JSON
      setHotspotsLoading(true);
      const parksResp = await fetch('/parks.json');
      if (!parksResp.ok) throw new Error('Failed to load parks data');
      const allParks: Park[] = await parksResp.json();


      // Filter parks by distance and country based on selected radius
      let parksToQuery = allParks;

      // Special handling for "USA" filter (radiusMiles === 2000)
      if (radiusMiles === 2000) {
        parksToQuery = allParks.filter(park => park.country === "US");
      } else if (radiusMiles < 2000) {
        // Distance-based filter for smaller radii
        parksToQuery = allParks.filter(park => {
          const dist = getDistance(location.lat, location.lng, park.lat, park.lng);
          return dist <= radiusMiles;
        });
      }
      // radiusMiles >= 10000 (Global) uses all parks, no filter needed

      // If no parks in range, return "no parks" message
      if (parksToQuery.length === 0) {
        setHotspotsLoading(false);
        return;
      }

      // Query each park for exact observation count
      const parkCounts: Array<{ name: string; count: number; lat: number; lng: number; distance: number }> = [];

      // Query parks in parallel (batch of 5 at a time to avoid rate limiting)
      // Added 250ms delay between batches to prevent dropping requests (Yellowstone/Banff missing glitch)
      const batchSize = 5;
      for (let i = 0; i < parksToQuery.length; i += batchSize) {
        const batch = parksToQuery.slice(i, i + batchSize);
        const promises = batch.map(async (park) => {
          try {
            const countUrl = `https://api.inaturalist.org/v1/observations?taxon_id=${taxonId}&place_id=${park.id}&quality_grade=research&per_page=0`;
            const countResp = await axios.get(countUrl);
            const totalCount = countResp.data.total_results || 0;

            if (totalCount > 0) {
              return {
                name: `${park.name}, ${park.state}`,
                count: totalCount,
                lat: park.lat,
                lng: park.lng,
                distance: Math.round(getDistance(location.lat, location.lng, park.lat, park.lng))
              };
            }
            return null;
          } catch (e) {
            console.error(`[Hotspot Error] Failed to fetch data for ${park.name}:`, e);
            return null;
          }
        });

        const results = await Promise.all(promises);
        parkCounts.push(...results.filter(Boolean) as typeof parkCounts);

        // Throttle to respect API limits
        if (i + batchSize < parksToQuery.length) {
          await new Promise(resolve => setTimeout(resolve, 250));
        }
      }

      // Sort by observation count (descending)
      const sortedParks = parkCounts.sort((a, b) => b.count - a.count);

      console.log("Hotspots found:", sortedParks.length, "Top:", sortedParks.slice(0, 5).map(p => `${p.name}: ${p.count}`));

      // Take top 20 results to show Canada/Mexico
      const finalParks = sortedParks.slice(0, 20);

      setNearbyHotspots(finalParks);
    } catch (error) {
      console.error("Failed to fetch hotspots:", error);
    }

    setHotspotsLoading(false);
  };

  const getContainerClass = () => {
    if (activeTab === "camera") return "max-w-5xl"; // HUGE for Laptop Camera
    if (selectedEntry) return "max-w-7xl"; // Even wider for new layout
    return "max-w-[1600px]";
  };

  // 4. "TRY AGAIN" LOGIC
  const handleTryAgain = async (entry: AnimalEntry) => {
    if (!entry.candidates || entry.candidates.length === 0) return;

    const currentIndex = entry.candidates.indexOf(entry.name);
    const nextIndex = currentIndex + 1;

    // If we've shown all 3 guesses (current is index 2), mark as stumped
    if (nextIndex >= entry.candidates.length || nextIndex >= 3) {
      // Mark as stumped - remove from anidex (don't count as discovered)
      const stumpedEntry: AnimalEntry = {
        ...entry,
        name: "Unknown Animal",
        isStumped: true,
        taxonId: 0,
      };
      setAnidex(prev => prev.map(e => e.id === entry.id ? stumpedEntry : e));
      setSelectedEntry(stumpedEntry);
      return;
    }

    const nextName = entry.candidates[nextIndex];
    setDetailsLoading(true);

    let nextTaxonId = 0;
    try {
      const taxRes = await axios.get(`https://api.inaturalist.org/v1/taxa?q=${nextName}&per_page=1`);
      if (taxRes.data.results.length > 0) nextTaxonId = taxRes.data.results[0].id;
    } catch (e) { }

    const updatedEntry = {
      ...entry,
      name: nextName,
      taxonId: nextTaxonId,
    };

    setAnidex(prev => prev.map(e => e.id === entry.id ? updatedEntry : e));
    setSelectedEntry(updatedEntry);

    setDetailsLoading(false);
  };

  const getIUCNColor = (status?: string) => {
    switch (status) {
      case "Least Concern": return "bg-emerald-50 text-emerald-700 border-emerald-100";
      case "Near Threatened": return "bg-yellow-50 text-yellow-700 border-yellow-100";
      case "Vulnerable": return "bg-orange-50 text-orange-700 border-orange-100";
      case "Endangered": return "bg-red-50 text-red-700 border-red-100";
      case "Critically Endangered": return "bg-red-100 text-red-800 border-red-200";
      case "Extinct in the Wild": return "bg-stone-800 text-stone-100 border-stone-800";
      default: return "bg-stone-50 text-stone-500 border-stone-100";
    }
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
              <span className="text-white font-bold">
                {getFilteredList().filter(s => isUnlocked(s.name)).length}
              </span>
              <span className="text-stone-500 mx-1">/</span>
              {getFilteredList().length} {filter.toUpperCase()}
            </div>
          )}
        </div>
      </header>

      {/* Main Container */}
      <div className={`mx-auto p-4 transition-all duration-300 ${getContainerClass()}`}>

        {/* --- CAMERA TAB --- */}
        {activeTab === "camera" && (
          <div className="animate-in fade-in">
            {/* RESPONSIVE LAYOUT: Side-by-side on desktop (md+), stacked on mobile */}
            <div className="flex flex-col md:flex-row gap-4 md:gap-6 max-w-5xl mx-auto">

              {/* LEFT PANEL - Photo Upload */}
              <div className="flex-1 flex flex-col gap-4">
                {/* PREVIEW BOX */}
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="aspect-[4/5] md:aspect-[3/4] w-full bg-stone-200 rounded-3xl border-4 border-dashed border-stone-300 flex flex-col items-center justify-center cursor-pointer hover:border-emerald-500 hover:bg-stone-100 transition-colors overflow-hidden relative shadow-inner"
                >
                  {preview ? (
                    <img src={preview} className="w-full h-full object-contain bg-black/5" alt="Preview" />
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
                <input type="file" ref={fileInputRef} onChange={(e) => {
                  handleFileChange(e);
                  // Reset observation location when new photo is uploaded
                  setObservationLocation(null);
                }} className="hidden" accept="image/*" />

                {/* IDENTIFY BUTTON - Only on mobile (shown below map on desktop) */}
                <div className="md:hidden">
                  <button
                    onClick={identifyAnimal}
                    disabled={!image || !observationLocation || loading}
                    className="w-full py-4 bg-emerald-800 hover:bg-emerald-700 text-white rounded-xl font-bold text-lg shadow-xl flex justify-center items-center gap-2 disabled:opacity-50 transition-transform active:scale-95"
                  >
                    {loading ? <Loader2 className="animate-spin" /> : <Search />}
                    {loading ? "Analyzing..." : (!image ? "Upload Photo First" : (!observationLocation ? "Set Location First" : "Identify"))}
                  </button>
                </div>
              </div>

              {/* RIGHT PANEL - Location Picker Map (Always visible) */}
              <div className="flex-1 flex flex-col gap-4">
                <div className="flex-1 min-h-[400px] md:min-h-0 md:aspect-[3/4] bg-stone-900 rounded-3xl overflow-hidden shadow-xl border border-stone-700">
                  <ObservationLocationPicker
                    initialLat={location?.lat}
                    initialLng={location?.lng}
                    photoPreview={preview || undefined}
                    isConfirmed={!!observationLocation}
                    onConfirm={(lat, lng) => {
                      setObservationLocation({ lat, lng });
                    }}
                    onLocationChange={() => {
                      setObservationLocation(null);
                    }}
                  />
                </div>

                {/* Location Status Badge */}
                {observationLocation && (
                  <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-3 flex items-center gap-3">
                    <div className="bg-emerald-500 p-2 rounded-lg">
                      <MapPin size={16} className="text-white" />
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-bold text-emerald-800">Location Set</p>
                      <p className="text-xs text-emerald-600">{observationLocation.lat.toFixed(4)}, {observationLocation.lng.toFixed(4)}</p>
                    </div>
                    <button
                      onClick={() => setObservationLocation(null)}
                      className="text-emerald-600 hover:text-emerald-800 text-xs font-medium"
                    >
                      Change
                    </button>
                  </div>
                )}

                {/* IDENTIFY BUTTON - Desktop only (hidden on mobile) */}
                <div className="hidden md:block">
                  <button
                    onClick={identifyAnimal}
                    disabled={!image || !observationLocation || loading}
                    className="w-full py-4 bg-emerald-800 hover:bg-emerald-700 text-white rounded-xl font-bold text-lg shadow-xl flex justify-center items-center gap-2 disabled:opacity-50 transition-transform active:scale-95"
                  >
                    {loading ? <Loader2 className="animate-spin" /> : <Search />}
                    {loading ? "Analyzing..." : (!image ? "Upload Photo First" : (!observationLocation ? "Set Location First" : "Identify"))}
                  </button>
                </div>
              </div>
            </div>

            {/* OOGWAY ATTRIBUTION - CUSTOM BADGE */}
            <div className="text-center mt-6 opacity-70 hover:opacity-100 transition-opacity">
              <p className="text-[11px] font-mono uppercase tracking-[0.2em] font-bold text-emerald-900">
                Powered by Oogway
              </p>
              <p className="text-[10px] text-emerald-700/60 font-medium italic mt-1">
                Custom Architecture Originally Developed for AniML
              </p>
            </div>
          </div>
        )}

        {/* --- ANIDEX GRID --- */}
        {activeTab === "anidex" && !selectedEntry && (
          <div className="animate-in slide-in-from-right">
            <div className="flex bg-stone-200 p-1 rounded-xl mb-6 max-w-md mx-auto">
              {(["Total", "Native", "Invasive"] as const).map((f) => (
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
          <div className="animate-in slide-in-from-bottom pb-10">
            <button onClick={() => setSelectedEntry(null)} className="text-sm font-bold text-stone-500 mb-4 flex items-center gap-1">← Back to Collection</button>

            {/* STUMPED STATE */}
            {selectedEntry.isStumped ? (
              <div className="bg-white rounded-3xl shadow-xl overflow-hidden border border-stone-100 p-8 text-center">
                <div className="w-24 h-24 mx-auto mb-6 bg-stone-100 rounded-full flex items-center justify-center">
                  <PawPrint className="w-12 h-12 text-stone-400" />
                </div>
                <h2 className="text-2xl font-bold text-stone-700 mb-2">🤖 I'm Stumped!</h2>
                <p className="text-stone-500 mb-6">I couldn't identify this animal after my best 3 guesses.</p>

                {selectedEntry.imageUrl && (
                  <div className="mb-6">
                    <p className="text-sm text-stone-400 mb-2">Your photo:</p>
                    <img src={selectedEntry.imageUrl} alt="Uploaded" className="max-h-48 mx-auto rounded-xl border border-stone-200" />
                  </div>
                )}

                <p className="text-sm text-stone-400 mb-4">My guesses were: {selectedEntry.candidates?.join(", ")}</p>

                <button
                  onClick={() => {
                    // Remove this stumped entry from collection
                    setAnidex(prev => prev.filter(e => e.id !== selectedEntry.id));
                    setSelectedEntry(null);
                  }}
                  className="bg-stone-200 text-stone-700 px-6 py-2 rounded-full font-bold hover:bg-stone-300 transition-colors"
                >
                  Dismiss
                </button>
              </div>
            ) : (
              <div className="bg-white rounded-3xl shadow-xl overflow-hidden border border-stone-100">
                <div className="flex flex-col lg:flex-row">

                  {/* LEFT PANEL - Species Info (40%) */}
                  <div className="w-full lg:w-2/5 p-6 flex flex-col relative border-b lg:border-b-0 lg:border-r border-stone-100">
                    <div className="mb-4">
                      <h2 className="text-3xl font-bold text-emerald-950 leading-tight">{selectedEntry.name}</h2>
                      <p className="text-stone-500 text-sm italic">{getStaticInfo(selectedEntry.name)?.scientific_name || "Species Name"}</p>

                      {/* TAXONOMY PILLS */}
                      {(() => {
                        const info = getStaticInfo(selectedEntry.name);
                        const tax = info?.taxonomy;
                        if (tax && (tax.class || tax.order || tax.family)) {
                          return (
                            <div className="mt-2 flex flex-wrap gap-1 text-[10px]">
                              {tax.class && (
                                <span className="bg-blue-50 text-blue-700 px-2 py-0.5 rounded-full font-medium">
                                  {tax.class}
                                </span>
                              )}
                              {tax.order && (
                                <span className="bg-purple-50 text-purple-700 px-2 py-0.5 rounded-full font-medium">
                                  {tax.order}
                                </span>
                              )}
                              {tax.family && (
                                <span className="bg-amber-50 text-amber-700 px-2 py-0.5 rounded-full font-medium">
                                  {tax.family}
                                </span>
                              )}
                            </div>
                          );
                        }
                        return null;
                      })()}

                      {/* TRY AGAIN BUTTON */}
                      {!selectedEntry.isLocked && selectedEntry.candidates && (
                        <button
                          onClick={() => handleTryAgain(selectedEntry)}
                          className="mt-2 text-[10px] font-bold text-stone-400 border border-stone-200 px-2 py-1 rounded hover:bg-stone-100 hover:text-stone-600 transition-colors"
                        >
                          Wrong? Try next guess ↻
                        </button>
                      )}
                    </div>

                    {/* STATUS BADGES */}
                    <div className="flex flex-wrap gap-2 mb-4">
                      {!selectedEntry.isLocked ? (
                        selectedEntry.verified
                          ? <span className="text-green-700 bg-green-100 text-xs px-3 py-1 rounded-full font-bold flex items-center gap-1"><CheckCircle size={14} /> Verified Local</span>
                          : <span className="text-yellow-700 bg-yellow-100 text-xs px-3 py-1 rounded-full font-bold flex items-center gap-1"><AlertTriangle size={14} /> Out of Range</span>
                      ) : (
                        <span className="text-stone-500 bg-stone-100 text-xs px-3 py-1 rounded-full font-bold flex items-center gap-1">Not Yet Observed</span>
                      )}
                      <span className={`text-[10px] font-bold px-3 py-1 rounded-full border flex items-center ${getStaticInfo(selectedEntry.name)?.category === "Invasive" ? "bg-red-50 text-red-700 border-red-100" : "bg-emerald-50 text-emerald-700 border-emerald-100"}`}>
                        {getStaticInfo(selectedEntry.name)?.category?.toUpperCase() || "NATIVE"}
                      </span>
                      {getStaticInfo(selectedEntry.name)?.iucn && (
                        <span className={`text-[10px] font-bold px-3 py-1 rounded-full border flex items-center ${getIUCNColor(getStaticInfo(selectedEntry.name)?.iucn)}`}>
                          {getStaticInfo(selectedEntry.name)?.iucn?.toUpperCase()}
                        </span>
                      )}
                    </div>

                    {/* SPECIES IMAGE */}
                    <div className="flex-1 bg-stone-50 rounded-2xl flex items-center justify-center border border-stone-100 mb-4 min-h-[150px] p-4 relative overflow-hidden">
                      <img
                        src={getIconPath(selectedEntry.name)}
                        className={`w-full h-full object-contain drop-shadow-lg max-h-[180px] ${selectedEntry.isLocked ? "grayscale opacity-50" : ""}`}
                        alt={selectedEntry.name}
                      />
                    </div>

                    {/* DESCRIPTION */}
                    <div className="text-sm text-stone-600 leading-relaxed">
                      {getStaticInfo(selectedEntry.name)?.description}
                    </div>
                  </div>

                  {/* RIGHT PANEL - Tabbed Content (60%) */}
                  <div className="w-full lg:w-3/5 flex flex-col">
                    {/* TAB HEADERS */}
                    <div className="flex border-b border-stone-200">
                      <button
                        onClick={() => setDetailTab("map")}
                        className={`flex-1 py-3 px-4 text-sm font-bold flex items-center justify-center gap-2 transition-colors ${detailTab === "map" ? "bg-stone-900 text-white" : "text-stone-500 hover:bg-stone-100"}`}
                      >
                        <MapIcon size={16} /> Map
                      </button>
                      <button
                        onClick={() => setDetailTab("where")}
                        className={`flex-1 py-3 px-4 text-sm font-bold flex items-center justify-center gap-2 transition-colors ${detailTab === "where" ? "bg-stone-900 text-white" : "text-stone-500 hover:bg-stone-100"}`}
                      >
                        <Globe size={16} /> Where to See
                      </button>
                      <button
                        onClick={() => setDetailTab("taxonomy")}
                        className={`flex-1 py-3 px-4 text-sm font-bold flex items-center justify-center gap-2 transition-colors ${detailTab === "taxonomy" ? "bg-stone-900 text-white" : "text-stone-500 hover:bg-stone-100"}`}
                      >
                        <Dna size={16} /> Taxonomy
                      </button>
                    </div>

                    {/* TAB CONTENT */}
                    <div className="flex-1 min-h-[550px]">
                      {/* MAP TAB */}
                      {detailTab === "map" && (
                        <div className="h-full bg-stone-900">
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
                      )}

                      {/* WHERE TO SEE TAB */}
                      {detailTab === "where" && (() => {
                        // Get the iNat taxon_id from species config
                        const speciesTaxonId = getStaticInfo(selectedEntry.name)?.taxonomy?.taxon_id;

                        return (
                          <div className="p-6 overflow-y-auto max-h-[600px]">
                            <h4 className="font-bold text-stone-700 text-sm uppercase mb-4 flex items-center gap-2">
                              <Globe size={16} className="text-emerald-600" /> Nearby Sighting Hotspots
                            </h4>

                            {/* RADIUS SELECTOR */}
                            <div className="flex flex-wrap gap-2 mb-4">
                              <span className="text-sm text-stone-500 mr-2 w-full">Search radius:</span>
                              {[
                                { value: 10, label: "10mi" },
                                { value: 25, label: "25mi" },
                                { value: 50, label: "50mi" },
                                { value: 100, label: "100mi" },
                                { value: 200, label: "200mi" },
                                { value: 500, label: "500mi" },
                                { value: 2000, label: "USA" },
                                { value: 10000, label: "Global" },
                              ].map(r => (
                                <button
                                  key={r.value}
                                  onClick={() => {
                                    setSearchRadius(r.value);
                                    if (speciesTaxonId) fetchNearbyHotspots(speciesTaxonId, r.value);
                                  }}
                                  className={`px-3 py-1 rounded-full text-xs font-bold transition-colors ${searchRadius === r.value
                                    ? "bg-emerald-600 text-white"
                                    : "bg-stone-100 text-stone-600 hover:bg-stone-200"
                                    }`}
                                >
                                  {r.label}
                                </button>
                              ))}
                            </div>

                            {/* AUTO-FETCH ON TAB OPEN */}
                            {nearbyHotspots.length === 0 && !hotspotsLoading && speciesTaxonId && (
                              <button
                                onClick={() => fetchNearbyHotspots(speciesTaxonId, searchRadius)}
                                className="w-full py-3 bg-emerald-50 text-emerald-700 rounded-xl font-bold mb-4 hover:bg-emerald-100 transition-colors"
                              >
                                🔍 Search for nearby sightings
                              </button>
                            )}

                            {/* LOADING STATE */}
                            {hotspotsLoading && (
                              <div className="flex items-center justify-center py-8 text-stone-500">
                                <Loader2 className="animate-spin mr-2" size={20} />
                                Searching for {selectedEntry.name} sightings...
                              </div>
                            )}

                            {/* HOTSPOT RESULTS */}
                            {!hotspotsLoading && nearbyHotspots.length > 0 && (
                              <div className="space-y-3">
                                {nearbyHotspots.map((spot, idx) => (
                                  <div key={idx} className="p-4 bg-stone-50 rounded-xl border border-stone-100">
                                    <div className="flex justify-between items-start">
                                      <div className="flex-1">
                                        <p className="font-bold text-stone-800 flex items-center gap-2">
                                          <MapPin size={14} className="text-emerald-600" />
                                          {spot.name}
                                        </p>
                                        <div className="flex gap-3 mt-2">
                                          <span className="text-xs bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded-full font-medium">
                                            {spot.count} sighting{spot.count !== 1 ? "s" : ""}
                                          </span>
                                          <span className="text-xs text-stone-500">
                                            ~{spot.distance} mi away
                                          </span>
                                        </div>
                                      </div>
                                      <a
                                        href={`https://www.google.com/maps/search/?api=1&query=${spot.lat},${spot.lng}`}
                                        target="_blank"
                                        rel="noreferrer"
                                        className="text-emerald-600 hover:text-emerald-700 p-2"
                                      >
                                        <ExternalLink size={16} />
                                      </a>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            )}

                            {/* NO RESULTS */}
                            {!hotspotsLoading && nearbyHotspots.length === 0 && speciesTaxonId && (
                              <p className="text-sm text-stone-500 text-center py-4">
                                No verified sightings found within {searchRadius} miles.
                                Try increasing the search radius.
                              </p>
                            )}

                            {/* NO TAXON ID */}
                            {!speciesTaxonId && (
                              <p className="text-sm text-stone-500 text-center py-4">
                                Unable to search - species data not available.
                              </p>
                            )}
                          </div>
                        );
                      })()}

                      {/* TAXONOMY TAB */}
                      {detailTab === "taxonomy" && (
                        <div className="p-6 overflow-y-auto max-h-[600px]">
                          <h4 className="font-bold text-stone-700 text-sm uppercase mb-4 flex items-center gap-2">
                            <Dna size={16} className="text-purple-600" /> Taxonomic Classification
                          </h4>

                          {(() => {
                            const tax = getStaticInfo(selectedEntry.name)?.taxonomy;
                            if (!tax) return <p className="text-stone-500 text-sm">Taxonomy data not available</p>;

                            const levels = [
                              { label: "Kingdom", value: tax.kingdom, color: "bg-stone-100 text-stone-700" },
                              { label: "Phylum", value: tax.phylum, color: "bg-stone-100 text-stone-700" },
                              { label: "Class", value: tax.class, color: "bg-blue-50 text-blue-700" },
                              { label: "Order", value: tax.order, color: "bg-purple-50 text-purple-700" },
                              { label: "Family", value: tax.family, color: "bg-amber-50 text-amber-700" },
                              { label: "Genus", value: tax.genus, color: "bg-emerald-50 text-emerald-700" },
                              { label: "Species", value: tax.species, color: "bg-emerald-100 text-emerald-800" },
                            ];

                            return (
                              <div className="space-y-2 mb-6">
                                {levels.map((level, idx) => level.value && (
                                  <div key={idx} className="flex items-center gap-3">
                                    <span className="text-xs text-stone-400 w-16">{level.label}</span>
                                    <span className={`text-sm font-medium px-3 py-1 rounded-full ${level.color}`}>
                                      {level.value}
                                    </span>
                                  </div>
                                ))}
                              </div>
                            );
                          })()}

                          {/* RELATED SPECIES */}
                          {getRelatedSpecies(selectedEntry.name).length > 0 && (
                            <>
                              <h4 className="font-bold text-stone-700 text-sm uppercase mb-3 mt-6 flex items-center gap-2">
                                Related Species ({getStaticInfo(selectedEntry.name)?.taxonomy?.family})
                              </h4>
                              <div className="grid grid-cols-4 gap-2">
                                {getRelatedSpecies(selectedEntry.name).slice(0, 8).map(species => {
                                  const found = isUnlocked(species.name);
                                  return (
                                    <div
                                      key={species.name}
                                      onClick={() => {
                                        const entry = anidex.find(e => e.name === species.name);
                                        if (entry) setSelectedEntry(entry);
                                        else {
                                          // Create a locked entry for viewing
                                          setSelectedEntry({
                                            id: `locked-${species.name}`,
                                            name: species.name,
                                            timestamp: new Date(),
                                            isLocked: true,
                                            taxonId: 0
                                          });
                                        }
                                      }}
                                      className={`aspect-square rounded-xl p-2 flex flex-col items-center justify-center cursor-pointer transition-all hover:scale-105 ${found ? "bg-emerald-50 border border-emerald-200" : "bg-stone-100 border border-stone-200 opacity-60"}`}
                                    >
                                      <img src={getIconPath(species.name)} alt={species.name} className={`w-8 h-8 object-contain ${found ? "" : "grayscale"}`} />
                                      <p className="text-[8px] text-center mt-1 font-medium text-stone-600 truncate w-full">{species.name}</p>
                                    </div>
                                  );
                                })}
                              </div>
                            </>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {!selectedEntry.isLocked && !selectedEntry.isStumped && (
              <div className="mt-8 bg-white rounded-2xl p-6 shadow-lg border border-stone-100">
                <h3 className="font-bold text-stone-700 mb-4 flex items-center gap-2 text-sm uppercase tracking-wider">
                  <Calendar size={18} /> Your Field Observations
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {getObservationHistory(selectedEntry.name).map((obs) => {
                    // Check if we have cached location name
                    const locKey = `${obs.lat?.toFixed(3)},${obs.lng?.toFixed(3)}`;
                    const cachedLocation = locationNames[locKey];

                    // Trigger async fetch if not cached
                    if (!cachedLocation && obs.lat && obs.lng) {
                      getLocationName(obs.lat, obs.lng);
                    }

                    return (
                      <div key={obs.id} className="bg-stone-50 p-4 rounded-xl border border-stone-200 flex gap-4 hover:border-emerald-400 transition-colors cursor-pointer">
                        <img src={obs.imageUrl} className="w-20 h-20 rounded-xl object-cover bg-stone-200 shrink-0" alt="Observation" />
                        <div className="flex-1 min-w-0">
                          <div className="flex justify-between items-start">
                            <p className="font-bold text-stone-800">Observed in Field</p>
                            {obs.verified ? (
                              <CheckCircle size={18} className="text-emerald-500 shrink-0" />
                            ) : (
                              <AlertTriangle size={18} className="text-yellow-500 shrink-0" />
                            )}
                          </div>
                          <p className="text-sm text-stone-500 mt-1">{obs.timestamp.toLocaleString()}</p>
                          <p className="text-xs text-stone-600 mt-2 flex items-center gap-1">
                            <MapPin size={12} className="text-emerald-600" />
                            {cachedLocation || `${obs.lat?.toFixed(3)}, ${obs.lng?.toFixed(3)}`}
                          </p>
                        </div>
                      </div>
                    );
                  })}
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