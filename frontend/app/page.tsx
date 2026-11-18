'use client';

import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import {
  Camera,
  Search,
  Check,
  Lock,
  Loader2,
  Sparkles,
  MapPin,
  RotateCcw,
  BookOpen,
} from 'lucide-react';
import WelcomeOverlay from '@/components/WelcomeOverlay';
import dynamic from 'next/dynamic';
import 'leaflet/dist/leaflet.css';
import AppHeader from '@/components/AppHeader';
import AuthModal from '@/components/AuthModal';
import SpotInkBackdrop from '@/components/SpotInkBackdrop';
import { speciesIconUrl } from '@/lib/species';
import { useApp } from '@/lib/AppContext';
import {
  getSpeciesEcosystem,
  formatLocationStamp,
  ParkRecord,
} from '@/lib/ecosystems';
import { supabase } from '@/lib/supabase';
import { useRouter } from 'next/navigation';

const ObservationLocationPicker = dynamic(
  () => import('@/components/ObservationLocationPicker'),
  { ssr: false }
);

// --- SIDEBAR RANGE BADGE (Live iNaturalist Check) ---
const SidebarRangeBadge = ({
  taxonId,
  lat,
  lng,
}: {
  taxonId: number;
  lat: number;
  lng: number;
}) => {
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
        const API_BASE =
          process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
        const res = await axios.get(`${API_BASE}/inat/observations`, {
          params: {
            taxon_id: taxonId,
            lat: lat,
            lng: lng,
            radius: 100,
            captive: 'false',
            per_page: 1,
          },
        });
        if (!cancelled) {
          setStatus(res.data.total_results > 0 ? 'yes' : 'no');
        }
      } catch (e) {
        if (!cancelled) setStatus(null);
      }
    };

    check();
    return () => {
      cancelled = true;
    };
  }, [taxonId, lat, lng]);

  if (status === 'loading') {
    return (
      <span className="text-[9px] px-1.5 py-0.5 rounded-full font-mono-tech font-bold uppercase bg-[#EAE3D2] text-[#5A655E] border border-[#162A20]/30 animate-pulse">
        Checking...
      </span>
    );
  }
  if (status === 'yes') {
    return (
      <span className="text-[9px] px-1.5 py-0.5 rounded-full font-mono-tech font-bold uppercase bg-[#234733] text-[#F4DE9B] border border-[#162A20]">
        ✓ In Range
      </span>
    );
  }
  if (status === 'no') {
    return (
      <span className="text-[9px] px-1.5 py-0.5 rounded-full font-mono-tech font-bold uppercase bg-[#C84B31]/15 text-[#C84B31] border border-[#C84B31]">
        Out of Range
      </span>
    );
  }
  return null;
};

export default function IdentifyPage() {
  const router = useRouter();
  const { user, allSpecies, setAnidex, location } = useApp();

  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [observationLocation, setObservationLocation] = useState<{
    lat: number;
    lng: number;
  } | null>(null);
  const [isLocationConfirmed, setIsLocationConfirmed] = useState(false);
  const [showSignupNudge, setShowSignupNudge] = useState(false);
  const [showAuthModal, setShowAuthModal] = useState(false);

  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [selectedCandidate, setSelectedCandidate] = useState<any>(null);
  const [boundingBox, setBoundingBox] = useState<any>(null);
  const [rangeStatus, setRangeStatus] = useState<'loading' | 'yes' | 'no' | null>(null);
  const [parks, setParks] = useState<ParkRecord[]>([]);
  const [factIndex, setFactIndex] = useState(0);

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setFactIndex(Math.floor(Math.random() * 7));
    fetch('/parks.json')
      .then((r) => r.json())
      .then((data) => setParks(data))
      .catch(() => {});
  }, []);

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

  // Live iNaturalist Range Check for Selected Candidate
  useEffect(() => {
    if (!selectedCandidate || !observationLocation || !selectedCandidate.taxon_id) {
      setRangeStatus(null);
      return;
    }

    setRangeStatus('loading');

    const timer = setTimeout(async () => {
      try {
        const API_BASE =
          process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
        const res = await axios.get(`${API_BASE}/inat/observations`, {
          params: {
            taxon_id: selectedCandidate.taxon_id,
            lat: observationLocation.lat,
            lng: observationLocation.lng,
            radius: 100,
            captive: 'false',
            per_page: 1,
          },
        });

        if (res.data.total_results > 0) {
          setRangeStatus('yes');
        } else {
          setRangeStatus('no');
        }
      } catch (e) {
        console.error('Range check failed', e);
        setRangeStatus(null);
      }
    }, 450);

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

      const API_BASE =
        process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
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
        const taxRes = await axios.get(
          `https://api.inaturalist.org/v1/taxa?q=${finalPrediction}&per_page=1`
        );
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
        candidates: candidates.map((c: any) => c.name),
      };

      setAnidex((prev: any) => [newEntry, ...prev]);

      const speciesInfo = allSpecies.find((s) => s.name === finalPrediction);

      // Save to Supabase if logged in
      if (user) {
        try {
          await supabase.from('observations').insert({
            user_id: user.id,
            image_url: preview || '',
            species: finalPrediction,
            confidence: topScore / 100,
            family:
              candidates[0].taxonomy?.family ||
              speciesInfo?.taxonomy?.family ||
              null,
            class:
              candidates[0].taxonomy?.class ||
              speciesInfo?.taxonomy?.class ||
              null,
            latitude: observationLocation.lat,
            longitude: observationLocation.lng,
            observed_at: new Date().toISOString(),
          });
        } catch (e) {
          console.error('Supabase error:', e);
        }
      } else {
        const localObs = {
          id: Date.now(),
          user_id: 'anon',
          image_url: preview || '',
          common_name: finalPrediction,
          scientific_name: speciesInfo?.scientific_name || candidates[0].name,
          category: speciesInfo?.category || 'Native',
          family:
            candidates[0].taxonomy?.family ||
            speciesInfo?.taxonomy?.family ||
            null,
          class:
            candidates[0].taxonomy?.class ||
            speciesInfo?.taxonomy?.class ||
            null,
          lat: observationLocation.lat,
          lng: observationLocation.lng,
          country: 'Unknown',
          created_at: new Date().toISOString(),
        };

        try {
          const existing = JSON.parse(
            localStorage.getItem('local_observations') || '[]'
          );
          localStorage.setItem(
            'local_observations',
            JSON.stringify([localObs, ...existing])
          );
        } catch (e) {
          console.error('LocalStorage error:', e);
        }
      }

      const enrichedCandidates = candidates.map((c: any) => {
        const match = allSpecies.find((s) => s.name === c.name);
        return {
          ...c,
          scientific_name: c.scientific_name || match?.scientific_name || '',
          taxon_id: c.taxon_id || match?.taxonomy?.taxon_id || bestTaxonId,
          taxonomy: {
            ...c.taxonomy,
            family:
              c.taxonomy?.family && c.taxonomy.family !== 'Unknown'
                ? c.taxonomy.family
                : match?.taxonomy?.family || 'Unknown',
            class:
              c.taxonomy?.class && c.taxonomy.class !== 'Unknown'
                ? c.taxonomy.class
                : match?.taxonomy?.class || 'Unknown',
          },
        };
      });

      setPredictionResult({
        candidates: enrichedCandidates,
        bestTaxonId,
        taxonId: bestTaxonId,
      });
      setSelectedCandidate(enrichedCandidates[0]);

      if (!user) {
        setTimeout(() => setShowSignupNudge(true), 2000);
      }
    } catch (error) {
      alert('Backend Error. Is Python running?');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const selectedSpeciesInfo = selectedCandidate
    ? allSpecies.find((s) => s.name === selectedCandidate.name)
    : undefined;
  const selectedEcosystem = getSpeciesEcosystem(
    selectedSpeciesInfo || selectedCandidate
  );
  const speciesIndex = selectedCandidate
    ? Math.max(
        1,
        allSpecies.findIndex((s) => s.name === selectedCandidate.name) + 1
      )
    : 1;
  const locationStamp = formatLocationStamp(
    observationLocation?.lat,
    observationLocation?.lng,
    parks
  );

  return (
    <main className="min-h-screen bg-[#EAE3D2] text-[#162A20] pb-28">
      <WelcomeOverlay />
      <AppHeader />

      <div className="max-w-6xl mx-auto px-4 pt-5">
        {/* ============================================================
            1. E2 OPEN-AIR MAGAZINE COVER REVEAL (AFTER IDENTIFICATION)
           ============================================================ */}
        {predictionResult && selectedCandidate && preview ? (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* LEFT / CENTER (7 Cols): E2 Open-Air Magazine Cover Hero */}
            <div className="lg:col-span-7 bg-[#FBF8F1] rounded-2xl border-2 border-[#162A20] tactile-shadow-lg overflow-hidden flex flex-col">
              {/* Top Magazine Issue Masthead Strip */}
              <div className="bg-[#162A20] text-[#F4DE9B] px-5 py-2.5 flex flex-wrap items-center justify-between gap-2 border-b-2 border-[#162A20]">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-[#C84B31] animate-pulse" />
                  <span className="font-mono-tech text-[11px] font-extrabold uppercase tracking-widest">
                    FIELD PRESS ISSUE · PLATE #{String(speciesIndex).padStart(3, '0')}
                  </span>
                </div>
                <span className="font-mono-tech text-[10px] font-bold uppercase tracking-widest text-[#FBF8F1]/80">
                  POWERED BY OOGWAY-V1.0
                </span>
              </div>

              {/* Open-Air Cover Stage (Spot-Ink Backdrop + Giant Editorial Title + Breakout Animal Art) */}
              <div className="relative h-72 sm:h-88 w-full overflow-hidden border-b-2 border-[#162A20]">
                <SpotInkBackdrop
                  ecosystemId={selectedEcosystem.id}
                  variant="cover"
                  className="w-full h-full object-cover"
                />

                {/* Giant Editorial Cover Display Typography Behind the Cutout Animal */}
                <div className="absolute inset-x-0 top-3 px-5 flex justify-between items-start pointer-events-none select-none">
                  <h2 className="font-poster text-6xl sm:text-8xl font-black uppercase tracking-tight leading-[0.85] text-[#162A20]/18">
                    {selectedCandidate.name}
                  </h2>
                </div>

                {/* Top Floating Status & Ecosystem Pills */}
                <div className="absolute top-3.5 left-4 right-4 flex flex-wrap items-center justify-between gap-2 z-20">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span
                      className={`px-3 py-1 rounded-full text-[11px] font-mono-tech font-extrabold uppercase border-2 border-[#162A20] tactile-shadow-sm ${
                        selectedSpeciesInfo?.category === 'Introduced'
                          ? 'bg-[#C84B31] text-[#FBF8F1]'
                          : 'bg-[#F4DE9B] text-[#162A20]'
                      }`}
                    >
                      {selectedSpeciesInfo?.category === 'Introduced'
                        ? '⚠ INVASIVE / INTRODUCED'
                        : '✓ NATIVE FAUNA'}
                    </span>
                    <span className="px-3 py-1 rounded-full text-[11px] font-mono-tech font-extrabold uppercase bg-[#FBF8F1] text-[#162A20] border-2 border-[#162A20] tactile-shadow-sm">
                      {selectedEcosystem.icon} {selectedEcosystem.shortName}
                    </span>
                  </div>

                  {/* Live iNaturalist Range Verification Pill */}
                  <div>
                    {rangeStatus === 'loading' && (
                      <span className="px-3 py-1 rounded-full text-[11px] font-mono-tech font-bold uppercase bg-[#FBF8F1] text-[#162A20] border-2 border-[#162A20] animate-pulse">
                        📡 Checking iNat Range...
                      </span>
                    )}
                    {rangeStatus === 'yes' && (
                      <span className="px-3 py-1 rounded-full text-[11px] font-mono-tech font-extrabold uppercase bg-[#234733] text-[#F4DE9B] border-2 border-[#162A20] tactile-shadow-sm">
                        ✅ Verified In Range
                      </span>
                    )}
                    {rangeStatus === 'no' && (
                      <span className="px-3 py-1 rounded-full text-[11px] font-mono-tech font-extrabold uppercase bg-[#C84B31] text-[#FBF8F1] border-2 border-[#162A20] tactile-shadow-sm">
                        ⚠️ Out of Local Range
                      </span>
                    )}
                    {rangeStatus === null &&
                      selectedCandidate.in_range !== undefined && (
                        <span
                          className={`px-3 py-1 rounded-full text-[11px] font-mono-tech font-extrabold uppercase border-2 border-[#162A20] tactile-shadow-sm ${
                            selectedCandidate.in_range
                              ? 'bg-[#234733] text-[#F4DE9B]'
                              : 'bg-[#C84B31] text-[#FBF8F1]'
                          }`}
                        >
                          {selectedCandidate.in_range
                            ? '✓ In Expected Range'
                            : '⚠️ Outside Typical Range'}
                        </span>
                      )}
                  </div>
                </div>

                {/* Center Cutout Species Illustration */}
                <div className="absolute inset-0 flex items-center justify-center pt-6 z-10 pointer-events-none">
                  <img
                    src={speciesIconUrl(selectedCandidate.name)}
                    alt={selectedCandidate.name}
                    className="h-52 sm:h-68 w-auto object-contain cutout-animal-art transition-transform duration-500 hover:scale-105"
                    onError={(e) => {
                      const img = e.target as HTMLImageElement;
                      if (!img.src.endsWith('/icons/default.png')) {
                        img.src = '/icons/default.png';
                      }
                    }}
                  />
                </div>

                {/* Bottom Location Cancellation Stamp */}
                <div className="absolute bottom-3 left-4 right-4 flex items-end justify-between z-20">
                  <div className="bg-[#FBF8F1]/95 backdrop-blur-xs px-3.5 py-1.5 rounded-xl border-2 border-[#162A20] tactile-shadow-sm">
                    <p className="font-mono-tech text-[10px] font-extrabold uppercase tracking-wider text-[#C84B31] flex items-center gap-1">
                      <MapPin size={11} />
                      <span>{locationStamp}</span>
                    </p>
                  </div>

                  {selectedCandidate.score < 40 && (
                    <span className="px-3 py-1 rounded-full bg-[#C84B31] text-[#FBF8F1] border-2 border-[#162A20] font-mono-tech text-[10px] font-extrabold uppercase">
                      ⚠️ Low Confidence
                    </span>
                  )}
                </div>
              </div>

              {/* Cover Dossier Body */}
              <div className="p-5 sm:p-7 space-y-5">
                <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-2 pb-4 border-b-2 border-[#162A20]/15">
                  <div>
                    <p className="font-mono-tech text-xs font-extrabold uppercase tracking-widest text-[#C84B31]">
                      {selectedCandidate.scientific_name ||
                        selectedSpeciesInfo?.scientific_name ||
                        'Verified Specimen'}
                    </p>
                    <h1 className="font-editorial text-4xl sm:text-5xl text-[#162A20] leading-none mt-1">
                      {selectedCandidate.name}
                    </h1>
                  </div>

                  {/* Confidence Stamp */}
                  <div className="bg-[#F7F3E8] px-4 py-2 rounded-xl border-2 border-[#162A20] tactile-shadow-sm text-right self-start sm:self-end">
                    <span className="font-mono-tech text-[10px] font-extrabold uppercase tracking-widest text-[#5A655E] block">
                      OOGWAY CONFIDENCE
                    </span>
                    <span className="font-mono-tech text-2xl font-extrabold text-[#162A20]">
                      {selectedCandidate.score.toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Confidence Progress Bar */}
                <div className="h-3.5 bg-[#EAE3D2] rounded-full overflow-hidden border-2 border-[#162A20]">
                  <div
                    className={`h-full transition-all duration-700 ${
                      selectedCandidate.score > 75
                        ? 'bg-[#234733]'
                        : selectedCandidate.score > 40
                        ? 'bg-[#E8B84B]'
                        : 'bg-[#C84B31]'
                    }`}
                    style={{ width: `${Math.min(100, selectedCandidate.score)}%` }}
                  />
                </div>

                {/* Taxonomy & Description Row */}
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  <div className="p-3.5 bg-[#F7F3E8] rounded-xl border-2 border-[#162A20]">
                    <p className="font-mono-tech text-[10px] uppercase tracking-wider text-[#5A655E] font-bold">
                      Class
                    </p>
                    <p className="font-extrabold text-base text-[#162A20] mt-0.5">
                      {selectedCandidate.taxonomy?.class || 'Unknown'}
                    </p>
                  </div>
                  <div className="p-3.5 bg-[#F7F3E8] rounded-xl border-2 border-[#162A20]">
                    <p className="font-mono-tech text-[10px] uppercase tracking-wider text-[#5A655E] font-bold">
                      Family
                    </p>
                    <p className="font-extrabold text-base text-[#162A20] mt-0.5">
                      {selectedCandidate.taxonomy?.family || 'Unknown'}
                    </p>
                  </div>
                  <div className="col-span-2 sm:col-span-1 p-3.5 bg-[#F7F3E8] rounded-xl border-2 border-[#162A20]">
                    <p className="font-mono-tech text-[10px] uppercase tracking-wider text-[#5A655E] font-bold">
                      Habitat Biome
                    </p>
                    <p className="font-extrabold text-sm text-[#162A20] mt-0.5 truncate">
                      {selectedEcosystem.icon} {selectedEcosystem.name}
                    </p>
                  </div>
                </div>

                {selectedSpeciesInfo?.description && (
                  <p className="text-sm text-[#162A20]/85 bg-[#F7F3E8] p-3.5 rounded-xl border border-[#162A20]/20 italic">
                    “{selectedSpeciesInfo.description}”
                  </p>
                )}

                {/* Action Buttons */}
                <div className="flex flex-col sm:flex-row gap-3 pt-2">
                  <button
                    onClick={resetFlow}
                    className="tactile-btn py-3.5 px-5 bg-[#F7F3E8] hover:bg-[#F4DE9B] text-[#162A20] font-extrabold rounded-full text-xs uppercase tracking-widest flex items-center justify-center gap-2 cursor-pointer"
                  >
                    <RotateCcw size={15} />
                    <span>Scan Another</span>
                  </button>
                  <button
                    onClick={() =>
                      router.push(
                        `/anidex/${encodeURIComponent(selectedCandidate.name)}`
                      )
                    }
                    className="flex-1 tactile-btn py-3.5 px-6 bg-[#C84B31] hover:bg-[#b23f27] text-[#FBF8F1] font-extrabold rounded-full text-xs uppercase tracking-widest flex items-center justify-center gap-2 cursor-pointer"
                  >
                    <BookOpen size={16} />
                    <span>Open Full Collector Card & Polaroid Wall →</span>
                  </button>
                </div>
              </div>
            </div>

            {/* RIGHT COLUMN (5 Cols): Your Uploaded Polaroid + Top-5 Alternatives + OOGWAY Field Note */}
            <div className="lg:col-span-5 space-y-5">
              {/* 1. Your Captured Field Polaroid with YOLOv8 Bounding Box & Park Stamp */}
              <div className="bg-[#FBF8F1] p-3.5 pb-4 rounded-2xl border-2 border-[#162A20] tactile-shadow">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-mono-tech text-[10px] font-extrabold uppercase tracking-widest text-[#5A655E]">
                    📷 YOUR FIELD CAPTURE · YOLOv8 CROP
                  </span>
                  <span className="font-mono-tech text-[10px] font-bold text-[#234733]">
                    ✓ Logged to Polaroid Wall
                  </span>
                </div>
                <div className="relative h-52 w-full bg-[#162A20] rounded-xl overflow-hidden border-2 border-[#162A20]">
                  <img
                    src={preview}
                    alt="Field Observation"
                    className="w-full h-full object-cover"
                  />
                  {boundingBox && (
                    <div
                      className="absolute border-2 border-[#F4DE9B] bg-[#F4DE9B]/15 z-10 rounded-md pointer-events-none"
                      style={{
                        left: `${boundingBox.x * 100}%`,
                        top: `${boundingBox.y * 100}%`,
                        width: `${boundingBox.w * 100}%`,
                        height: `${boundingBox.h * 100}%`,
                      }}
                    >
                      <span className="absolute -top-5 left-0 bg-[#162A20] text-[#F4DE9B] font-mono-tech text-[9px] font-bold px-1.5 py-0.5 rounded">
                        YOLOv8
                      </span>
                    </div>
                  )}
                </div>
                {/* Polaroid Caption Strip with Inked Park Stamp */}
                <div className="mt-3 pt-2 border-t border-dashed border-[#162A20]/30 flex items-center justify-between">
                  <span className="font-mono-tech text-xs font-extrabold uppercase tracking-wider text-[#C84B31] border border-[#C84B31] px-2 py-0.5 rounded bg-[#C84B31]/10 -rotate-1">
                    STAMPED: {locationStamp}
                  </span>
                  <span className="font-mono-tech text-[11px] text-[#5A655E] font-bold">
                    {new Date().toLocaleDateString(undefined, {
                      month: 'short',
                      day: 'numeric',
                      year: 'numeric',
                    })}
                  </span>
                </div>
              </div>

              {/* 2. Top-5 Alternative Matches */}
              <div className="bg-[#FBF8F1] p-4 rounded-2xl border-2 border-[#162A20] tactile-shadow">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-mono-tech text-xs font-extrabold text-[#162A20] uppercase tracking-widest">
                    OOGWAY Ensemble Candidates
                  </h3>
                  <span className="font-mono-tech text-[10px] text-[#5A655E]">
                    Click to inspect
                  </span>
                </div>
                <div className="space-y-2.5">
                  {predictionResult.candidates.map((candidate: any, idx: number) => {
                    const isSelected =
                      candidate.name === selectedCandidate.name;
                    return (
                      <div
                        key={idx}
                        onClick={() => setSelectedCandidate(candidate)}
                        className={`flex items-center justify-between p-2.5 rounded-xl border-2 transition-all cursor-pointer ${
                          isSelected
                            ? 'bg-[#F4DE9B] border-[#162A20] tactile-shadow-sm'
                            : 'bg-[#F7F3E8] border-[#162A20]/30 hover:border-[#162A20]'
                        }`}
                      >
                        <div className="flex items-center gap-3">
                          <div className="w-11 h-11 rounded-lg bg-[#FBF8F1] flex items-center justify-center overflow-hidden border-2 border-[#162A20] shrink-0">
                            <img
                              src={speciesIconUrl(candidate.name)}
                              alt={candidate.name}
                              className="w-full h-full object-contain p-0.5"
                              onError={(e) => {
                                e.currentTarget.style.display = 'none';
                                e.currentTarget.parentElement!.innerText = '🐾';
                              }}
                            />
                          </div>
                          <div>
                            <p className="font-extrabold text-sm text-[#162A20] leading-tight">
                              {candidate.name}
                            </p>
                            <div className="flex items-center gap-2 mt-1">
                              <span className="text-[11px] text-[#162A20] font-mono-tech font-extrabold">
                                {candidate.score.toFixed(1)}%
                              </span>
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
                        <span className="font-mono-tech text-[10px] font-extrabold uppercase px-2 py-1 rounded bg-[#FBF8F1] border border-[#162A20]">
                          #{idx + 1}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* 3. Did You Know? (OOGWAY & Wildlife Field Note) */}
              <div className="bg-[#162A20] text-[#FBF8F1] p-5 rounded-2xl border-2 border-[#162A20] tactile-shadow relative overflow-hidden">
                <p className="font-mono-tech text-[10px] font-extrabold uppercase tracking-widest text-[#E8B84B] mb-1.5 flex items-center gap-1.5">
                  <Sparkles size={13} />
                  <span>OOGWAY FIELD NOTE · DID YOU KNOW?</span>
                </p>
                <p className="text-xs sm:text-sm font-medium leading-relaxed text-[#F7F3E8]">
                  {
                    [
                      <>
                        Our AI analyzes over{' '}
                        <span className="text-[#F4DE9B] font-extrabold">
                          1,000 unique features
                        </span>{' '}
                        to distinguish between similar species like the Gray Wolf
                        and Coyote.
                      </>,
                      <>
                        The model was trained on{' '}
                        <span className="text-[#F4DE9B] font-extrabold">
                          50,000+ images
                        </span>{' '}
                        from verified wildlife photographers across North America.
                      </>,
                      <>
                        Location data helps boost accuracy by{' '}
                        <span className="text-[#F4DE9B] font-extrabold">15%</span>{' '}
                        — a Grizzly in Alaska is more likely than one in Florida!
                      </>,
                      <>
                        Our{' '}
                        <span className="text-[#F4DE9B] font-extrabold">
                          OOGWAY architecture
                        </span>{' '}
                        combines two neural networks for maximum precision.
                      </>,
                      <>
                        Black Bears can be{' '}
                        <span className="text-[#F4DE9B] font-extrabold">
                          brown, cinnamon, or blonde
                        </span>{' '}
                        — fur color doesn&apos;t always match the name!
                      </>,
                      <>
                        The AI checks{' '}
                        <span className="text-[#F4DE9B] font-extrabold">
                          iNaturalist&apos;s 50M+ observations
                        </span>{' '}
                        to verify if a species is in your area.
                      </>,
                      <>
                        We use{' '}
                        <span className="text-[#F4DE9B] font-extrabold">
                          YOLOv8
                        </span>{' '}
                        to automatically find and crop animals before
                        identification.
                      </>,
                    ][factIndex]
                  }
                </p>
              </div>
            </div>
          </div>
        ) : (
          /* ============================================================
              2. UPLOAD & LOCATION CONFIRMATION FLOW (WITH OOGWAY TAG!)
             ============================================================ */
          <div className="max-w-4xl mx-auto">
            {!preview ? (
              <div className="space-y-6">
                {/* Hero Header Banner */}
                <div className="text-center space-y-2">
                  <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-[#F4DE9B] border-2 border-[#162A20] tactile-shadow-sm">
                    <span className="w-2 h-2 rounded-full bg-[#234733] animate-pulse" />
                    <span className="font-mono-tech text-[11px] font-extrabold uppercase tracking-widest text-[#162A20]">
                      POWERED BY OOGWAY-V1.0 · FIELD PRESS SCANNER
                    </span>
                  </div>
                  <h1 className="font-editorial text-4xl sm:text-5xl text-[#162A20] leading-tight">
                    Spot Wildlife. Ink Your Collector Deck.
                  </h1>
                  <p className="text-sm text-[#5A655E] max-w-xl mx-auto font-medium">
                    Upload a field photo to identify the species, verify its live
                    iNaturalist range, and mint its custom Spot-Ink Collector Card.
                  </p>
                </div>

                {/* Archival Field Camera Upload Frame */}
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="bg-[#FBF8F1] rounded-3xl border-2 border-[#162A20] tactile-shadow-lg p-5 sm:p-8 cursor-pointer group hover:bg-[#F7F3E8] transition-all"
                >
                  <div className="aspect-[16/10] sm:aspect-[16/9] w-full rounded-2xl border-2 border-dashed border-[#162A20]/60 group-hover:border-[#C84B31] parchment-dots flex flex-col items-center justify-center p-6 text-center relative overflow-hidden transition-colors">
                    {/* Corner Viewfinder Ticks */}
                    <div className="absolute top-3 left-3 w-4 h-4 border-t-2 border-l-2 border-[#162A20]" />
                    <div className="absolute top-3 right-3 w-4 h-4 border-t-2 border-r-2 border-[#162A20]" />
                    <div className="absolute bottom-3 left-3 w-4 h-4 border-b-2 border-l-2 border-[#162A20]" />
                    <div className="absolute bottom-3 right-3 w-4 h-4 border-b-2 border-r-2 border-[#162A20]" />

                    <div className="w-20 h-20 rounded-2xl bg-[#C84B31] text-[#FBF8F1] border-2 border-[#162A20] tactile-shadow flex items-center justify-center mb-4 group-hover:scale-105 transition-transform">
                      <Camera size={38} />
                    </div>

                    <p className="font-editorial text-3xl text-[#162A20]">
                      Tap to Upload Field Photo
                    </p>
                    <p className="font-mono-tech text-xs font-bold uppercase tracking-wider text-[#5A655E] mt-1">
                      JPG, PNG, OR WEBP • AUTO-CROPPED VIA YOLOv8
                    </p>

                    <div className="mt-5 inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-[#162A20] text-[#F4DE9B] font-mono-tech text-xs font-extrabold uppercase tracking-widest border-2 border-[#162A20] tactile-shadow-sm">
                      <span>Select Photo from Device →</span>
                    </div>
                  </div>
                </div>

                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  className="hidden"
                  accept="image/*"
                />

                {/* OOGWAY Model Tag & Architecture Specification Plaque */}
                <div className="bg-[#FBF8F1] rounded-2xl border-2 border-[#162A20] tactile-shadow p-5 text-center space-y-4">
                  <div>
                    <p className="font-mono-tech text-xs font-extrabold text-[#162A20] tracking-[0.18em] uppercase">
                      Powered by OOGWAY
                    </p>
                    <p className="text-xs text-[#C84B31] font-bold mt-0.5 italic">
                      Custom Architecture Originally Developed for aniML vision
                    </p>
                  </div>

                  <div className="flex flex-wrap justify-center items-center gap-6 sm:gap-10 py-3 px-6 rounded-xl bg-[#F7F3E8] border-2 border-[#162A20] max-w-lg mx-auto">
                    <div className="flex flex-col">
                      <span className="font-mono-tech font-extrabold text-xl text-[#162A20]">
                        90.7%
                      </span>
                      <span className="font-mono-tech text-[10px] font-bold uppercase tracking-wider text-[#5A655E]">
                        Accuracy
                      </span>
                    </div>
                    <div className="w-0.5 h-8 bg-[#162A20]/20" />
                    <div className="flex flex-col">
                      <span className="font-mono-tech font-extrabold text-xl text-[#C84B31]">
                        {allSpecies.length || 100}
                      </span>
                      <span className="font-mono-tech text-[10px] font-bold uppercase tracking-wider text-[#5A655E]">
                        Species Deck
                      </span>
                    </div>
                    <div className="w-0.5 h-8 bg-[#162A20]/20" />
                    <div className="flex flex-col">
                      <span className="font-mono-tech font-extrabold text-xl text-[#234733]">
                        248
                      </span>
                      <span className="font-mono-tech text-[10px] font-bold uppercase tracking-wider text-[#5A655E]">
                        Parks Tracked
                      </span>
                    </div>
                  </div>

                  <div className="flex flex-col sm:flex-row items-center justify-center gap-3 text-xs text-[#5A655E]">
                    <span className="flex items-center gap-1 font-medium">
                      <Lock size={12} className="text-[#234733]" />
                      Your privacy matters — GPS is only used for range
                      verification.
                    </span>
                    {!user && (
                      <button
                        onClick={() => setShowAuthModal(true)}
                        className="font-extrabold text-[#C84B31] underline cursor-pointer"
                      >
                        🎯 Sign in to save your AniDex across devices →
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              /* Step 2: Location Picker Map + Identify CTA */
              <div className="w-full flex flex-col space-y-4">
                <div className="h-[500px]">
                  <ObservationLocationPicker
                    initialLat={location?.lat}
                    initialLng={location?.lng}
                    photoPreview={preview}
                    isConfirmed={isLocationConfirmed}
                    onConfirm={handleLocationConfirm}
                    onLocationChange={handleLocationChange}
                  />
                </div>

                <div className="flex flex-col sm:flex-row gap-3">
                  <button
                    onClick={resetFlow}
                    className="tactile-btn py-4 px-6 bg-[#FBF8F1] hover:bg-[#F4DE9B] text-[#162A20] font-extrabold text-xs uppercase tracking-widest rounded-full cursor-pointer"
                  >
                    ← Change Photo
                  </button>
                  <button
                    onClick={identifyAnimal}
                    disabled={!isLocationConfirmed || loading}
                    className="flex-1 tactile-btn py-4 px-6 bg-[#C84B31] hover:bg-[#b23f27] disabled:bg-[#EAE3D2] disabled:text-[#5A655E] text-[#FBF8F1] font-extrabold text-sm uppercase tracking-widest rounded-full flex items-center justify-center gap-3 cursor-pointer disabled:cursor-not-allowed"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="animate-spin" size={20} />
                        <span>OOGWAY Analyzing Specimen...</span>
                      </>
                    ) : (
                      <>
                        <Search size={20} />
                        <span>
                          {!isLocationConfirmed
                            ? 'Lock Sighting Location Above First'
                            : 'Identify Species with OOGWAY →'}
                        </span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Signup Nudge Modal */}
      {showSignupNudge && (
        <div className="fixed inset-0 bg-[#162A20]/70 backdrop-blur-xs z-50 flex items-center justify-center p-4">
          <div className="bg-[#FBF8F1] rounded-2xl border-2 border-[#162A20] tactile-shadow-lg max-w-md w-full p-7 relative text-center space-y-4">
            <button
              onClick={() => setShowSignupNudge(false)}
              className="absolute top-3.5 right-3.5 w-8 h-8 rounded-full border-2 border-[#162A20] bg-[#F7F3E8] font-bold flex items-center justify-center cursor-pointer"
            >
              ✕
            </button>

            <div className="text-4xl">🎯</div>
            <h2 className="font-editorial text-3xl text-[#162A20]">
              Collector Plate Minted!
            </h2>
            <p className="text-sm text-[#5A655E] font-medium">
              Create a free Ranger Pass to permanently save your Polaroids and
              build your{' '}
              <span className="font-extrabold text-[#162A20]">AniDex</span>!
            </p>

            <div className="pt-2 space-y-2.5">
              <button
                onClick={() => {
                  setShowSignupNudge(false);
                  setShowAuthModal(true);
                }}
                className="w-full py-3 bg-[#C84B31] text-[#FBF8F1] font-extrabold text-xs uppercase tracking-widest rounded-full tactile-btn cursor-pointer"
              >
                Create Free Ranger Pass
              </button>
              <button
                onClick={() => setShowSignupNudge(false)}
                className="w-full py-2 text-[#5A655E] hover:text-[#162A20] text-xs font-bold uppercase tracking-wider cursor-pointer"
              >
                Maybe later
              </button>
            </div>
          </div>
        </div>
      )}

      <AuthModal
        isOpen={showAuthModal}
        onClose={() => setShowAuthModal(false)}
        onSuccess={() => setShowAuthModal(false)}
      />
    </main>
  );
}
