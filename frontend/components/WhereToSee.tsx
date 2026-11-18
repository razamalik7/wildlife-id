'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';
import Link from 'next/link';
import { Loader2, Navigation, Trees, Calendar, Info, MapPin, Sparkles } from 'lucide-react';

interface WhereToSeeProps {
    taxonId?: number;
    userLat?: number;
    userLng?: number;
    staticHotspots?: string[];
}

interface Park {
    id: number;
    name: string;
    state: string;
    country: string;
    lat: number;
    lng: number;
}

interface ParkWithCount extends Park {
    observationCount: number;
    distance: number;
    recentObsDate?: string;
}

const RADIUS_OPTIONS = [
    { label: '10 mi', value: 16, km: 16 },
    { label: '25 mi', value: 40, km: 40 },
    { label: '50 mi', value: 80, km: 80 },
    { label: '100 mi', value: 160, km: 160 },
    { label: '200 mi', value: 320, km: 320 },
    { label: '500 mi', value: 800, km: 800 },
    { label: 'All US', value: 5000, km: 5000 },
    { label: 'North America', value: 0, km: 99999 },
];

export default function WhereToSee({
    taxonId,
    userLat,
    userLng,
    staticHotspots = [],
}: WhereToSeeProps) {
    const hasLocation = !!(userLat && userLng);
    const [radius, setRadius] = useState(hasLocation ? 320 : 5000);
    const [parks, setParks] = useState<Park[]>([]);
    const [parkResults, setParkResults] = useState<ParkWithCount[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [totalObsFetched, setTotalObsFetched] = useState(0);

    useEffect(() => {
        const loadParks = async () => {
            try {
                const res = await axios.get('/parks.json');
                setParks(res.data);
            } catch (e) {
                console.error('Failed to load parks:', e);
            }
        };
        loadParks();
    }, []);

    const getDistanceKm = (lat1: number, lng1: number, lat2: number, lng2: number) => {
        const R = 6371;
        const dLat = ((lat2 - lat1) * Math.PI) / 180;
        const dLng = ((lng2 - lng1) * Math.PI) / 180;
        const a =
            Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos((lat1 * Math.PI) / 180) *
                Math.cos((lat2 * Math.PI) / 180) *
                Math.sin(dLng / 2) *
                Math.sin(dLng / 2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return R * c;
    };

    const kmToMiles = (km: number) => km * 0.621371;

    useEffect(() => {
        const fetchAndProcess = async () => {
            if (!taxonId) {
                setLoading(false);
                return;
            }

            setLoading(true);
            setError(null);
            setParkResults([]);

            try {
                const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
                const res = await axios.get(`${API_BASE}/parks/recent-sightings?taxon_id=${taxonId}`);

                const results = (res.data.results || []) as ParkWithCount[];
                setTotalObsFetched(res.data.total_observations_scanned || 0);

                if (userLat && userLng) {
                    results.forEach((p) => {
                        p.distance = getDistanceKm(userLat, userLng, p.lat, p.lng);
                    });
                } else {
                    results.forEach((p) => {
                        p.distance = 0;
                    });
                }

                const currentOption = RADIUS_OPTIONS.find((r) => r.value === radius);
                const isGlobal = currentOption?.label === 'North America';
                const isUS = currentOption?.label === 'All US';

                let filtered = results;

                if (!isGlobal && !isUS) {
                    if (!userLat || !userLng) {
                        setError('Enable GPS location to filter by local mileage, or select All US.');
                        setLoading(false);
                        return;
                    }
                    filtered = filtered.filter((p) => p.distance <= (currentOption?.km || 320));
                } else if (isUS) {
                    filtered = filtered.filter((p) => p.country === 'US');
                }

                setParkResults(filtered.slice(0, 10));
            } catch (e) {
                console.error('Failed to load park data:', e);
                setError('Live park feed offline — showing curated National Park hotspots below.');
            }
            setLoading(false);
        };

        fetchAndProcess();
    }, [taxonId, radius, userLat, userLng]);

    return (
        <div className="p-5 sm:p-6 bg-[#FBF8F1] text-[#162A20]">
            {/* Curated Signature Parks from species_config.json */}
            {staticHotspots.length > 0 && (
                <div className="mb-6 p-4 rounded-xl bg-[#F7F3E8] border-2 border-[#162A20] tactile-shadow-sm">
                    <div className="flex items-center gap-2 mb-2.5">
                        <Sparkles size={15} className="text-[#C84B31]" />
                        <span className="font-mono-tech text-[11px] font-extrabold uppercase tracking-wider text-[#162A20]">
                            Prime Field Guide Hotspots
                        </span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                        {staticHotspots.map((spot, idx) => {
                            const matchedPark = parks.find((p) =>
                                spot.toLowerCase().includes(p.name.replace(' NP', '').toLowerCase())
                            );
                            return (
                                <Link
                                    key={idx}
                                    href={
                                        matchedPark
                                            ? `/map?park=${encodeURIComponent(matchedPark.name)}`
                                            : `/map?search=${encodeURIComponent(spot)}`
                                    }
                                    className="tactile-btn px-3 py-1.5 rounded-full bg-[#F4DE9B] hover:bg-[#E8B84B] text-[#162A20] text-xs font-extrabold flex items-center gap-1.5"
                                >
                                    <span>🏞️ {spot}</span>
                                    <span className="text-[10px] opacity-75">→ Map</span>
                                </Link>
                            );
                        })}
                    </div>
                </div>
            )}

            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
                <h3 className="font-editorial text-2xl text-[#162A20] flex items-center gap-2">
                    <Trees size={20} className="text-[#234733]" />
                    <span>Live Park Sightings Radar</span>
                </h3>
                <p className="text-[11px] font-mono-tech text-[#5A655E] flex items-center gap-1">
                    <Info size={13} className="shrink-0 text-[#C84B31]" />
                    <span>
                        Scanned {totalObsFetched > 0 ? totalObsFetched : 'recent'} research-grade
                        sightings across 248 parks
                    </span>
                </p>
            </div>

            {/* Radius Filter Pills */}
            <div className="flex flex-wrap gap-1.5 mb-5">
                {RADIUS_OPTIONS.map((opt) => {
                    const isLocalOption =
                        opt.label !== 'All US' && opt.label !== 'North America';
                    const isDisabled = isLocalOption && !hasLocation;
                    const isActive = radius === opt.value;
                    return (
                        <button
                            key={opt.value}
                            onClick={() => !isDisabled && setRadius(opt.value)}
                            disabled={isDisabled}
                            title={isDisabled ? 'Enable location for local radius search' : ''}
                            className={`px-3 py-1.5 rounded-full text-xs font-extrabold uppercase tracking-wider border-2 border-[#162A20] transition-all cursor-pointer ${
                                isActive
                                    ? 'bg-[#162A20] text-[#F4DE9B] tactile-shadow-sm'
                                    : isDisabled
                                    ? 'bg-[#EAE3D2]/50 text-[#5A655E]/40 border-[#162A20]/20 cursor-not-allowed'
                                    : 'bg-[#F7F3E8] text-[#162A20] hover:bg-[#F4DE9B]'
                            }`}
                        >
                            {opt.label}
                        </button>
                    );
                })}
            </div>

            {/* Results */}
            {loading ? (
                <div className="flex items-center justify-center py-10 bg-[#F7F3E8] rounded-xl border-2 border-dashed border-[#162A20]/30">
                    <Loader2 className="animate-spin text-[#C84B31]" size={22} />
                    <span className="ml-2.5 text-xs font-mono-tech font-bold uppercase tracking-wider text-[#5A655E]">
                        Scanning National & State Park Hotspots...
                    </span>
                </div>
            ) : error ? (
                <div className="p-4 rounded-xl bg-[#F4DE9B]/40 border-2 border-[#162A20] text-xs font-bold text-[#162A20] text-center">
                    {error}
                </div>
            ) : parkResults.length === 0 ? (
                <div className="text-center py-8 bg-[#F7F3E8] rounded-xl border-2 border-dashed border-[#162A20]/30">
                    <Trees size={32} className="mx-auto mb-2 text-[#5A655E]" />
                    <p className="font-bold text-sm text-[#162A20]">
                        No recent sightings within this radius
                    </p>
                    <p className="text-xs text-[#5A655E] mt-1">
                        Switch to <span className="font-bold">All US</span> or{' '}
                        <span className="font-bold">North America</span> to see active parks
                    </p>
                </div>
            ) : (
                <div className="space-y-2.5">
                    {parkResults.map((park, i) => (
                        <div
                            key={`${park.id}-${i}`}
                            className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-[#F7F3E8] p-3.5 rounded-xl border-2 border-[#162A20] tactile-shadow-sm"
                        >
                            <div className="flex items-center gap-3">
                                <div className="w-9 h-9 bg-[#162A20] text-[#F4DE9B] rounded-lg border border-[#162A20] flex items-center justify-center font-mono-tech font-extrabold text-xs shrink-0">
                                    #{i + 1}
                                </div>
                                <div>
                                    <div className="flex items-center gap-2">
                                        <p className="font-extrabold text-sm text-[#162A20]">
                                            {park.name}
                                        </p>
                                        <span className="px-2 py-0.5 rounded-full bg-[#EAE3D2] border border-[#162A20] font-mono-tech text-[10px] font-bold">
                                            {park.state}
                                        </span>
                                    </div>
                                    <div className="flex flex-wrap items-center gap-3 text-xs text-[#5A655E] mt-0.5 font-mono-tech">
                                        <span className="font-bold text-[#234733] flex items-center gap-1">
                                            <Calendar size={11} />
                                            {park.observationCount} recent sightings
                                        </span>
                                        {park.recentObsDate && (
                                            <span>
                                                Last:{' '}
                                                {new Date(park.recentObsDate).toLocaleDateString(
                                                    undefined,
                                                    { month: 'short', day: 'numeric' }
                                                )}
                                            </span>
                                        )}
                                        {park.distance > 0 && (
                                            <span className="flex items-center gap-1">
                                                <Navigation size={11} />
                                                {kmToMiles(park.distance).toFixed(0)} mi away
                                            </span>
                                        )}
                                    </div>
                                </div>
                            </div>

                            <div className="flex items-center gap-2 self-end sm:self-center">
                                <Link
                                    href={`/map?park=${encodeURIComponent(park.name)}`}
                                    className="tactile-btn px-3 py-1.5 bg-[#F4DE9B] hover:bg-[#E8B84B] text-[#162A20] text-xs font-extrabold uppercase tracking-wider rounded-full flex items-center gap-1"
                                >
                                    <MapPin size={12} />
                                    <span>View on Atlas</span>
                                </Link>
                                <a
                                    href={`https://www.google.com/maps/search/?api=1&query=${park.lat},${park.lng}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="tactile-btn px-3 py-1.5 bg-[#C84B31] text-[#FBF8F1] text-xs font-extrabold uppercase tracking-wider rounded-full"
                                >
                                    Directions ↗
                                </a>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
