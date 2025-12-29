'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';
import { Loader2, Navigation, Trees, Calendar, Info } from 'lucide-react';

interface WhereToSeeProps {
    taxonId?: number;
    userLat?: number;
    userLng?: number;
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
    { label: '10', value: 16, km: 16 },
    { label: '25', value: 40, km: 40 },
    { label: '50', value: 80, km: 80 },
    { label: '100', value: 160, km: 160 },
    { label: '200', value: 320, km: 320 },
    { label: '500', value: 800, km: 800 },
    { label: 'US', value: 5000, km: 5000 },
    { label: 'Global', value: 0, km: 99999 },
];

export default function WhereToSee({ taxonId, userLat, userLng }: WhereToSeeProps) {
    const hasLocation = !!(userLat && userLng);
    const [radius, setRadius] = useState(hasLocation ? 160 : 5000);
    const [parks, setParks] = useState<Park[]>([]);
    const [parkResults, setParkResults] = useState<ParkWithCount[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [totalObsFetched, setTotalObsFetched] = useState(0);

    // Load parks data on mount
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

    // Calculate distance in km
    const getDistanceKm = (lat1: number, lng1: number, lat2: number, lng2: number) => {
        const R = 6371; // Earth radius in km
        const dLat = (lat2 - lat1) * Math.PI / 180;
        const dLng = (lng2 - lng1) * Math.PI / 180;
        const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
            Math.sin(dLng / 2) * Math.sin(dLng / 2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return R * c;
    };

    const kmToMiles = (km: number) => km * 0.621371;

    // Fetch and process recent observations via Backend Proxy
    useEffect(() => {
        const fetchAndProcess = async () => {
            if (!taxonId) return;

            setLoading(true);
            setError(null);
            setParkResults([]);

            try {
                // Call our new backend proxy
                // Use env var or fallback for local dev
                const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
                const res = await axios.get(`${API_BASE}/parks/recent-sightings?taxon_id=${taxonId}`);

                const results = res.data.results as ParkWithCount[];
                setTotalObsFetched(res.data.total_observations_scanned || 0);

                // Add distance calculation client-side since backend doesn't know user location
                if (userLat && userLng) {
                    results.forEach(p => {
                        p.distance = getDistanceKm(userLat, userLng, p.lat, p.lng);
                    });
                } else {
                    results.forEach(p => { p.distance = 0; });
                }

                // Filter and sort results
                const currentOption = RADIUS_OPTIONS.find(r => r.value === radius);
                const isGlobal = currentOption?.label === 'Global';
                const isUS = currentOption?.label === 'US';

                let filtered = results;

                if (!isGlobal && !isUS) {
                    if (!userLat || !userLng) {
                        setError('Location required for local search');
                        setLoading(false);
                        return;
                    }
                    filtered = filtered.filter(p => p.distance <= (currentOption?.km || 160));
                } else if (isUS) {
                    filtered = filtered.filter(p => p.country === 'US');
                }

                // Sort by count
                setParkResults(filtered.slice(0, 10));

            } catch (e) {
                console.error('Failed to load park data:', e);
                setError('Unable to load recent sightings. Attempting retry...');
            }
            setLoading(false);
        };

        fetchAndProcess();
    }, [taxonId, radius, userLat, userLng]);


    return (
        <div className="p-6">
            <h3 className="font-bold text-stone-800 mb-2 flex items-center gap-2">
                <Trees size={18} className="text-emerald-600" /> Where to See (Recent)
            </h3>

            <p className="text-xs text-stone-500 mb-4 flex items-start gap-1">
                <Info size={14} className="mt-0.5 shrink-0" />
                Based on {totalObsFetched > 0 ? totalObsFetched : 'checking'} recent sightings. Shows active hotspots.
            </p>

            {/* Radius Tabs */}
            <div className="flex flex-wrap gap-2 mb-4">
                {RADIUS_OPTIONS.map((opt) => {
                    const isLocalOption = opt.label !== 'US' && opt.label !== 'Global';
                    const isDisabled = isLocalOption && !hasLocation;
                    return (
                        <button
                            key={opt.value}
                            onClick={() => !isDisabled && setRadius(opt.value)}
                            disabled={isDisabled}
                            title={isDisabled ? 'Enable location for local search' : ''}
                            className={`py-2 px-3 rounded-lg text-xs font-bold transition-all ${radius === opt.value
                                ? 'bg-emerald-600 text-white'
                                : isDisabled
                                    ? 'bg-stone-50 text-stone-300 cursor-not-allowed'
                                    : 'bg-stone-100 text-stone-500 hover:bg-stone-200'
                                }`}
                        >
                            {opt.label}
                        </button>
                    );
                })}
            </div>

            {/* Results */}
            {loading ? (
                <div className="flex items-center justify-center py-8">
                    <Loader2 className="animate-spin text-emerald-600" size={24} />
                    <span className="ml-2 text-stone-500">Finding active hotspots...</span>
                </div>
            ) : error ? (
                <p className="text-red-500 text-sm text-center py-4">{error}</p>
            ) : parkResults.length === 0 ? (
                <div className="text-center py-8">
                    <Trees size={32} className="mx-auto mb-2 text-stone-300" />
                    <p className="text-stone-500">No recent sightings in nearby parks</p>
                    <p className="text-stone-400 text-sm">Try the Global search to see current activity</p>
                </div>
            ) : (
                <div className="space-y-3">
                    {parkResults.map((park, i) => (
                        <div key={`${park.id}-${i}`} className="flex items-center gap-3 bg-stone-50 p-4 rounded-xl hover:bg-stone-100 transition-colors">
                            <div className="w-10 h-10 bg-emerald-100 rounded-full flex items-center justify-center text-emerald-700 font-bold">
                                {i + 1}
                            </div>
                            <div className="flex-1">
                                <p className="font-semibold text-stone-700">{park.name}</p>
                                <div className="flex items-center gap-3 text-xs text-stone-400">
                                    <span className="font-bold text-emerald-600 flex items-center gap-1">
                                        <Calendar size={10} />
                                        {park.observationCount} recent
                                    </span>
                                    {park.recentObsDate && (
                                        <span className="text-stone-400">
                                            Last: {new Date(park.recentObsDate).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                                        </span>
                                    )}
                                    {park.distance > 0 && (
                                        <span className="flex items-center gap-1 border-l pl-2 border-stone-300">
                                            <Navigation size={10} />
                                            {kmToMiles(park.distance).toFixed(0)} mi
                                        </span>
                                    )}
                                </div>
                            </div>
                            <a
                                href={`https://www.google.com/maps/search/?api=1&query=${park.lat},${park.lng}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="px-3 py-1.5 bg-emerald-600 text-white text-xs font-bold rounded-lg hover:bg-emerald-700 transition-colors"
                            >
                                Directions
                            </a>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
