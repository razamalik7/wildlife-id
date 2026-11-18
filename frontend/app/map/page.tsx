'use client';

import { useState, useEffect, useMemo, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import {
    Loader2,
    Filter,
    Search,
    Layers,
    MapPin,
    BookOpen,
    X,
    ExternalLink,
    Compass,
} from 'lucide-react';
import AppHeader from '@/components/AppHeader';
import { supabase } from '@/lib/supabase';
import { useApp } from '@/lib/AppContext';
import { speciesIconUrl } from '@/lib/species';
import {
    ECOSYSTEM_LIST,
    ParkRecord,
    haversineKm,
    findNearestPark,
} from '@/lib/ecosystems';

const ObservationsMap = dynamic(() => import('@/components/ObservationsMap'), {
    ssr: false,
    loading: () => (
        <div className="w-full h-full flex items-center justify-center bg-[#EAE3D2]">
            <div className="text-center bg-[#FBF8F1] px-6 py-5 rounded-2xl border-2 border-[#162A20] tactile-shadow">
                <Loader2 className="w-7 h-7 animate-spin text-[#C84B31] mx-auto mb-2" />
                <p className="text-[#162A20] text-xs font-mono-tech font-extrabold tracking-widest uppercase">
                    UNFOLDING FIELD ATLAS...
                </p>
            </div>
        </div>
    ),
});

const CLASSES = [
    'All',
    'Mammalia',
    'Aves',
    'Reptilia',
    'Amphibia',
    'Insecta',
    'Arachnida',
];
const STATUSES = ['All', 'Native', 'Introduced', 'Domestic'];

function MapPageContent() {
    const searchParams = useSearchParams();
    const { user, allSpecies, isUnlocked } = useApp();

    const [observations, setObservations] = useState<any[]>([]);
    const [parks, setParks] = useState<ParkRecord[]>([]);
    const [loading, setLoading] = useState(true);

    // Map Overlays & Filters
    const [showEcosystems, setShowEcosystems] = useState(false);
    const [showParks, setShowParks] = useState(true);
    const [filterClass, setFilterClass] = useState('All');
    const [filterStatus, setFilterStatus] = useState('All');
    const [isFilterOpen, setIsFilterOpen] = useState(false);

    // Park Search & Selected Park Spotlight Drawer
    const [parkQuery, setParkQuery] = useState('');
    const [selectedPark, setSelectedPark] = useState<ParkRecord | null>(null);
    const [flyTarget, setFlyTarget] = useState<{
        lat: number;
        lng: number;
        zoom?: number;
    } | null>(null);

    // Load 248 Parks from /parks.json
    useEffect(() => {
        fetch('/parks.json')
            .then((r) => r.json())
            .then((data: ParkRecord[]) => {
                setParks(data);
                const parkParam = searchParams.get('park');
                if (parkParam) {
                    const found = data.find(
                        (p) =>
                            p.name.toLowerCase() === parkParam.toLowerCase() ||
                            p.name
                                .toLowerCase()
                                .includes(parkParam.replace('National Park', 'NP').toLowerCase())
                    );
                    if (found) {
                        setSelectedPark(found);
                        setFlyTarget({ lat: found.lat, lng: found.lng, zoom: 9 });
                    }
                }
            })
            .catch((e) => console.error('Failed to load parks.json:', e));
    }, [searchParams]);

    // Load User Observations (LocalStorage + Supabase)
    useEffect(() => {
        async function fetchObservations() {
            let allObs: any[] = [];

            try {
                const local = localStorage.getItem('local_observations');
                if (local) {
                    allObs = [...allObs, ...JSON.parse(local)];
                }
            } catch (e) {
                console.error('Error loading local observations:', e);
            }

            if (user) {
                try {
                    const { data, error } = await supabase
                        .from('observations')
                        .select('*')
                        .eq('user_id', user.id)
                        .order('created_at', { ascending: false });

                    if (!error && data) {
                        const supabaseObs = data.map((obs) => ({
                            ...obs,
                            lat: obs.lat ?? obs.latitude,
                            lng: obs.lng ?? obs.longitude,
                            common_name: obs.species,
                        }));
                        allObs = [...allObs, ...supabaseObs];
                    }
                } catch (e) {
                    console.error('Error loading Supabase observations:', e);
                }
            }

            const uniqueObs = allObs.filter(
                (obs, index, self) =>
                    index ===
                    self.findIndex(
                        (t) => t.created_at === obs.created_at && t.lat === obs.lat
                    )
            );

            const enrichedObs = uniqueObs.map((obs) => {
                const info = allSpecies.find((s) => s.name === obs.common_name);
                return {
                    ...obs,
                    category: info?.category || 'Unknown',
                    class: obs.class || info?.taxonomy?.class || 'Unknown',
                };
            });

            setObservations(enrichedObs);
            setLoading(false);
        }

        if (allSpecies.length > 0 || !loading) {
            fetchObservations();
        }
    }, [user, allSpecies]);

    const filteredObservations = useMemo(() => {
        return observations.filter((obs) => {
            const matchesClass = filterClass === 'All' || obs.class === filterClass;
            const matchesStatus =
                filterStatus === 'All' || obs.category === filterStatus;
            return matchesClass && matchesStatus;
        });
    }, [observations, filterClass, filterStatus]);

    const uniqueSpeciesCount = new Set(
        filteredObservations.map((o) => o.common_name || o.species)
    ).size;

    const stampedParkIds = useMemo(() => {
        const ids = new Set<number>();
        if (!parks.length) return ids;
        for (const obs of filteredObservations) {
            const m = findNearestPark(obs.lat, obs.lng, parks, 45);
            if (m) ids.add(m.park.id);
        }
        return ids;
    }, [filteredObservations, parks]);

    // Matching parks for search bar
    const searchResults = useMemo(() => {
        const q = parkQuery.trim().toLowerCase();
        if (!q) return [];
        return parks
            .filter(
                (p) =>
                    p.name.toLowerCase().includes(q) ||
                    p.state.toLowerCase() === q ||
                    `${p.name} ${p.state}`.toLowerCase().includes(q)
            )
            .slice(0, 8);
    }, [parkQuery, parks]);

    const handleSelectPark = (park: ParkRecord) => {
        setSelectedPark(park);
        setFlyTarget({ lat: park.lat, lng: park.lng, zoom: 9 });
        setParkQuery('');
    };

    // Species & User Observations associated with the currently selected park
    const selectedParkObs = useMemo(() => {
        if (!selectedPark) return [];
        return filteredObservations.filter(
            (o) =>
                o.lat &&
                o.lng &&
                haversineKm(o.lat, o.lng, selectedPark.lat, selectedPark.lng) <= 50
        );
    }, [selectedPark, filteredObservations]);

    const selectedParkKnownSpecies = useMemo(() => {
        if (!selectedPark) return [];
        const baseName = selectedPark.name.replace(' NP', '').toLowerCase();
        return allSpecies.filter((s) =>
            (s.hotspots || []).some((h) => h.toLowerCase().includes(baseName))
        );
    }, [selectedPark, allSpecies]);

    return (
        <div className="w-full h-screen bg-[#EAE3D2] flex flex-col overflow-hidden">
            <AppHeader />

            <div className="relative flex-1 w-full overflow-hidden">
                {/* ============================================================
                    TOP CONTROL BAR: PARK SEARCH + ECOSYSTEM OVERLAY + FILTERS
                   ============================================================ */}
                <div className="absolute top-3.5 left-14 right-3.5 z-[1000] flex flex-wrap items-start justify-between gap-2.5 pointer-events-none">
                    {/* Left: National & State Park Search Box */}
                    <div className="relative w-full sm:w-80 pointer-events-auto">
                        <div className="flex items-center bg-[#FBF8F1] border-2 border-[#162A20] rounded-full tactile-shadow px-3.5 py-2">
                            <Search size={16} className="text-[#C84B31] shrink-0 mr-2" />
                            <input
                                type="text"
                                value={parkQuery}
                                onChange={(e) => setParkQuery(e.target.value)}
                                placeholder="Search 248 National & State Parks..."
                                className="w-full bg-transparent text-xs font-extrabold text-[#162A20] placeholder:text-[#5A655E] focus:outline-none"
                            />
                            {parkQuery && (
                                <button
                                    onClick={() => setParkQuery('')}
                                    className="text-[#5A655E] hover:text-[#162A20] ml-1 cursor-pointer"
                                >
                                    <X size={14} />
                                </button>
                            )}
                        </div>

                        {/* Park Autocomplete Dropdown */}
                        {searchResults.length > 0 && (
                            <div className="mt-1.5 bg-[#FBF8F1] border-2 border-[#162A20] rounded-2xl tactile-shadow-lg overflow-hidden max-h-72 overflow-y-auto">
                                {searchResults.map((park) => {
                                    const isStamped = stampedParkIds.has(park.id);
                                    return (
                                        <button
                                            key={park.id}
                                            onClick={() => handleSelectPark(park)}
                                            className="w-full text-left px-4 py-2.5 hover:bg-[#F4DE9B] border-b border-[#162A20]/15 last:border-none flex items-center justify-between transition-colors cursor-pointer"
                                        >
                                            <div className="flex items-center gap-2">
                                                <span>🏞️</span>
                                                <span className="font-extrabold text-xs text-[#162A20]">
                                                    {park.name}
                                                </span>
                                                {isStamped && (
                                                    <span className="px-1.5 py-0.5 rounded bg-[#234733] text-[#F4DE9B] font-mono-tech text-[9px] font-bold">
                                                        STAMPED
                                                    </span>
                                                )}
                                            </div>
                                            <span className="font-mono-tech text-[10px] font-bold text-[#5A655E] bg-[#EAE3D2] px-2 py-0.5 rounded-full border border-[#162A20]/30">
                                                {park.state} · {park.country}
                                            </span>
                                        </button>
                                    );
                                })}
                            </div>
                        )}
                    </div>

                    {/* Right: Ecosystem Overlay Toggle + Park Patch Toggle + Class/Status Filters */}
                    <div className="flex flex-wrap items-center gap-2 pointer-events-auto ml-auto">
                        {/* Step D: Optional Ecosystem Overlay Toggle */}
                        <button
                            onClick={() => setShowEcosystems(!showEcosystems)}
                            className={`tactile-btn px-3.5 py-2 rounded-full text-xs font-extrabold uppercase tracking-wider flex items-center gap-1.5 cursor-pointer ${
                                showEcosystems
                                    ? 'bg-[#234733] text-[#F4DE9B]'
                                    : 'bg-[#FBF8F1] text-[#162A20] hover:bg-[#F4DE9B]'
                            }`}
                        >
                            <Layers size={14} />
                            <span>🌲 Ecosystems: {showEcosystems ? 'ON' : 'OFF'}</span>
                        </button>

                        {/* M2 Park Patch Markers Toggle */}
                        <button
                            onClick={() => setShowParks(!showParks)}
                            className={`tactile-btn px-3.5 py-2 rounded-full text-xs font-extrabold uppercase tracking-wider flex items-center gap-1.5 cursor-pointer ${
                                showParks
                                    ? 'bg-[#F4DE9B] text-[#162A20]'
                                    : 'bg-[#FBF8F1] text-[#5A655E]'
                            }`}
                        >
                            <Compass size={14} />
                            <span>🏞️ Parks: {showParks ? 'ON' : 'OFF'}</span>
                        </button>

                        {/* Class & Status Filter Menu */}
                        <div className="relative">
                            <button
                                onClick={() => setIsFilterOpen(!isFilterOpen)}
                                className="tactile-btn bg-[#FBF8F1] hover:bg-[#F4DE9B] px-3.5 py-2 rounded-full text-[#162A20] text-xs font-extrabold uppercase tracking-wider flex items-center gap-1.5 cursor-pointer"
                            >
                                <Filter size={14} />
                                <span>Filters</span>
                                {(filterClass !== 'All' || filterStatus !== 'All') && (
                                    <span className="w-2 h-2 rounded-full bg-[#C84B31]" />
                                )}
                            </button>

                            {isFilterOpen && (
                                <div className="absolute right-0 mt-2 bg-[#FBF8F1] rounded-2xl border-2 border-[#162A20] tactile-shadow-lg p-3.5 min-w-[210px] space-y-3.5 max-h-[70vh] overflow-y-auto z-50">
                                    <div>
                                        <div className="font-mono-tech text-[10px] text-[#5A655E] font-extrabold px-2 mb-1 uppercase tracking-widest">
                                            By Taxonomic Class
                                        </div>
                                        <div className="space-y-1">
                                            {CLASSES.map((cls) => (
                                                <button
                                                    key={cls}
                                                    onClick={() => setFilterClass(cls)}
                                                    className={`w-full text-left px-3 py-1.5 rounded-lg text-xs font-bold transition-colors cursor-pointer ${
                                                        filterClass === cls
                                                            ? 'bg-[#162A20] text-[#F4DE9B]'
                                                            : 'hover:bg-[#F7F3E8] text-[#162A20]'
                                                    }`}
                                                >
                                                    {cls}
                                                </button>
                                            ))}
                                        </div>
                                    </div>

                                    <div className="w-full h-0.5 bg-[#162A20]/15" />

                                    <div>
                                        <div className="font-mono-tech text-[10px] text-[#5A655E] font-extrabold px-2 mb-1 uppercase tracking-widest">
                                            By Native Status
                                        </div>
                                        <div className="space-y-1">
                                            {STATUSES.map((stat) => (
                                                <button
                                                    key={stat}
                                                    onClick={() => setFilterStatus(stat)}
                                                    className={`w-full text-left px-3 py-1.5 rounded-lg text-xs font-bold transition-colors cursor-pointer ${
                                                        filterStatus === stat
                                                            ? 'bg-[#C84B31] text-[#FBF8F1]'
                                                            : 'hover:bg-[#F7F3E8] text-[#162A20]'
                                                    }`}
                                                >
                                                    {stat}
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* ============================================================
                    REAL LEAFLET MAP CANVAS
                   ============================================================ */}
                <div className="w-full h-full relative z-0">
                    <ObservationsMap
                        observations={filteredObservations}
                        parks={parks}
                        showEcosystems={showEcosystems}
                        showParks={showParks}
                        selectedPark={selectedPark}
                        flyTarget={flyTarget}
                        onSelectPark={handleSelectPark}
                    />
                </div>

                {/* ============================================================
                    SELECTED NATIONAL / STATE PARK PATCH SPOTLIGHT DRAWER (M2)
                   ============================================================ */}
                {selectedPark && (
                    <div className="absolute bottom-20 right-4 left-4 sm:left-auto sm:w-96 z-[1000] bg-[#FBF8F1] rounded-2xl border-2 border-[#162A20] tactile-shadow-lg overflow-hidden">
                        <div className="bg-[#162A20] text-[#F4DE9B] px-4 py-3 flex items-center justify-between border-b-2 border-[#162A20]">
                            <div className="flex items-center gap-2">
                                <span className="text-lg">🏞️</span>
                                <div>
                                    <span className="font-mono-tech text-[9px] font-extrabold uppercase tracking-widest text-[#C84B31] block">
                                        NATIONAL / STATE PARK PATCH · {selectedPark.state}
                                    </span>
                                    <h3 className="font-editorial text-2xl text-[#FBF8F1] leading-none">
                                        {selectedPark.name}
                                    </h3>
                                </div>
                            </div>
                            <button
                                onClick={() => setSelectedPark(null)}
                                className="w-7 h-7 rounded-full bg-[#FBF8F1] text-[#162A20] font-bold flex items-center justify-center cursor-pointer"
                            >
                                ✕
                            </button>
                        </div>

                        <div className="p-4 space-y-3.5 max-h-[52vh] overflow-y-auto">
                            {/* Park Sighting Stats */}
                            <div className="grid grid-cols-2 gap-2.5">
                                <div className="bg-[#F7F3E8] p-2.5 rounded-xl border border-[#162A20]">
                                    <span className="font-mono-tech text-[9px] font-bold uppercase text-[#5A655E] block">
                                        Your Park Polaroids
                                    </span>
                                    <span className="font-mono-tech text-lg font-extrabold text-[#C84B31]">
                                        {selectedParkObs.length} Logged
                                    </span>
                                </div>
                                <div className="bg-[#F7F3E8] p-2.5 rounded-xl border border-[#162A20]">
                                    <span className="font-mono-tech text-[9px] font-bold uppercase text-[#5A655E] block">
                                        Featured Park Fauna
                                    </span>
                                    <span className="font-mono-tech text-lg font-extrabold text-[#162A20]">
                                        {selectedParkKnownSpecies.length} Species
                                    </span>
                                </div>
                            </div>

                            {/* Featured Species in This Park */}
                            {selectedParkKnownSpecies.length > 0 && (
                                <div>
                                    <p className="font-mono-tech text-[10px] font-extrabold uppercase tracking-wider text-[#5A655E] mb-1.5">
                                        Signature Species in {selectedPark.name}:
                                    </p>
                                    <div className="flex flex-wrap gap-1.5">
                                        {selectedParkKnownSpecies.slice(0, 6).map((sp) => {
                                            const unl = isUnlocked(sp.name);
                                            return (
                                                <Link
                                                    key={sp.name}
                                                    href={`/anidex/${encodeURIComponent(sp.name)}`}
                                                    className={`px-2.5 py-1 rounded-full border border-[#162A20] text-[11px] font-extrabold flex items-center gap-1.5 ${
                                                        unl
                                                            ? 'bg-[#F4DE9B] text-[#162A20]'
                                                            : 'bg-[#F7F3E8] text-[#5A655E]'
                                                    }`}
                                                >
                                                    <img
                                                        src={speciesIconUrl(sp.name)}
                                                        alt={sp.name}
                                                        className={`w-4 h-4 object-contain ${
                                                            unl ? '' : 'woodcut-ghost-art'
                                                        }`}
                                                    />
                                                    <span>{sp.name}</span>
                                                </Link>
                                            );
                                        })}
                                    </div>
                                </div>
                            )}

                            {/* Cross-Navigation Actions */}
                            <div className="flex items-center gap-2 pt-1">
                                <Link
                                    href={`/anidex?park=${encodeURIComponent(
                                        selectedPark.name
                                    )}`}
                                    className="flex-1 tactile-btn py-2.5 px-4 rounded-full bg-[#C84B31] text-[#FBF8F1] text-xs font-extrabold uppercase tracking-wider flex items-center justify-center gap-1.5"
                                >
                                    <BookOpen size={14} />
                                    <span>Filter AniDex to Park →</span>
                                </Link>
                                <a
                                    href={`https://www.google.com/maps/search/?api=1&query=${selectedPark.lat},${selectedPark.lng}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="tactile-btn py-2.5 px-3.5 rounded-full bg-[#F7F3E8] text-[#162A20] text-xs font-extrabold uppercase tracking-wider flex items-center gap-1"
                                >
                                    <ExternalLink size={13} />
                                </a>
                            </div>
                        </div>
                    </div>
                )}

                {/* ============================================================
                    ECOSYSTEM OVERLAY LEGEND (WHEN TOGGLED ON)
                   ============================================================ */}
                {showEcosystems && !selectedPark && (
                    <div className="absolute bottom-20 right-4 z-[1000] bg-[#FBF8F1]/95 backdrop-blur-xs p-3.5 rounded-2xl border-2 border-[#162A20] tactile-shadow max-w-xs">
                        <p className="font-mono-tech text-[10px] font-extrabold uppercase tracking-widest text-[#162A20] mb-2">
                            🌲 North American Biomes (Click to Filter Deck)
                        </p>
                        <div className="grid grid-cols-2 gap-1.5">
                            {ECOSYSTEM_LIST.map((eco) => (
                                <Link
                                    key={eco.id}
                                    href={`/anidex?ecosystem=${eco.id}`}
                                    className="px-2 py-1 rounded-lg border border-[#162A20] bg-[#F7F3E8] hover:bg-[#F4DE9B] text-[10px] font-extrabold text-[#162A20] flex items-center gap-1.5 truncate"
                                >
                                    <span>{eco.icon}</span>
                                    <span className="truncate">{eco.shortName}</span>
                                </Link>
                            ))}
                        </div>
                    </div>
                )}

                {/* ============================================================
                    BOTTOM-LEFT ATLAS STATS STAMP
                   ============================================================ */}
                <div className="absolute bottom-20 left-4 z-[1000] pointer-events-auto">
                    <div className="bg-[#FBF8F1] px-4 py-2.5 rounded-2xl border-2 border-[#162A20] tactile-shadow flex items-center gap-4">
                        <div className="text-center">
                            <p className="font-mono-tech text-[9px] text-[#5A655E] uppercase font-extrabold tracking-widest">
                                Polaroids
                            </p>
                            <p className="font-mono-tech text-lg font-extrabold text-[#162A20]">
                                {filteredObservations.length}
                            </p>
                        </div>
                        <div className="w-0.5 h-7 bg-[#162A20]/20"></div>
                        <div className="text-center">
                            <p className="font-mono-tech text-[9px] text-[#5A655E] uppercase font-extrabold tracking-widest">
                                Species
                            </p>
                            <p className="font-mono-tech text-lg font-extrabold text-[#C84B31]">
                                {uniqueSpeciesCount}
                            </p>
                        </div>
                        <div className="w-0.5 h-7 bg-[#162A20]/20"></div>
                        <div className="text-center">
                            <p className="font-mono-tech text-[9px] text-[#5A655E] uppercase font-extrabold tracking-widest">
                                Parks Stamped
                            </p>
                            <p className="font-mono-tech text-lg font-extrabold text-[#234733]">
                                {stampedParkIds.size}
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default function MapPage() {
    return (
        <Suspense
            fallback={
                <div className="w-full h-screen bg-[#EAE3D2] flex items-center justify-center font-mono-tech text-xs font-bold uppercase">
                    Loading Field Atlas...
                </div>
            }
        >
            <MapPageContent />
        </Suspense>
    );
}
