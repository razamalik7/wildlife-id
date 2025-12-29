'use client';

import { useParams, useRouter } from 'next/navigation';
import { useApp } from '@/lib/AppContext';
import { useState, useEffect } from 'react';
import { ArrowLeft, MapPin, CheckCircle, AlertTriangle, Globe, Dna, ExternalLink } from 'lucide-react';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import axios from 'axios';

const SpeciesMap = dynamic(() => import('@/components/SpeciesMap'), { ssr: false });
import WhereToSee from '@/components/WhereToSee';

export default function SpeciesDetailPage() {
    const params = useParams();
    const router = useRouter();
    const { allSpecies, anidex, getStaticInfo, location, isUnlocked } = useApp();
    const [detailTab, setDetailTab] = useState<'map' | 'where' | 'taxonomy'>('map');
    const [nearbyHotspots, setNearbyHotspots] = useState<Array<{ name: string; count: number; lat: number; lng: number }>>([]);
    const [hotspotsLoading, setHotspotsLoading] = useState(false);

    const speciesName = decodeURIComponent(params.id as string);
    const unlocked = isUnlocked(speciesName);
    const speciesInfo = getStaticInfo(speciesName);
    const entries = anidex.filter(e => e.name === speciesName);
    const latestEntry = entries[0];

    // Fetch nearby hotspots - use the observation location or user location
    // Hotspots logic removed - now handled by WhereToSee component
    useEffect(() => {
        // No-op to keep hooks consistent if needed, or better to remove entirely
    }, []);

    if (!speciesInfo) {
        return (
            <main className="min-h-screen bg-stone-100 flex items-center justify-center">
                <div className="text-center">
                    <p className="text-stone-500 mb-4">Species not found</p>
                    <Link href="/anidex" className="text-emerald-600 font-bold">← Back to AniDex</Link>
                </div>
            </main>
        );
    }

    const getIconPath = (name: string) => {
        const slug = name.toLowerCase().replace(/ /g, '_');
        return `/icons/${slug}.png`;
    };

    const observations = entries.map(e => ({
        lat: e.lat || 0,
        lng: e.lng || 0,
        date: e.timestamp
    })).filter(o => o.lat !== 0);

    const getIUCNColor = (status?: string) => {
        switch (status) {
            case 'Least Concern': return 'bg-emerald-50 text-emerald-700 border-emerald-100';
            case 'Near Threatened': return 'bg-yellow-50 text-yellow-700 border-yellow-100';
            case 'Vulnerable': return 'bg-orange-50 text-orange-700 border-orange-100';
            case 'Endangered': return 'bg-red-50 text-red-700 border-red-100';
            case 'Critically Endangered': return 'bg-red-100 text-red-800 border-red-200';
            default: return 'bg-stone-50 text-stone-500 border-stone-100';
        }
    };

    return (
        <main className="min-h-screen bg-stone-100 pb-24">
            {/* Header */}
            <header className="bg-stone-900 text-white p-4 sticky top-0 z-50">
                <div className="max-w-6xl mx-auto flex items-center gap-4">
                    <button onClick={() => router.back()} className="p-2 hover:bg-stone-800 rounded-lg transition-colors">
                        <ArrowLeft size={24} />
                    </button>
                    <div className="flex-1">
                        <h1 className="text-xl font-bold">{speciesName}</h1>
                        {speciesInfo.scientific_name && (
                            <p className="text-sm text-stone-400 italic">{speciesInfo.scientific_name}</p>
                        )}
                    </div>
                </div>
            </header>

            <div className="max-w-6xl mx-auto p-4">
                {/* Species Card */}
                <div className="bg-white rounded-3xl shadow-xl overflow-hidden mb-6">
                    <div className="flex flex-col md:flex-row">
                        {/* Image */}
                        <div className="md:w-1/3 bg-stone-100 p-8 flex items-center justify-center">
                            <img
                                src={getIconPath(speciesName)}
                                alt={speciesName}
                                className={`w-48 h-48 object-contain ${!unlocked ? 'grayscale opacity-60' : ''}`}
                            />
                        </div>

                        {/* Info */}
                        <div className="flex-1 p-6">
                            <div className="flex flex-wrap gap-2 mb-4">
                                <span className={`text-xs font-bold px-3 py-1 rounded-full border ${speciesInfo.category === 'Invasive'
                                    ? 'bg-red-50 text-red-700 border-red-100'
                                    : 'bg-emerald-50 text-emerald-700 border-emerald-100'
                                    }`}>
                                    {speciesInfo.category?.toUpperCase()}
                                </span>
                                {speciesInfo.iucn && (
                                    <span className={`text-xs font-bold px-3 py-1 rounded-full border ${getIUCNColor(speciesInfo.iucn)}`}>
                                        {speciesInfo.iucn}
                                    </span>
                                )}
                                {entries.length > 0 && (
                                    <span className="text-xs font-bold px-3 py-1 rounded-full bg-emerald-100 text-emerald-700 flex items-center gap-1">
                                        <CheckCircle size={12} /> {entries.length} Observation{entries.length > 1 ? 's' : ''}
                                    </span>
                                )}
                            </div>

                            {speciesInfo.description && (
                                <p className="text-stone-600 text-sm mb-4">{speciesInfo.description}</p>
                            )}

                            {/* Taxonomy Pills */}
                            {speciesInfo.taxonomy && (
                                <div className="flex flex-wrap gap-2 text-xs">
                                    {speciesInfo.taxonomy.class && (
                                        <span className="bg-blue-50 text-blue-700 px-2 py-1 rounded-full">{speciesInfo.taxonomy.class}</span>
                                    )}
                                    {speciesInfo.taxonomy.order && (
                                        <span className="bg-purple-50 text-purple-700 px-2 py-1 rounded-full">{speciesInfo.taxonomy.order}</span>
                                    )}
                                    {speciesInfo.taxonomy.family && (
                                        <span className="bg-amber-50 text-amber-700 px-2 py-1 rounded-full">{speciesInfo.taxonomy.family}</span>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* Tabs */}
                <div className="flex gap-2 mb-4">
                    <button
                        onClick={() => setDetailTab('map')}
                        className={`flex-1 py-3 rounded-xl font-bold text-sm flex items-center justify-center gap-2 transition-all ${detailTab === 'map'
                            ? 'bg-stone-800 text-white'
                            : 'bg-white text-stone-600 border border-stone-200'
                            }`}
                    >
                        <Globe size={16} /> Range Map
                    </button>
                    <button
                        onClick={() => setDetailTab('where')}
                        className={`flex-1 py-3 rounded-xl font-bold text-sm flex items-center justify-center gap-2 transition-all ${detailTab === 'where'
                            ? 'bg-stone-800 text-white'
                            : 'bg-white text-stone-600 border border-stone-200'
                            }`}
                    >
                        <MapPin size={16} /> Where to See
                    </button>
                    <button
                        onClick={() => setDetailTab('taxonomy')}
                        className={`flex-1 py-3 rounded-xl font-bold text-sm flex items-center justify-center gap-2 transition-all ${detailTab === 'taxonomy'
                            ? 'bg-stone-800 text-white'
                            : 'bg-white text-stone-600 border border-stone-200'
                            }`}
                    >
                        <Dna size={16} /> Taxonomy
                    </button>
                </div>

                {/* Tab Content */}
                <div className="bg-white rounded-3xl shadow-xl overflow-hidden min-h-[400px]">
                    {detailTab === 'map' && (
                        <div className="h-[500px]">
                            {speciesInfo.taxonomy?.taxon_id ? (
                                <SpeciesMap
                                    taxonId={speciesInfo.taxonomy.taxon_id}
                                    userLat={location?.lat}
                                    userLng={location?.lng}
                                    animalName={speciesName}
                                    observations={observations}
                                />
                            ) : (
                                <div className="h-full flex items-center justify-center text-stone-400">
                                    Map unavailable - no taxon ID
                                </div>
                            )}
                        </div>
                    )}

                    {detailTab === 'where' && (
                        <WhereToSee
                            taxonId={latestEntry?.taxonId || speciesInfo?.taxonomy?.taxon_id}
                            userLat={location?.lat}
                            userLng={location?.lng}
                        />
                    )}

                    {detailTab === 'taxonomy' && (
                        <div className="p-6">
                            <h3 className="font-bold text-stone-800 mb-4 flex items-center gap-2">
                                <Dna size={18} className="text-emerald-600" /> Scientific Classification
                            </h3>
                            <div className="grid grid-cols-2 gap-3">
                                {speciesInfo.taxonomy?.kingdom && (
                                    <div className="bg-stone-50 p-3 rounded-xl">
                                        <p className="text-xs text-stone-400 uppercase font-bold">Kingdom</p>
                                        <p className="text-stone-700 font-semibold">{speciesInfo.taxonomy.kingdom}</p>
                                    </div>
                                )}
                                {speciesInfo.taxonomy?.phylum && (
                                    <div className="bg-stone-50 p-3 rounded-xl">
                                        <p className="text-xs text-stone-400 uppercase font-bold">Phylum</p>
                                        <p className="text-stone-700 font-semibold">{speciesInfo.taxonomy.phylum}</p>
                                    </div>
                                )}
                                {speciesInfo.taxonomy?.class && (
                                    <div className="bg-stone-50 p-3 rounded-xl">
                                        <p className="text-xs text-stone-400 uppercase font-bold">Class</p>
                                        <p className="text-stone-700 font-semibold">{speciesInfo.taxonomy.class}</p>
                                    </div>
                                )}
                                {speciesInfo.taxonomy?.order && (
                                    <div className="bg-stone-50 p-3 rounded-xl">
                                        <p className="text-xs text-stone-400 uppercase font-bold">Order</p>
                                        <p className="text-stone-700 font-semibold">{speciesInfo.taxonomy.order}</p>
                                    </div>
                                )}
                                {speciesInfo.taxonomy?.family && (
                                    <div className="bg-stone-50 p-3 rounded-xl">
                                        <p className="text-xs text-stone-400 uppercase font-bold">Family</p>
                                        <p className="text-stone-700 font-semibold">{speciesInfo.taxonomy.family}</p>
                                    </div>
                                )}
                                {speciesInfo.taxonomy?.genus && (
                                    <div className="bg-stone-50 p-3 rounded-xl">
                                        <p className="text-xs text-stone-400 uppercase font-bold">Genus</p>
                                        <p className="text-stone-700 font-semibold italic">{speciesInfo.taxonomy.genus}</p>
                                    </div>
                                )}
                            </div>

                            {speciesInfo.taxonomy?.taxon_id && (
                                <a
                                    href={`https://www.inaturalist.org/taxa/${speciesInfo.taxonomy.taxon_id}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="mt-4 inline-flex items-center gap-2 text-emerald-600 hover:text-emerald-700 font-semibold text-sm"
                                >
                                    View on iNaturalist <ExternalLink size={14} />
                                </a>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </main>
    );
}
