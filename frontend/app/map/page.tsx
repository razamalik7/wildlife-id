'use client';

import { useState, useEffect, useMemo } from 'react';
import dynamic from 'next/dynamic';
import { Loader2, Filter } from 'lucide-react';
import { supabase } from '@/lib/supabase';
import { useApp } from '@/lib/AppContext';

// Dynamically import Map to disable SSR
const ObservationsMap = dynamic(() => import('@/components/ObservationsMap'), {
    ssr: false,
    loading: () => (
        <div className="w-full h-full flex items-center justify-center bg-stone-50">
            <div className="text-center">
                <Loader2 className="w-8 h-8 animate-spin text-emerald-500 mx-auto mb-2" />
                <p className="text-stone-400 text-sm font-mono tracking-wider">LOADING MAP DATA...</p>
            </div>
        </div>
    )
});

const CLASSES = ['All', 'Mammalia', 'Aves', 'Reptilia', 'Amphibia', 'Insecta', 'Arachnida'];
const STATUSES = ['All', 'Native', 'Introduced', 'Domestic'];

export default function MapPage() {
    const { user, allSpecies } = useApp();
    const [observations, setObservations] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);

    // Filters
    const [filterClass, setFilterClass] = useState('All');
    const [filterStatus, setFilterStatus] = useState('All');
    const [isFilterOpen, setIsFilterOpen] = useState(false);

    useEffect(() => {
        async function fetchObservations() {
            let allObs: any[] = [];

            // 1. Load Local Storage (Anonymous)
            try {
                const local = localStorage.getItem('local_observations');
                if (local) {
                    allObs = [...allObs, ...JSON.parse(local)];
                }
            } catch (e) {
                console.error("Error loading local observations:", e);
            }

            // 2. Load Supabase (Logged In)
            if (user) {
                try {
                    const { data, error } = await supabase
                        .from('observations')
                        .select('*')
                        .eq('user_id', user.id)
                        .order('created_at', { ascending: false });

                    if (!error && data) {
                        const supabaseObs = data.map(obs => ({
                            ...obs,
                            common_name: obs.species // Map expects common_name
                        }));
                        allObs = [...allObs, ...supabaseObs];
                    }
                } catch (e) {
                    console.error("Error loading Supabase observations:", e);
                }
            }

            // Deduplicate
            const uniqueObs = allObs.filter((obs, index, self) =>
                index === self.findIndex((t) => (
                    t.created_at === obs.created_at && t.lat === obs.lat
                ))
            );

            // Enrich with Category from allSpecies
            const enrichedObs = uniqueObs.map(obs => {
                const info = allSpecies.find(s => s.name === obs.common_name);
                return {
                    ...obs,
                    category: info?.category || 'Unknown'
                };
            });

            setObservations(enrichedObs);
            setLoading(false);
        }

        if (allSpecies.length > 0 || !loading) {
            fetchObservations();
        }
    }, [user, allSpecies]); // Re-run when species data loads

    // Filter Logic
    const filteredObservations = useMemo(() => {
        return observations.filter(obs => {
            const matchesClass = filterClass === 'All' || obs.class === filterClass;
            const matchesStatus = filterStatus === 'All' || obs.category === filterStatus;

            // Treat 'Introduced' as 'Invasive' if user thinks that way? 
            // Or strict match. Let's do strict match for now based on our data.
            return matchesClass && matchesStatus;
        });
    }, [observations, filterClass, filterStatus]);

    const uniqueSpeciesCount = new Set(filteredObservations.map(o => o.common_name || o.species)).size;

    return (
        <div className="relative w-full h-screen bg-stone-100 flex flex-col pb-16">

            {/* Filter Overlay (Top Right) */}
            <div className="absolute top-4 right-4 z-[1000] flex flex-col items-end gap-2">
                <button
                    onClick={() => setIsFilterOpen(!isFilterOpen)}
                    className="bg-white/90 backdrop-blur-md px-4 py-2 rounded-full shadow-lg border border-white/50 text-stone-700 font-bold flex items-center gap-2 hover:bg-white transition-all"
                >
                    <Filter size={16} />
                    <span>Filters</span>
                    {(filterClass !== 'All' || filterStatus !== 'All') && (
                        <div className="w-2 h-2 rounded-full bg-emerald-500" />
                    )}
                </button>

                {isFilterOpen && (
                    <div className="bg-white/90 backdrop-blur-md rounded-2xl shadow-xl border border-white/50 p-3 min-w-[200px] animate-in fade-in slide-in-from-top-2 space-y-4 max-h-[80vh] overflow-y-auto">

                        {/* Class Filter */}
                        <div>
                            <div className="text-[10px] text-stone-400 font-bold px-2 mb-1 uppercase tracking-wider">
                                By Class
                            </div>
                            <div className="space-y-1">
                                {CLASSES.map(cls => (
                                    <button
                                        key={cls}
                                        onClick={() => setFilterClass(cls)}
                                        className={`w-full text-left px-3 py-1.5 rounded-lg text-sm transition-colors ${filterClass === cls
                                                ? 'bg-emerald-100 text-emerald-700 font-bold'
                                                : 'hover:bg-stone-100 text-stone-600'
                                            }`}
                                    >
                                        {cls}
                                    </button>
                                ))}
                            </div>
                        </div>

                        <div className="w-full h-px bg-stone-200" />

                        {/* Status Filter */}
                        <div>
                            <div className="text-[10px] text-stone-400 font-bold px-2 mb-1 uppercase tracking-wider">
                                By Status
                            </div>
                            <div className="space-y-1">
                                {STATUSES.map(stat => (
                                    <button
                                        key={stat}
                                        onClick={() => setFilterStatus(stat)}
                                        className={`w-full text-left px-3 py-1.5 rounded-lg text-sm transition-colors ${filterStatus === stat
                                                ? 'bg-amber-100 text-amber-700 font-bold'
                                                : 'hover:bg-stone-100 text-stone-600'
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

            {/* Map */}
            <div className="flex-1 w-full h-full relative z-0">
                <ObservationsMap observations={filteredObservations} />
            </div>

            {/* Stats */}
            <div className="absolute bottom-20 left-4 z-[1000] pointer-events-auto">
                <div className="bg-white/90 backdrop-blur-md px-5 py-3 rounded-2xl shadow-xl border border-white/50 flex items-center gap-5">
                    <div className="text-center">
                        <p className="text-[10px] text-stone-400 uppercase font-bold tracking-wider">Observations</p>
                        <p className="text-xl font-black text-stone-800">{filteredObservations.length}</p>
                    </div>
                    <div className="w-px h-8 bg-stone-200"></div>
                    <div className="text-center">
                        <p className="text-[10px] text-stone-400 uppercase font-bold tracking-wider">Species</p>
                        <p className="text-xl font-black text-emerald-600">{uniqueSpeciesCount}</p>
                    </div>
                </div>
            </div>
        </div>
    );
}
