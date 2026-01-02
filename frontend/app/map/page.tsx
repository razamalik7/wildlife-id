'use client';

import { useState, useEffect, useMemo } from 'react';
import dynamic from 'next/dynamic';
import { Loader2, Filter, Layers } from 'lucide-react';
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

export default function MapPage() {
    const { user } = useApp();
    const [observations, setObservations] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [filterClass, setFilterClass] = useState('All');
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
                        .select('*, species_config:species(class)') // Try to join if possible, or just select *
                        // Actually, our observations table has 'class' column now? 
                        // Let's assume select * gets the columns we added: family, class, etc.
                        .select('*')
                        .eq('user_id', user.id)
                        .order('created_at', { ascending: false });

                    if (!error && data) {
                        // Merge strategies? Or just prefer Supabase?
                        // If user logged in, maybe local storage is old or duplicates?
                        // For now, let's just combine them, filtering duplicates by unique ID/Timestamp if needed.
                        // But simple concat is safest to not lose data.
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

            // Deduplicate based on created_at + lat/lng to avoid showing same obs twice
            // (e.g. if we eventually implement sync)
            const uniqueObs = allObs.filter((obs, index, self) =>
                index === self.findIndex((t) => (
                    t.created_at === obs.created_at && t.lat === obs.lat
                ))
            );

            setObservations(uniqueObs);
            setLoading(false);
        }

        fetchObservations();
    }, [user]);

    // Filter Logic
    const filteredObservations = useMemo(() => {
        if (filterClass === 'All') return observations;
        return observations.filter(obs => obs.class === filterClass);
    }, [observations, filterClass]);

    const uniqueSpeciesCount = new Set(filteredObservations.map(o => o.common_name || o.species)).size;

    return (
        <div className="relative w-full h-screen bg-stone-100 flex flex-col pb-16"> {/* pb-16 for BottomNav space */}

            {/* Filter Overlay (Top Right) */}
            <div className="absolute top-4 right-4 z-[1000] flex flex-col items-end gap-2">
                <button
                    onClick={() => setIsFilterOpen(!isFilterOpen)}
                    className="bg-white/90 backdrop-blur-md px-4 py-2 rounded-full shadow-lg border border-white/50 text-stone-700 font-bold flex items-center gap-2 hover:bg-white transition-all"
                >
                    <Filter size={16} />
                    <span>{filterClass === 'All' ? 'Filter Map' : filterClass}</span>
                </button>

                {isFilterOpen && (
                    <div className="bg-white/90 backdrop-blur-md rounded-2xl shadow-xl border border-white/50 p-2 min-w-[160px] animate-in fade-in slide-in-from-top-2">
                        <div className="text-[10px] text-stone-400 font-bold px-3 py-1 uppercase tracking-wider">
                            By Class
                        </div>
                        {CLASSES.map(cls => (
                            <button
                                key={cls}
                                onClick={() => {
                                    setFilterClass(cls);
                                    setIsFilterOpen(false);
                                }}
                                className={`w-full text-left px-3 py-2 rounded-xl text-sm font-medium transition-colors ${filterClass === cls
                                        ? 'bg-emerald-100 text-emerald-700'
                                        : 'hover:bg-stone-100 text-stone-600'
                                    }`}
                            >
                                {cls}
                            </button>
                        ))}
                    </div>
                )}
            </div>

            {/* Map Container */}
            <div className="flex-1 w-full h-full relative z-0">
                <ObservationsMap observations={filteredObservations} />
            </div>

            {/* Stats Overlay (Bottom Left) */}
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
