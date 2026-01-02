'use client';

import { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { ArrowLeft, Map as MapIcon, Loader2 } from 'lucide-react';
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

export default function MapPage() {
    const { user } = useApp();
    const [observations, setObservations] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        async function fetchObservations() {
            if (!user) {
                setLoading(false);
                return;
            }

            try {
                const { data, error } = await supabase
                    .from('observations')
                    .select('*')
                    .eq('user_id', user.id)
                    .order('created_at', { ascending: false });

                if (error) throw error;
                setObservations(data || []);
            } catch (e) {
                console.error("Error loading observations:", e);
            } finally {
                setLoading(false);
            }
        }

        fetchObservations();
    }, [user]);

    return (
        <div className="relative w-full h-screen bg-stone-100 flex flex-col">
            {/* Simple Header Overlay */}
            <div className="absolute top-0 left-0 right-0 z-[1000] p-4 flex items-center justify-between pointer-events-none">
                <Link href="/" className="pointer-events-auto flex items-center gap-2 px-4 py-2 bg-white/90 backdrop-blur-md rounded-full shadow-lg border border-white/50 text-stone-700 font-bold hover:bg-white hover:scale-105 transition-all">
                    <ArrowLeft className="w-4 h-4" />
                    <span>Back</span>
                </Link>

                <div className="bg-emerald-500/90 backdrop-blur-md text-white px-4 py-2 rounded-full shadow-lg font-mono text-sm pointer-events-auto">
                    MY WORLD 🌍
                </div>
            </div>

            {/* Map Container */}
            <div className="flex-1 w-full h-full relative z-0">
                <ObservationsMap observations={observations} />
            </div>

            {/* Footer Stats Overlay */}
            <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-[1000] pointer-events-auto">
                <div className="bg-white/90 backdrop-blur-md px-6 py-3 rounded-2xl shadow-xl border border-white/50 flex items-center gap-6">
                    <div className="text-center">
                        <p className="text-[10px] text-stone-400 uppercase font-bold tracking-wider">Discoveries</p>
                        <p className="text-xl font-black text-stone-800">{observations.length}</p>
                    </div>
                    <div className="w-px h-8 bg-stone-200"></div>
                    <div className="text-center">
                        <p className="text-[10px] text-stone-400 uppercase font-bold tracking-wider">Countries</p>
                        <p className="text-xl font-black text-stone-800">1</p> {/* Placeholder for now */}
                    </div>
                </div>
            </div>
        </div>
    );
}
