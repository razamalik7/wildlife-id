'use client';

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import axios from 'axios';
import { supabase } from '@/lib/supabase';
import { User } from '@supabase/supabase-js';

// --- TYPES ---
export interface AnimalEntry {
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

export interface SpeciesInfo {
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

interface AppContextType {
    user: User | null;
    allSpecies: SpeciesInfo[];
    anidex: AnimalEntry[];
    setAnidex: (entries: AnimalEntry[] | ((prev: AnimalEntry[]) => AnimalEntry[])) => void;
    isUnlocked: (name: string) => boolean;
    getStaticInfo: (name: string) => SpeciesInfo | undefined;
    location: { lat: number; lng: number } | null;
    setLocation: (loc: { lat: number; lng: number } | null) => void;
}

const AppContext = createContext<AppContextType | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
    const [user, setUser] = useState<User | null>(null);
    const [allSpecies, setAllSpecies] = useState<SpeciesInfo[]>([]);
    const [anidex, setAnidex] = useState<AnimalEntry[]>([]);
    const [location, setLocation] = useState<{ lat: number; lng: number } | null>(null);

    // Load species data
    useEffect(() => {
        const fetchSpecies = async () => {
            try {
                const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
                const res = await axios.get(`${API_BASE}/species`);
                setAllSpecies(res.data);
            } catch (e) {
                console.error('Failed to load species:', e);
            }
        };
        fetchSpecies();
    }, []);

    // Track auth state
    useEffect(() => {
        supabase.auth.getSession().then(({ data: { session } }) => {
            setUser(session?.user ?? null);
            if (session?.user) {
                loadUserObservations(session.user.id);
            }
        });
        const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
            setUser(session?.user ?? null);
            if (session?.user) {
                loadUserObservations(session.user.id);
            } else {
                setAnidex([]);
            }
        });
        return () => subscription.unsubscribe();
    }, []);

    // Load user observations from Supabase
    const loadUserObservations = async (userId: string) => {
        const { data, error } = await supabase
            .from('observations')
            .select('*')
            .eq('user_id', userId)
            .order('created_at', { ascending: false });

        if (!error && data) {
            const speciesTaxonIds: Record<string, number> = {};

            for (const obs of data) {
                if (!speciesTaxonIds[obs.species]) {
                    try {
                        const taxRes = await axios.get(`https://api.inaturalist.org/v1/taxa?q=${obs.species}&per_page=1`);
                        if (taxRes.data.results.length > 0) {
                            speciesTaxonIds[obs.species] = taxRes.data.results[0].id;
                        }
                    } catch (e) {
                        speciesTaxonIds[obs.species] = 0;
                    }
                }
            }

            const entries: AnimalEntry[] = data.map((obs: any) => ({
                id: obs.id,
                name: obs.species,
                timestamp: new Date(obs.observed_at),
                imageUrl: obs.image_url,
                verified: true,
                taxonId: speciesTaxonIds[obs.species] || 0,
                lat: obs.latitude,
                lng: obs.longitude,
                isLocked: false,
            }));
            setAnidex(entries);
        }
    };

    // Get location on mount
    useEffect(() => {
        navigator.geolocation.getCurrentPosition(
            (pos) => setLocation({ lat: pos.coords.latitude, lng: pos.coords.longitude }),
            () => console.log('GPS unavailable'),
            { timeout: 10000 }
        );
    }, []);

    const isUnlocked = (name: string) => anidex.some(e => e.name === name && !e.isLocked);
    const getStaticInfo = (name: string) => allSpecies.find(s => s.name === name);

    return (
        <AppContext.Provider value={{
            user,
            allSpecies,
            anidex,
            setAnidex,
            isUnlocked,
            getStaticInfo,
            location,
            setLocation,
        }}>
            {children}
        </AppContext.Provider>
    );
}

export function useApp() {
    const context = useContext(AppContext);
    if (!context) throw new Error('useApp must be used within AppProvider');
    return context;
}
