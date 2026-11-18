'use client';

import { useState, useEffect, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { useApp } from '@/lib/AppContext';
import Link from 'next/link';
import AppHeader from '@/components/AppHeader';
import SpotInkBackdrop from '@/components/SpotInkBackdrop';
import { Search, Compass, Sparkles, X } from 'lucide-react';
import { speciesIconUrl } from '@/lib/species';
import {
    ECOSYSTEM_LIST,
    EcosystemId,
    getSpeciesEcosystem,
} from '@/lib/ecosystems';

const CLASSES = [
    { id: 'All', label: 'All Classes' },
    { id: 'Mammalia', label: '🦌 Mammals' },
    { id: 'Aves', label: '🦅 Birds' },
    { id: 'Reptilia', label: '🦎 Reptiles' },
    { id: 'Amphibia', label: '🐸 Amphibians' },
    { id: 'Insecta', label: '🦋 Insects' },
    { id: 'Arachnida', label: '🕷️ Arachnids' },
];

function AnidexContent() {
    const searchParams = useSearchParams();
    const router = useRouter();
    const { allSpecies, anidex, isUnlocked } = useApp();

    const [filter, setFilter] = useState<'Total' | 'Native' | 'Invasive'>('Total');
    const [ecosystemFilter, setEcosystemFilter] = useState<EcosystemId | 'All'>('All');
    const [classFilter, setClassFilter] = useState<string>('All');
    const [parkFilter, setParkFilter] = useState<string>('');
    const [searchQuery, setSearchQuery] = useState('');

    useEffect(() => {
        const parkParam = searchParams.get('park');
        const ecoParam = searchParams.get('ecosystem') as EcosystemId | null;
        if (parkParam) setParkFilter(parkParam);
        if (ecoParam && ECOSYSTEM_LIST.some((e) => e.id === ecoParam)) {
            setEcosystemFilter(ecoParam);
        }
    }, [searchParams]);

    const sortedAll = [...allSpecies].sort((a, b) => a.name.localeCompare(b.name));

    const getFilteredList = () => {
        let list = sortedAll;
        if (filter === 'Native') list = list.filter((s) => s.category === 'Native');
        if (filter === 'Invasive')
            list = list.filter((s) => s.category === 'Introduced');
        if (classFilter !== 'All') {
            list = list.filter((s) => s.taxonomy?.class === classFilter);
        }
        if (ecosystemFilter !== 'All') {
            list = list.filter(
                (s) => getSpeciesEcosystem(s).id === ecosystemFilter
            );
        }
        if (parkFilter) {
            const q = parkFilter.replace(' NP', '').toLowerCase();
            list = list.filter((s) =>
                (s.hotspots || []).some((h) => h.toLowerCase().includes(q))
            );
        }
        if (searchQuery.trim()) {
            const q = searchQuery.toLowerCase();
            list = list.filter(
                (s) =>
                    s.name.toLowerCase().includes(q) ||
                    (s.scientific_name || '').toLowerCase().includes(q) ||
                    (s.hotspots || []).some((h) => h.toLowerCase().includes(q))
            );
        }
        return list;
    };

    const filteredList = getFilteredList();
    const observedCount = filteredList.filter((s) => isUnlocked(s.name)).length;
    const totalUnlockedAll = sortedAll.filter((s) => isUnlocked(s.name)).length;

    return (
        <main className="min-h-screen bg-[#EAE3D2] text-[#162A20] pb-28">
            <AppHeader
                observedCount={observedCount}
                totalCount={filteredList.length}
                filter={filter}
            />

            <div className="max-w-6xl mx-auto px-4 pt-5">
                {/* ============================================================
                    TOP MASTHEAD & BARCODE PROGRESS STRIP (B1 COLLECTOR DECK)
                   ============================================================ */}
                <div className="bg-[#FBF8F1] rounded-2xl border-2 border-[#162A20] tactile-shadow p-4 sm:p-5 mb-6 space-y-4">
                    <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
                        <div>
                            <div className="flex items-center gap-2">
                                <span className="px-2.5 py-0.5 rounded-full bg-[#162A20] text-[#F4DE9B] font-mono-tech text-[10px] font-extrabold uppercase tracking-widest">
                                    ISSUE 01 · NORTH AMERICAN DECK
                                </span>
                                <span className="font-mono-tech text-xs font-extrabold text-[#C84B31]">
                                    {observedCount} / {filteredList.length} PLATES INKED
                                </span>
                            </div>
                            <h2 className="font-editorial text-3xl sm:text-4xl text-[#162A20] leading-none mt-1.5">
                                AniDex Collector Press
                            </h2>
                        </div>

                        {/* Search Input + Total/Native/Invasive Segmented Pills */}
                        <div className="flex flex-wrap items-center gap-2.5">
                            <div className="relative flex-1 sm:w-64 min-w-[200px]">
                                <Search
                                    size={15}
                                    className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[#5A655E]"
                                />
                                <input
                                    type="text"
                                    placeholder="Search species or park..."
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    className="w-full pl-9 pr-8 py-2 bg-[#F7F3E8] border-2 border-[#162A20] rounded-full text-xs font-bold text-[#162A20] placeholder:text-[#5A655E] focus:outline-none focus:bg-white"
                                />
                                {searchQuery && (
                                    <button
                                        onClick={() => setSearchQuery('')}
                                        className="absolute right-3 top-1/2 -translate-y-1/2 text-[#5A655E] hover:text-[#162A20]"
                                    >
                                        <X size={14} />
                                    </button>
                                )}
                            </div>

                            <div className="flex bg-[#EAE3D2] rounded-full p-1 border-2 border-[#162A20]">
                                {(['Total', 'Native', 'Invasive'] as const).map((f) => (
                                    <button
                                        key={f}
                                        onClick={() => setFilter(f)}
                                        className={`px-3 py-1 rounded-full text-[11px] font-mono-tech font-extrabold uppercase tracking-wider transition-all cursor-pointer ${
                                            filter === f
                                                ? f === 'Invasive'
                                                    ? 'bg-[#C84B31] text-[#FBF8F1]'
                                                    : 'bg-[#162A20] text-[#F4DE9B]'
                                                : 'text-[#162A20] hover:bg-[#F4DE9B]/50'
                                        }`}
                                    >
                                        {f}
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* 100-Species Barcode Tick Matrix */}
                    {sortedAll.length > 0 && (
                        <div className="pt-1">
                            <div className="flex items-center justify-between text-[10px] font-mono-tech font-bold text-[#5A655E] mb-1">
                                <span>COLLECTION PRESS MATRIX</span>
                                <span>
                                    {Math.round(
                                        (totalUnlockedAll / Math.max(1, sortedAll.length)) * 100
                                    )}
                                    % COMPLETE ({totalUnlockedAll}/{sortedAll.length})
                                </span>
                            </div>
                            <div className="grid grid-flow-col auto-cols-fr gap-[2px] h-3 bg-[#EAE3D2] p-0.5 rounded-md border border-[#162A20]">
                                {sortedAll.map((s) => {
                                    const unl = isUnlocked(s.name);
                                    return (
                                        <div
                                            key={s.name}
                                            title={`${s.name} (${unl ? 'Inked' : 'Un-inked'})`}
                                            className={`h-full rounded-[1px] ${
                                                unl ? 'bg-[#C84B31]' : 'bg-[#162A20]/15'
                                            }`}
                                        />
                                    );
                                })}
                            </div>
                        </div>
                    )}

                    {/* Ecosystem Filter Row (Step D) + Class Filter Row */}
                    <div className="pt-2 border-t border-[#162A20]/15 flex flex-col gap-2.5">
                        {/* Ecosystem Pills */}
                        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 no-scrollbar">
                            <span className="font-mono-tech text-[10px] font-extrabold uppercase tracking-wider text-[#5A655E] mr-1 shrink-0">
                                Biome:
                            </span>
                            <button
                                onClick={() => setEcosystemFilter('All')}
                                className={`px-3 py-1 rounded-full text-xs font-extrabold border-2 border-[#162A20] shrink-0 transition-all cursor-pointer ${
                                    ecosystemFilter === 'All'
                                        ? 'bg-[#162A20] text-[#F4DE9B] tactile-shadow-sm'
                                        : 'bg-[#F7F3E8] text-[#162A20] hover:bg-[#F4DE9B]'
                                }`}
                            >
                                🌎 All Ecosystems
                            </button>
                            {ECOSYSTEM_LIST.map((eco) => {
                                const active = ecosystemFilter === eco.id;
                                return (
                                    <button
                                        key={eco.id}
                                        onClick={() =>
                                            setEcosystemFilter(active ? 'All' : eco.id)
                                        }
                                        className={`px-3 py-1 rounded-full text-xs font-extrabold border-2 border-[#162A20] shrink-0 transition-all cursor-pointer ${
                                            active
                                                ? 'bg-[#C84B31] text-[#FBF8F1] tactile-shadow-sm'
                                                : 'bg-[#F7F3E8] text-[#162A20] hover:bg-[#F4DE9B]'
                                        }`}
                                    >
                                        {eco.icon} {eco.shortName}
                                    </button>
                                );
                            })}
                        </div>

                        {/* Taxonomic Class Pills + Active Park Filter Chip */}
                        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 no-scrollbar">
                            <span className="font-mono-tech text-[10px] font-extrabold uppercase tracking-wider text-[#5A655E] mr-1 shrink-0">
                                Class:
                            </span>
                            {CLASSES.map((cls) => {
                                const active = classFilter === cls.id;
                                return (
                                    <button
                                        key={cls.id}
                                        onClick={() =>
                                            setClassFilter(active ? 'All' : cls.id)
                                        }
                                        className={`px-2.5 py-1 rounded-full text-[11px] font-bold border border-[#162A20] shrink-0 transition-all cursor-pointer ${
                                            active
                                                ? 'bg-[#F4DE9B] text-[#162A20] font-extrabold tactile-shadow-sm'
                                                : 'bg-[#FBF8F1] text-[#5A655E] hover:text-[#162A20]'
                                        }`}
                                    >
                                        {cls.label}
                                    </button>
                                );
                            })}

                            {parkFilter && (
                                <button
                                    onClick={() => {
                                        setParkFilter('');
                                        router.replace('/anidex');
                                    }}
                                    className="ml-auto px-3 py-1 rounded-full bg-[#234733] text-[#F4DE9B] border-2 border-[#162A20] text-xs font-extrabold flex items-center gap-1.5 shrink-0 cursor-pointer"
                                >
                                    <span>🏞️ Park: {parkFilter}</span>
                                    <X size={12} />
                                </button>
                            )}
                        </div>
                    </div>
                </div>

                {/* ============================================================
                    STAGE 1: UNIFORM COLLECTOR DECK GRID (A3 UNLOCKED + C1 LOCKED)
                   ============================================================ */}
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-x-4 gap-y-7 pt-3">
                    {filteredList.map((species) => {
                        const unlocked = isUnlocked(species.name);
                        const entries = anidex.filter((e) => e.name === species.name);
                        const photoCount = entries.length;
                        const plateNum = String(
                            sortedAll.findIndex((s) => s.name === species.name) + 1
                        ).padStart(3, '0');
                        const eco = getSpeciesEcosystem(species);
                        const primaryHotspot =
                            species.hotspots && species.hotspots.length > 0
                                ? species.hotspots[0]
                                      .replace('National Park', 'NP')
                                      .replace('State Park', 'SP')
                                : `${eco.name} Habitat`;

                        if (unlocked) {
                            /* ------------------------------------------------------------
                               A3 SPOT-INK COLLECTOR CARD (UNLOCKED)
                               ------------------------------------------------------------ */
                            return (
                                <Link
                                    key={species.name}
                                    href={`/anidex/${encodeURIComponent(species.name)}`}
                                    className="group relative bg-[#FBF8F1] rounded-2xl border-2 border-[#162A20] tactile-shadow hover:-translate-y-1 transition-all flex flex-col pt-2 overflow-visible cursor-pointer"
                                >
                                    {/* Top Collector Registration Strip */}
                                    <div className="px-3 pb-1.5 flex items-center justify-between text-[10px] font-mono-tech font-extrabold">
                                        <span className="text-[#162A20]">#{plateNum}</span>
                                        <div className="flex items-center gap-1">
                                            {species.category === 'Introduced' && (
                                                <span className="px-1.5 py-0.5 rounded bg-[#C84B31] text-[#FBF8F1] text-[8px] uppercase">
                                                    INVASIVE
                                                </span>
                                            )}
                                            <span className="px-1.5 py-0.5 rounded-full bg-[#F4DE9B] border border-[#162A20] text-[#162A20]">
                                                📷 {photoCount}
                                            </span>
                                        </div>
                                    </div>

                                    {/* Spot-Ink Geometric Habitat Window + Top Breakout Animal Art */}
                                    <div className="relative mx-2.5 h-32 rounded-xl border-2 border-[#162A20] overflow-visible">
                                        <div className="w-full h-full rounded-[10px] overflow-hidden">
                                            <SpotInkBackdrop
                                                ecosystemId={eco.id}
                                                className="w-full h-full object-cover"
                                            />
                                        </div>

                                        {/* Animal Illustration Breaking Out of the Top Frame */}
                                        <div className="absolute inset-x-0 -top-4 bottom-1 flex items-center justify-center pointer-events-none z-10">
                                            <img
                                                src={speciesIconUrl(species.name)}
                                                alt={species.name}
                                                className="h-34 w-auto max-w-[88%] object-contain cutout-animal-art group-hover:scale-108 transition-transform duration-300"
                                                onError={(e) => {
                                                    const img = e.target as HTMLImageElement;
                                                    if (!img.src.endsWith('/icons/default.png')) {
                                                        img.src = '/icons/default.png';
                                                    }
                                                }}
                                            />
                                        </div>
                                    </div>

                                    {/* Clean Species Title & Ecosystem Footer (No Park Clutter on Front) */}
                                    <div className="p-3 pt-2.5 flex-1 flex flex-col justify-between">
                                        <div>
                                            <h3 className="font-extrabold text-sm text-[#162A20] leading-tight line-clamp-1 group-hover:text-[#C84B31] transition-colors">
                                                {species.name}
                                            </h3>
                                            {species.scientific_name && (
                                                <p className="text-[10px] font-mono-tech italic text-[#5A655E] truncate mt-0.5">
                                                    {species.scientific_name}
                                                </p>
                                            )}
                                        </div>

                                        <div className="mt-2 pt-2 border-t border-[#162A20]/15 flex items-center justify-between text-[10px] font-mono-tech font-bold">
                                            <span className="text-[#234733] truncate">
                                                {eco.icon} {eco.shortName}
                                            </span>
                                            <span className="text-[#C84B31] group-hover:translate-x-0.5 transition-transform">
                                                Flip / Info →
                                            </span>
                                        </div>
                                    </div>
                                </Link>
                            );
                        }

                        /* ------------------------------------------------------------
                           C1 WOODCUT GHOST PLATE (LOCKED — FULL NAME + WHERE TO SPOT)
                           ------------------------------------------------------------ */
                        return (
                            <Link
                                key={species.name}
                                href={`/anidex/${encodeURIComponent(species.name)}?tab=where`}
                                className="group relative parchment-dots rounded-2xl border-2 border-dashed border-[#162A20]/70 hover:border-[#162A20] hover:bg-[#F7F3E8] hover:-translate-y-1 transition-all flex flex-col pt-2 overflow-hidden cursor-pointer"
                            >
                                {/* Top Un-Inked Plate Header */}
                                <div className="px-3 pb-1.5 flex items-center justify-between text-[10px] font-mono-tech font-bold text-[#5A655E]">
                                    <span>#{plateNum}</span>
                                    <span className="px-1.5 py-0.5 rounded bg-[#EAE3D2] border border-[#162A20]/40 text-[9px] uppercase tracking-wider text-[#162A20]">
                                        UN-INKED
                                    </span>
                                </div>

                                {/* 2-Tone Woodcut Silhouette Window */}
                                <div className="relative mx-2.5 h-32 rounded-xl border border-dashed border-[#162A20]/40 bg-[#EAE3D2]/60 flex items-center justify-center p-2 overflow-hidden">
                                    <img
                                        src={speciesIconUrl(species.name)}
                                        alt={species.name}
                                        className="h-26 w-auto max-w-[82%] object-contain woodcut-ghost-art group-hover:scale-105 transition-transform"
                                        onError={(e) => {
                                            const img = e.target as HTMLImageElement;
                                            if (!img.src.endsWith('/icons/default.png')) {
                                                img.src = '/icons/default.png';
                                            }
                                        }}
                                    />
                                    <span className="absolute bottom-1.5 right-2 text-[9px] font-mono-tech font-bold text-[#5A655E]">
                                        {eco.icon} {eco.shortName}
                                    </span>
                                </div>

                                {/* Full Species Name + Actionable "Where to Spot" Bar */}
                                <div className="p-3 pt-2.5 flex-1 flex flex-col justify-between">
                                    <div>
                                        <h3 className="font-extrabold text-sm text-[#162A20]/85 leading-tight line-clamp-1">
                                            {species.name}
                                        </h3>
                                        {species.scientific_name && (
                                            <p className="text-[10px] font-mono-tech italic text-[#5A655E] truncate mt-0.5">
                                                {species.scientific_name}
                                            </p>
                                        )}
                                    </div>

                                    <div className="mt-2 px-2 py-1 rounded-lg bg-[#F4DE9B]/80 border border-[#162A20] text-[9px] font-mono-tech font-extrabold text-[#162A20] flex items-center gap-1 truncate group-hover:bg-[#E8B84B]">
                                        <Compass size={11} className="text-[#C84B31] shrink-0" />
                                        <span className="truncate">Spot: {primaryHotspot}</span>
                                    </div>
                                </div>
                            </Link>
                        );
                    })}
                </div>

                {filteredList.length === 0 && (
                    <div className="text-center py-14 bg-[#FBF8F1] rounded-2xl border-2 border-dashed border-[#162A20]/40 mt-4">
                        <Sparkles size={40} className="mx-auto mb-3 text-[#C84B31]" />
                        <p className="font-editorial text-2xl text-[#162A20]">
                            No Collector Plates Match This Filter
                        </p>
                        <button
                            onClick={() => {
                                setFilter('Total');
                                setEcosystemFilter('All');
                                setClassFilter('All');
                                setParkFilter('');
                                setSearchQuery('');
                            }}
                            className="mt-4 tactile-btn px-4 py-2 rounded-full bg-[#F4DE9B] text-[#162A20] text-xs font-extrabold uppercase tracking-wider cursor-pointer"
                        >
                            Reset All Filters
                        </button>
                    </div>
                )}
            </div>
        </main>
    );
}

export default function AnidexPage() {
    return (
        <Suspense
            fallback={
                <div className="min-h-screen bg-[#EAE3D2] flex items-center justify-center font-mono-tech text-xs font-bold uppercase">
                    Loading Collector Deck...
                </div>
            }
        >
            <AnidexContent />
        </Suspense>
    );
}
