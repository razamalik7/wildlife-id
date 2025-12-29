'use client';

import { useState } from 'react';
import { useApp } from '@/lib/AppContext';
import Link from 'next/link';
import AppHeader from '@/components/AppHeader';
import { Search, Lock, CheckCircle, Leaf, PawPrint } from 'lucide-react';

export default function AnidexPage() {
    const { allSpecies, anidex, isUnlocked, getStaticInfo } = useApp();
    const [filter, setFilter] = useState<'Total' | 'Native' | 'Invasive'>('Total');
    const [searchQuery, setSearchQuery] = useState('');

    const getFilteredList = () => {
        let list = allSpecies;
        if (filter === 'Native') list = list.filter(s => s.category === 'Native');
        if (filter === 'Invasive') list = list.filter(s => s.category === 'Invasive');
        if (searchQuery) {
            list = list.filter(s => s.name.toLowerCase().includes(searchQuery.toLowerCase()));
        }
        // Sort alphabetically
        return list.sort((a, b) => a.name.localeCompare(b.name));
    };

    const getIconPath = (name: string) => {
        const slug = name.toLowerCase().replace(/ /g, '_');
        return `/icons/${slug}.png`;
    };

    const filteredList = getFilteredList();
    const observedCount = filteredList.filter(s => isUnlocked(s.name)).length;

    return (
        <main className="min-h-screen bg-stone-100 text-stone-900 pb-24">
            <AppHeader
                observedCount={observedCount}
                totalCount={filteredList.length}
                filter={filter}
            />

            <div className="max-w-6xl mx-auto p-4">
                {/* Header & Filters */}
                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-6">
                    <h2 className="text-2xl font-bold text-stone-800 flex items-center gap-2">
                        <PawPrint className="text-emerald-600" />
                        AniDex Collection
                    </h2>

                    <div className="flex flex-wrap items-center gap-2">
                        {/* Search */}
                        <div className="relative flex-1 min-w-[200px]">
                            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-stone-400" />
                            <input
                                type="text"
                                placeholder="Search species..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                className="w-full pl-9 pr-4 py-2 bg-white border border-stone-200 rounded-xl text-sm focus:outline-none focus:border-emerald-500"
                            />
                        </div>

                        {/* Filter Tabs */}
                        <div className="flex bg-stone-200 rounded-full p-1">
                            {(['Total', 'Native', 'Invasive'] as const).map((f) => (
                                <button
                                    key={f}
                                    onClick={() => setFilter(f)}
                                    className={`px-3 py-1 rounded-full text-xs font-bold transition-all ${filter === f
                                        ? 'bg-stone-800 text-white'
                                        : 'text-stone-500 hover:text-stone-700'
                                        }`}
                                >
                                    {f}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Species Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                    {filteredList.map((species) => {
                        const unlocked = isUnlocked(species.name);
                        const entry = anidex.find(e => e.name === species.name);

                        return (
                            <Link
                                key={species.name}
                                href={`/anidex/${encodeURIComponent(species.name)}`}
                                className={`relative bg-white rounded-2xl p-3 border transition-all ${unlocked
                                    ? 'border-emerald-300 shadow-md hover:shadow-lg hover:-translate-y-0.5'
                                    : 'border-stone-200 hover:shadow-md hover:-translate-y-0.5'
                                    } cursor-pointer`}
                            >
                                {/* Status Badge */}
                                <div className="absolute -top-1 -right-1 z-10">
                                    {unlocked ? (
                                        <CheckCircle size={18} className="text-emerald-500 bg-white rounded-full" />
                                    ) : (
                                        <Lock size={14} className="text-stone-400 bg-white rounded-full p-0.5" />
                                    )}
                                </div>

                                {/* Icon */}
                                <div className={`aspect-square bg-stone-100 rounded-xl mb-2 flex items-center justify-center overflow-hidden ${!unlocked && 'grayscale'}`}>
                                    <img
                                        src={getIconPath(species.name)}
                                        alt={species.name}
                                        className="w-3/4 h-3/4 object-contain"
                                        onError={(e) => {
                                            (e.target as HTMLImageElement).src = '/icons/default.png';
                                        }}
                                    />
                                </div>

                                {/* Name */}
                                <p className={`text-[10px] font-bold text-center leading-tight ${unlocked ? 'text-stone-700' : 'text-stone-400'
                                    }`}>
                                    {species.name}
                                </p>

                                {/* Category Indicator */}
                                {species.category === 'Invasive' && (
                                    <div className="mt-1 flex justify-center">
                                        <span className="text-[8px] bg-red-100 text-red-600 px-1.5 py-0.5 rounded-full font-bold">
                                            INVASIVE
                                        </span>
                                    </div>
                                )}
                            </Link>
                        );
                    })}
                </div>

                {filteredList.length === 0 && (
                    <div className="text-center py-12 text-stone-400">
                        <PawPrint size={48} className="mx-auto mb-4 opacity-30" />
                        <p>No species found</p>
                    </div>
                )}
            </div>
        </main>
    );
}
