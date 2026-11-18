'use client';

import { useParams, useRouter, useSearchParams } from 'next/navigation';
import { useApp } from '@/lib/AppContext';
import { useState, useEffect, Suspense } from 'react';
import {
    ArrowLeft,
    MapPin,
    CheckCircle,
    Globe,
    Dna,
    ExternalLink,
    RotateCw,
    Camera,
    Sparkles,
    Calendar,
} from 'lucide-react';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import AppHeader from '@/components/AppHeader';
import SpotInkBackdrop from '@/components/SpotInkBackdrop';
import WhereToSee from '@/components/WhereToSee';
import { speciesIconUrl } from '@/lib/species';
import {
    getSpeciesEcosystem,
    formatLocationStamp,
    ParkRecord,
} from '@/lib/ecosystems';

const SpeciesMap = dynamic(() => import('@/components/SpeciesMap'), {
    ssr: false,
});

const MONTHS = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'];

function SpeciesDetailContent() {
    const params = useParams();
    const router = useRouter();
    const searchParams = useSearchParams();
    const { allSpecies, anidex, getStaticInfo, location, isUnlocked } = useApp();

    const [isFlipped, setIsFlipped] = useState(false);
    const [detailTab, setDetailTab] = useState<'map' | 'where' | 'taxonomy'>('map');
    const [parks, setParks] = useState<ParkRecord[]>([]);

    useEffect(() => {
        const tabParam = searchParams.get('tab');
        if (tabParam === 'where' || tabParam === 'taxonomy' || tabParam === 'map') {
            setDetailTab(tabParam);
        }
        const sideParam = searchParams.get('side');
        if (sideParam === 'back') {
            setIsFlipped(true);
        }
    }, [searchParams]);

    useEffect(() => {
        fetch('/parks.json')
            .then((r) => r.json())
            .then((data) => setParks(data))
            .catch(() => {});
    }, []);

    const speciesName = decodeURIComponent(params.id as string);
    const unlocked = isUnlocked(speciesName);
    const speciesInfo = getStaticInfo(speciesName);
    const entries = anidex.filter((e) => e.name === speciesName);

    // Also include local anonymous observations from localStorage if not already in `anidex`
    const [localEntries, setLocalEntries] = useState<any[]>([]);
    useEffect(() => {
        try {
            const raw = localStorage.getItem('local_observations');
            if (raw) {
                const parsed = JSON.parse(raw);
                const matching = parsed
                    .filter((o: any) => o.common_name === speciesName)
                    .map((o: any) => ({
                        id: String(o.id),
                        name: o.common_name,
                        timestamp: new Date(o.created_at),
                        imageUrl: o.image_url,
                        verified: true,
                        lat: o.lat,
                        lng: o.lng,
                    }));
                setLocalEntries(matching);
            }
        } catch {}
    }, [speciesName]);

    const allSpeciesEntries = [
        ...entries,
        ...localEntries.filter(
            (le) =>
                !entries.some(
                    (e) =>
                        Math.abs(
                            new Date(e.timestamp).getTime() -
                                new Date(le.timestamp).getTime()
                        ) < 5000
                )
        ),
    ];
    const latestEntry = allSpeciesEntries[0];

    if (!speciesInfo) {
        return (
            <main className="min-h-screen bg-[#EAE3D2] flex flex-col">
                <AppHeader />
                <div className="flex-1 flex items-center justify-center p-6">
                    <div className="bg-[#FBF8F1] p-8 rounded-2xl border-2 border-[#162A20] tactile-shadow text-center">
                        <p className="font-editorial text-3xl text-[#162A20] mb-3">
                            Collector Plate Not Found
                        </p>
                        <Link
                            href="/anidex"
                            className="tactile-btn inline-block px-5 py-2.5 rounded-full bg-[#C84B31] text-[#FBF8F1] text-xs font-extrabold uppercase tracking-wider"
                        >
                            ← Back to AniDex Deck
                        </Link>
                    </div>
                </div>
            </main>
        );
    }

    const sortedAll = [...allSpecies].sort((a, b) => a.name.localeCompare(b.name));
    const plateNum = String(
        Math.max(1, sortedAll.findIndex((s) => s.name === speciesName) + 1)
    ).padStart(3, '0');
    const eco = getSpeciesEcosystem(speciesInfo);

    const observations = allSpeciesEntries
        .map((e) => ({
            lat: e.lat || 0,
            lng: e.lng || 0,
            date: new Date(e.timestamp),
        }))
        .filter((o) => o.lat !== 0);

    // Compute which months (0..11) have sightings
    const activeMonths = new Set(
        allSpeciesEntries.map((e) => new Date(e.timestamp).getMonth())
    );

    const getIUCNStyle = (status?: string) => {
        switch (status) {
            case 'Least Concern':
                return 'bg-[#234733] text-[#F4DE9B]';
            case 'Near Threatened':
                return 'bg-[#E8B84B] text-[#162A20]';
            case 'Vulnerable':
            case 'Endangered':
            case 'Critically Endangered':
                return 'bg-[#C84B31] text-[#FBF8F1]';
            default:
                return 'bg-[#EAE3D2] text-[#162A20]';
        }
    };

    return (
        <main className="min-h-screen bg-[#EAE3D2] text-[#162A20] pb-28">
            <AppHeader />

            <div className="max-w-5xl mx-auto px-4 pt-5">
                {/* ============================================================
                    TOP STAGE CONTROLS: BACK TO DECK + STAGE 2 / STAGE 3 FLIP BAR
                   ============================================================ */}
                <div className="flex flex-wrap items-center justify-between gap-3 mb-5">
                    <button
                        onClick={() => router.push('/anidex')}
                        className="tactile-btn px-4 py-2 rounded-full bg-[#FBF8F1] hover:bg-[#F4DE9B] text-[#162A20] text-xs font-extrabold uppercase tracking-wider flex items-center gap-2 cursor-pointer"
                    >
                        <ArrowLeft size={15} />
                        <span>Back to Deck</span>
                    </button>

                    <div className="flex items-center gap-2">
                        <span className="hidden sm:inline font-mono-tech text-[11px] font-bold text-[#5A655E]">
                            {isFlipped
                                ? 'STAGE 3 · CARD BACK (POLAROID WALL)'
                                : 'STAGE 2 · EXPANDED CARD FRONT'}
                        </span>
                        <button
                            onClick={() => setIsFlipped(!isFlipped)}
                            className={`tactile-btn px-4 py-2 rounded-full text-xs font-extrabold uppercase tracking-wider flex items-center gap-2 cursor-pointer ${
                                isFlipped
                                    ? 'bg-[#162A20] text-[#F4DE9B]'
                                    : 'bg-[#C84B31] text-[#FBF8F1]'
                            }`}
                        >
                            <RotateCw
                                size={15}
                                className={`transition-transform duration-500 ${
                                    isFlipped ? 'rotate-180' : ''
                                }`}
                            />
                            <span>
                                {isFlipped
                                    ? 'Flip to Front: Species Field Guide'
                                    : `Flip Card to Back: Polaroid Wall (${allSpeciesEntries.length})`}
                            </span>
                        </button>
                    </div>
                </div>

                {/* ============================================================
                    THE EXPANDED 2-SIDED COLLECTOR CARD (STAGE 2 FRONT vs STAGE 3 BACK)
                   ============================================================ */}
                {!isFlipped ? (
                    /* ------------------------------------------------------------
                       STAGE 2: EXPANDED COLLECTOR CARD FRONT (SPECIES ENCYCLOPEDIA)
                       ------------------------------------------------------------ */
                    <div className="bg-[#FBF8F1] rounded-3xl border-2 border-[#162A20] tactile-shadow-lg overflow-hidden transition-all duration-300">
                        {/* Top Collector Plate Registration Bar */}
                        <div className="bg-[#162A20] text-[#F4DE9B] px-5 py-3 flex flex-wrap items-center justify-between gap-2 border-b-2 border-[#162A20]">
                            <div className="flex items-center gap-2.5">
                                <span className="font-mono-tech text-xs font-extrabold uppercase tracking-widest px-2.5 py-0.5 rounded bg-[#C84B31] text-[#FBF8F1]">
                                    PLATE #{plateNum}
                                </span>
                                <span className="font-mono-tech text-xs font-bold uppercase tracking-wider">
                                    {unlocked
                                        ? '✓ INKED COLLECTOR EDITION'
                                        : 'UN-INKED WOODCUT SPECIMEN'}
                                </span>
                            </div>

                            <button
                                onClick={() => setIsFlipped(true)}
                                className="font-mono-tech text-[11px] font-extrabold uppercase tracking-wider text-[#F4DE9B] hover:text-white flex items-center gap-1.5 underline cursor-pointer"
                            >
                                <RotateCw size={13} />
                                <span>Click to Flip Card Over →</span>
                            </button>
                        </div>

                        {/* Hero Collector Card Front Upper Section (Clickable to Flip!) */}
                        <div className="grid grid-cols-1 md:grid-cols-12 border-b-2 border-[#162A20]">
                            {/* Left 5 Cols: A3 Spot-Ink Habitat Window + Breakout Art */}
                            <div
                                onClick={() => setIsFlipped(true)}
                                title="Click card art to flip to Polaroid Wall"
                                className="md:col-span-5 relative min-h-[260px] p-5 flex flex-col justify-between border-b-2 md:border-b-0 md:border-r-2 border-[#162A20] cursor-pointer group overflow-hidden"
                            >
                                <div className="absolute inset-0">
                                    {unlocked ? (
                                        <SpotInkBackdrop
                                            ecosystemId={eco.id}
                                            variant="cover"
                                            className="w-full h-full object-cover"
                                        />
                                    ) : (
                                        <div className="w-full h-full parchment-dots" />
                                    )}
                                </div>

                                <div className="relative z-10 flex items-center justify-between">
                                    <span className="px-2.5 py-1 rounded-full bg-[#FBF8F1] border-2 border-[#162A20] font-mono-tech text-[10px] font-extrabold uppercase text-[#162A20] tactile-shadow-sm">
                                        {eco.icon} {eco.name}
                                    </span>
                                    <span className="px-2.5 py-1 rounded-full bg-[#162A20]/85 text-[#F4DE9B] font-mono-tech text-[10px] font-bold uppercase opacity-90 group-hover:bg-[#C84B31] group-hover:text-white transition-colors">
                                        🔄 Tap to Flip
                                    </span>
                                </div>

                                {/* Center Animal Art */}
                                <div className="relative z-10 my-2 flex items-center justify-center">
                                    <img
                                        src={speciesIconUrl(speciesName)}
                                        alt={speciesName}
                                        className={`h-48 sm:h-54 w-auto object-contain transition-transform duration-300 group-hover:scale-105 ${
                                            unlocked
                                                ? 'cutout-animal-art'
                                                : 'woodcut-ghost-art'
                                        }`}
                                        onError={(e) => {
                                            const img = e.target as HTMLImageElement;
                                            if (!img.src.endsWith('/icons/default.png')) {
                                                img.src = '/icons/default.png';
                                            }
                                        }}
                                    />
                                </div>

                                <div className="relative z-10 flex items-center justify-between text-[10px] font-mono-tech font-bold">
                                    <span className="px-2.5 py-0.5 rounded bg-[#FBF8F1]/90 border border-[#162A20] text-[#162A20]">
                                        {speciesInfo.taxonomy?.class || 'Fauna'} ·{' '}
                                        {speciesInfo.taxonomy?.family || 'Wild'}
                                    </span>
                                    {allSpeciesEntries.length > 0 && (
                                        <span className="px-2.5 py-0.5 rounded bg-[#F4DE9B] border border-[#162A20] text-[#162A20] font-extrabold">
                                            📷 {allSpeciesEntries.length} Polaroid
                                            {allSpeciesEntries.length > 1 ? 's' : ''} on Back
                                        </span>
                                    )}
                                </div>
                            </div>

                            {/* Right 7 Cols: Species Field Dossier Info */}
                            <div className="md:col-span-7 p-6 sm:p-7 flex flex-col justify-between bg-[#FBF8F1]">
                                <div>
                                    {/* Badges Row: Native/Invasive + IUCN + Ecosystem Tag */}
                                    <div className="flex flex-wrap gap-2 mb-3">
                                        <span
                                            className={`text-[11px] font-mono-tech font-extrabold px-3 py-1 rounded-full border-2 border-[#162A20] tactile-shadow-sm ${
                                                speciesInfo.category === 'Introduced'
                                                    ? 'bg-[#C84B31] text-[#FBF8F1]'
                                                    : 'bg-[#F4DE9B] text-[#162A20]'
                                            }`}
                                        >
                                            {speciesInfo.category === 'Introduced'
                                                ? '⚠ INVASIVE / INTRODUCED'
                                                : `✓ ${speciesInfo.category?.toUpperCase() || 'NATIVE'}`}
                                        </span>

                                        {speciesInfo.iucn && (
                                            <span
                                                className={`text-[11px] font-mono-tech font-extrabold px-3 py-1 rounded-full border-2 border-[#162A20] tactile-shadow-sm ${getIUCNStyle(
                                                    speciesInfo.iucn
                                                )}`}
                                            >
                                                IUCN: {speciesInfo.iucn.toUpperCase()}
                                            </span>
                                        )}

                                        <Link
                                            href={`/anidex?ecosystem=${eco.id}`}
                                            className="text-[11px] font-mono-tech font-extrabold px-3 py-1 rounded-full border-2 border-[#162A20] bg-[#F7F3E8] hover:bg-[#F4DE9B] text-[#162A20] tactile-shadow-sm"
                                        >
                                            {eco.icon} {eco.name} →
                                        </Link>
                                    </div>

                                    {/* Title & Scientific Name */}
                                    <h1 className="font-editorial text-4xl sm:text-5xl text-[#162A20] leading-none">
                                        {speciesName}
                                    </h1>
                                    {speciesInfo.scientific_name && (
                                        <p className="font-mono-tech text-xs sm:text-sm italic text-[#C84B31] font-bold mt-1">
                                            {speciesInfo.scientific_name}
                                        </p>
                                    )}

                                    {/* Field Description */}
                                    {speciesInfo.description && (
                                        <p className="text-[#162A20]/85 text-sm leading-relaxed mt-4 bg-[#F7F3E8] p-4 rounded-xl border-2 border-[#162A20]">
                                            {speciesInfo.description}
                                        </p>
                                    )}
                                </div>

                                {/* Bottom Summary Strip: Latest Sighting or Where to Spot + Flip CTA */}
                                <div className="mt-5 pt-4 border-t-2 border-[#162A20]/15 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                                    {unlocked && latestEntry ? (
                                        <div className="text-xs">
                                            <span className="font-mono-tech text-[10px] font-extrabold uppercase tracking-wider text-[#234733] block">
                                                ✓ LAST STAMPED BY YOU
                                            </span>
                                            <span className="font-extrabold text-[#162A20]">
                                                📅{' '}
                                                {new Date(
                                                    latestEntry.timestamp
                                                ).toLocaleDateString()}{' '}
                                                ·{' '}
                                                {formatLocationStamp(
                                                    latestEntry.lat,
                                                    latestEntry.lng,
                                                    parks
                                                )}
                                            </span>
                                        </div>
                                    ) : (
                                        <div className="text-xs">
                                            <span className="font-mono-tech text-[10px] font-extrabold uppercase tracking-wider text-[#C84B31] block">
                                                🔎 UN-INKED SPECIMEN · PRIME HABITAT
                                            </span>
                                            <span className="font-extrabold text-[#162A20]">
                                                {speciesInfo.hotspots?.[0] || eco.description}
                                            </span>
                                        </div>
                                    )}

                                    <button
                                        onClick={() => setIsFlipped(true)}
                                        className="tactile-btn px-4 py-2 rounded-full bg-[#F4DE9B] hover:bg-[#E8B84B] text-[#162A20] text-xs font-extrabold uppercase tracking-wider flex items-center gap-1.5 self-start sm:self-auto cursor-pointer shrink-0"
                                    >
                                        <RotateCw size={13} />
                                        <span>
                                            Flip to Polaroid Wall ({allSpeciesEntries.length})
                                        </span>
                                    </button>
                                </div>
                            </div>
                        </div>

                        {/* ============================================================
                            3 INTERACTIVE FIELD GUIDE TABS ON STAGE 2 FRONT:
                            1. Range Map (SpeciesMap.tsx)
                            2. Where to See (WhereToSee.tsx)
                            3. Taxonomy (6-Rank Classification)
                           ============================================================ */}
                        <div className="bg-[#F7F3E8] px-4 sm:px-6 py-3 border-b-2 border-[#162A20] flex flex-wrap gap-2">
                            <button
                                onClick={() => setDetailTab('map')}
                                className={`flex-1 py-2.5 px-4 rounded-full font-extrabold text-xs uppercase tracking-wider flex items-center justify-center gap-2 border-2 border-[#162A20] transition-all cursor-pointer ${
                                    detailTab === 'map'
                                        ? 'bg-[#162A20] text-[#F4DE9B] tactile-shadow-sm'
                                        : 'bg-[#FBF8F1] text-[#162A20] hover:bg-[#F4DE9B]'
                                }`}
                            >
                                <Globe size={15} />
                                <span>Range Map</span>
                            </button>
                            <button
                                onClick={() => setDetailTab('where')}
                                className={`flex-1 py-2.5 px-4 rounded-full font-extrabold text-xs uppercase tracking-wider flex items-center justify-center gap-2 border-2 border-[#162A20] transition-all cursor-pointer ${
                                    detailTab === 'where'
                                        ? 'bg-[#C84B31] text-[#FBF8F1] tactile-shadow-sm'
                                        : 'bg-[#FBF8F1] text-[#162A20] hover:bg-[#F4DE9B]'
                                }`}
                            >
                                <MapPin size={15} />
                                <span>Where to See</span>
                            </button>
                            <button
                                onClick={() => setDetailTab('taxonomy')}
                                className={`flex-1 py-2.5 px-4 rounded-full font-extrabold text-xs uppercase tracking-wider flex items-center justify-center gap-2 border-2 border-[#162A20] transition-all cursor-pointer ${
                                    detailTab === 'taxonomy'
                                        ? 'bg-[#234733] text-[#F4DE9B] tactile-shadow-sm'
                                        : 'bg-[#FBF8F1] text-[#162A20] hover:bg-[#F4DE9B]'
                                }`}
                            >
                                <Dna size={15} />
                                <span>Taxonomy</span>
                            </button>
                        </div>

                        {/* Tab Content Area */}
                        <div className="min-h-[420px] bg-[#FBF8F1]">
                            {detailTab === 'map' && (
                                <div className="h-[480px]">
                                    {speciesInfo.taxonomy?.taxon_id ? (
                                        <SpeciesMap
                                            taxonId={speciesInfo.taxonomy.taxon_id}
                                            userLat={location?.lat}
                                            userLng={location?.lng}
                                            animalName={speciesName}
                                            observations={observations}
                                        />
                                    ) : (
                                        <div className="h-full flex items-center justify-center text-[#5A655E] font-mono-tech text-xs">
                                            Range map unavailable — no iNaturalist taxon ID
                                        </div>
                                    )}
                                </div>
                            )}

                            {detailTab === 'where' && (
                                <WhereToSee
                                    taxonId={
                                        latestEntry?.taxonId ||
                                        speciesInfo?.taxonomy?.taxon_id
                                    }
                                    userLat={location?.lat}
                                    userLng={location?.lng}
                                    staticHotspots={speciesInfo.hotspots || []}
                                />
                            )}

                            {detailTab === 'taxonomy' && (
                                <div className="p-6">
                                    <div className="flex items-center justify-between mb-4">
                                        <h3 className="font-editorial text-2xl text-[#162A20] flex items-center gap-2">
                                            <Dna size={20} className="text-[#C84B31]" />
                                            <span>Linnaean Scientific Classification</span>
                                        </h3>
                                        {speciesInfo.taxonomy?.taxon_id && (
                                            <a
                                                href={`https://www.inaturalist.org/taxa/${speciesInfo.taxonomy.taxon_id}`}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="tactile-btn px-3.5 py-1.5 rounded-full bg-[#F4DE9B] text-[#162A20] text-xs font-extrabold uppercase tracking-wider inline-flex items-center gap-1.5"
                                            >
                                                <span>View on iNaturalist</span>
                                                <ExternalLink size={13} />
                                            </a>
                                        )}
                                    </div>

                                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3.5">
                                        {(
                                            [
                                                ['Kingdom', speciesInfo.taxonomy?.kingdom],
                                                ['Phylum', speciesInfo.taxonomy?.phylum],
                                                ['Class', speciesInfo.taxonomy?.class],
                                                ['Order', speciesInfo.taxonomy?.order],
                                                ['Family', speciesInfo.taxonomy?.family],
                                                ['Genus', speciesInfo.taxonomy?.genus],
                                            ] as const
                                        ).map(([rank, val]) =>
                                            val ? (
                                                <div
                                                    key={rank}
                                                    className="bg-[#F7F3E8] p-4 rounded-xl border-2 border-[#162A20] tactile-shadow-sm"
                                                >
                                                    <p className="font-mono-tech text-[10px] text-[#5A655E] uppercase font-extrabold tracking-widest">
                                                        {rank}
                                                    </p>
                                                    <p
                                                        className={`text-base font-extrabold text-[#162A20] mt-1 ${
                                                            rank === 'Genus' ? 'italic' : ''
                                                        }`}
                                                    >
                                                        {val}
                                                    </p>
                                                </div>
                                            ) : null
                                        )}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                ) : (
                    /* ------------------------------------------------------------
                       STAGE 3: FLIPPED CARD BACK — YOUR PERSONAL POLAROID WALL!
                       ------------------------------------------------------------ */
                    <div className="bg-[#FBF8F1] rounded-3xl border-2 border-[#162A20] tactile-shadow-lg overflow-hidden">
                        {/* Reverse Card Header Strip */}
                        <div className="bg-[#162A20] text-[#F4DE9B] px-5 py-3.5 flex flex-wrap items-center justify-between gap-3 border-b-2 border-[#162A20]">
                            <div className="flex items-center gap-3">
                                <img
                                    src={speciesIconUrl(speciesName)}
                                    alt={speciesName}
                                    className="w-10 h-10 object-contain bg-[#FBF8F1] rounded-lg border border-[#F4DE9B] p-0.5"
                                />
                                <div>
                                    <span className="font-mono-tech text-[10px] font-extrabold uppercase tracking-widest text-[#C84B31] block">
                                        REVERSE SIDE · PLATE #{plateNum}
                                    </span>
                                    <h2 className="font-editorial text-2xl sm:text-3xl text-[#FBF8F1] leading-none">
                                        {speciesName} · Personal Polaroid Wall
                                    </h2>
                                </div>
                            </div>

                            <button
                                onClick={() => setIsFlipped(false)}
                                className="tactile-btn px-4 py-1.5 rounded-full bg-[#F4DE9B] text-[#162A20] text-xs font-extrabold uppercase tracking-wider flex items-center gap-1.5 cursor-pointer"
                            >
                                <RotateCw size={13} />
                                <span>← Flip Back to Front</span>
                            </button>
                        </div>

                        {/* 12-Month Seasonal Sightings Calendar Strip */}
                        <div className="bg-[#F7F3E8] px-5 py-3.5 border-b-2 border-[#162A20] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                            <div className="flex items-center gap-2">
                                <Calendar size={16} className="text-[#C84B31]" />
                                <span className="font-mono-tech text-xs font-extrabold uppercase tracking-wider text-[#162A20]">
                                    Seasonal Sightings Log ({allSpeciesEntries.length} Photo
                                    {allSpeciesEntries.length === 1 ? '' : 's'})
                                </span>
                            </div>

                            <div className="grid grid-cols-12 gap-1 w-full sm:w-auto sm:min-w-[320px]">
                                {MONTHS.map((m, idx) => {
                                    const seenInMonth = activeMonths.has(idx);
                                    return (
                                        <div
                                            key={idx}
                                            title={
                                                seenInMonth
                                                    ? 'Spotted in this month!'
                                                    : 'Not yet spotted in this month'
                                            }
                                            className={`py-1 rounded text-center font-mono-tech text-[10px] font-extrabold border border-[#162A20] ${
                                                seenInMonth
                                                    ? 'bg-[#C84B31] text-[#FBF8F1]'
                                                    : 'bg-[#EAE3D2] text-[#5A655E]'
                                            }`}
                                        >
                                            {m}
                                        </div>
                                    );
                                })}
                            </div>
                        </div>

                        {/* Cork / Archival Linen Polaroid Wall Surface */}
                        <div className="polaroid-wall-surface p-6 sm:p-8 min-h-[440px]">
                            {allSpeciesEntries.length > 0 ? (
                                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
                                    {allSpeciesEntries.map((entry, idx) => {
                                        const stamp = formatLocationStamp(
                                            entry.lat,
                                            entry.lng,
                                            parks
                                        );
                                        const tiltClass =
                                            idx % 3 === 0
                                                ? '-rotate-1'
                                                : idx % 3 === 1
                                                ? 'rotate-1'
                                                : 'rotate-0';
                                        return (
                                            <div
                                                key={entry.id || idx}
                                                className={`relative bg-[#FBF8F1] p-3.5 pb-4 rounded-xl border-2 border-[#162A20] tactile-shadow-lg transition-transform hover:rotate-0 hover:scale-[1.02] ${tiltClass}`}
                                            >
                                                {/* Masking Tape Strip at Top of Polaroid */}
                                                <div className="absolute -top-2.5 left-1/2 -translate-x-1/2 w-20 h-5 bg-[#F4DE9B]/90 border border-[#162A20]/50 rotate-1 shadow-xs pointer-events-none" />

                                                {/* Polaroid Photo Window */}
                                                <div className="aspect-square w-full bg-[#162A20] rounded-lg overflow-hidden border-2 border-[#162A20] relative">
                                                    {entry.imageUrl ? (
                                                        <img
                                                            src={entry.imageUrl}
                                                            alt={`${speciesName} observation`}
                                                            className="w-full h-full object-cover"
                                                        />
                                                    ) : (
                                                        <div className="w-full h-full flex flex-col items-center justify-center bg-[#F7F3E8] p-4">
                                                            <img
                                                                src={speciesIconUrl(speciesName)}
                                                                alt={speciesName}
                                                                className="w-28 h-28 object-contain cutout-animal-art"
                                                            />
                                                        </div>
                                                    )}

                                                    <span className="absolute top-2 right-2 px-2 py-0.5 rounded-full bg-[#162A20]/85 text-[#F4DE9B] font-mono-tech text-[9px] font-extrabold uppercase border border-[#F4DE9B]/40">
                                                        #{idx + 1}
                                                    </span>
                                                </div>

                                                {/* Polaroid Bottom White Margin with Rubber Park/Location Cancellation Stamp */}
                                                <div className="mt-3 pt-2 border-t border-dashed border-[#162A20]/25 space-y-2">
                                                    <div className="flex items-center justify-between gap-2">
                                                        <span className="inline-block font-mono-tech text-[11px] font-extrabold uppercase tracking-wider text-[#C84B31] border-2 border-[#C84B31] px-2 py-0.5 rounded bg-[#C84B31]/10 -rotate-1 truncate">
                                                            STAMP: {stamp}
                                                        </span>
                                                        {entry.verified && (
                                                            <span className="font-mono-tech text-[10px] font-extrabold text-[#234733] flex items-center gap-0.5 shrink-0">
                                                                <CheckCircle size={12} />
                                                                <span>ID</span>
                                                            </span>
                                                        )}
                                                    </div>

                                                    <div className="flex items-center justify-between text-[11px] font-mono-tech text-[#5A655E]">
                                                        <span>
                                                            {new Date(
                                                                entry.timestamp
                                                            ).toLocaleDateString(undefined, {
                                                                month: 'short',
                                                                day: 'numeric',
                                                                year: 'numeric',
                                                            })}
                                                        </span>
                                                        {entry.lat && entry.lng && (
                                                            <Link
                                                                href="/map"
                                                                className="font-extrabold text-[#162A20] hover:text-[#C84B31] underline flex items-center gap-0.5"
                                                            >
                                                                <MapPin size={11} />
                                                                <span>Pin →</span>
                                                            </Link>
                                                        )}
                                                    </div>
                                                </div>
                                            </div>
                                        );
                                    })}

                                    {/* Add Another Polaroid Slot */}
                                    <Link
                                        href="/"
                                        className="border-2 border-dashed border-[#162A20]/60 rounded-xl p-6 flex flex-col items-center justify-center text-center bg-[#FBF8F1]/50 hover:bg-[#FBF8F1] transition-colors min-h-[260px] group"
                                    >
                                        <div className="w-12 h-12 rounded-full bg-[#F4DE9B] border-2 border-[#162A20] tactile-shadow-sm flex items-center justify-center mb-3 group-hover:scale-105 transition-transform">
                                            <Camera size={22} className="text-[#162A20]" />
                                        </div>
                                        <p className="font-editorial text-2xl text-[#162A20]">
                                            Log Another Sighting
                                        </p>
                                        <p className="text-xs font-mono-tech text-[#5A655E] mt-1">
                                            Stamp a new park or season on your wall →
                                        </p>
                                    </Link>
                                </div>
                            ) : (
                                /* Empty Polaroid Wall State for Un-Inked Species */
                                <div className="max-w-md mx-auto bg-[#FBF8F1] rounded-2xl border-2 border-dashed border-[#162A20] p-8 text-center space-y-4">
                                    <div className="w-16 h-16 mx-auto rounded-2xl bg-[#F4DE9B] border-2 border-[#162A20] tactile-shadow-sm flex items-center justify-center">
                                        <Sparkles size={28} className="text-[#C84B31]" />
                                    </div>
                                    <h3 className="font-editorial text-3xl text-[#162A20]">
                                        No Polaroids Pinned Yet
                                    </h3>
                                    <p className="text-xs sm:text-sm text-[#5A655E] font-medium">
                                        You haven&apos;t photographed a{' '}
                                        <span className="font-extrabold text-[#162A20]">
                                            {speciesName}
                                        </span>{' '}
                                        in the wild yet. Spot one to ink Plate #{plateNum} and pin
                                        your first park-stamped Polaroid here!
                                    </p>
                                    <div className="flex flex-col sm:flex-row gap-2.5 justify-center pt-2">
                                        <Link
                                            href="/"
                                            className="tactile-btn px-5 py-2.5 rounded-full bg-[#C84B31] text-[#FBF8F1] text-xs font-extrabold uppercase tracking-wider"
                                        >
                                            📷 Upload Sighting
                                        </Link>
                                        <button
                                            onClick={() => {
                                                setIsFlipped(false);
                                                setDetailTab('where');
                                            }}
                                            className="tactile-btn px-5 py-2.5 rounded-full bg-[#F4DE9B] text-[#162A20] text-xs font-extrabold uppercase tracking-wider cursor-pointer"
                                        >
                                            📍 See Where to Spot
                                        </button>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </main>
    );
}

export default function SpeciesDetailPage() {
    return (
        <Suspense
            fallback={
                <div className="min-h-screen bg-[#EAE3D2] flex items-center justify-center font-mono-tech text-xs font-bold uppercase">
                    Opening Collector Plate...
                </div>
            }
        >
            <SpeciesDetailContent />
        </Suspense>
    );
}
