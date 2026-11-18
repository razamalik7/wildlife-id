'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Camera, BookOpen, Map } from 'lucide-react';
import UserHeader from './UserHeader';
import { useApp } from '@/lib/AppContext';

interface AppHeaderProps {
    observedCount?: number;
    totalCount?: number;
    filter?: string;
}

export default function AppHeader({ observedCount, totalCount, filter }: AppHeaderProps) {
    const pathname = usePathname();
    const { allSpecies, isUnlocked } = useApp();

    const isIdentify = pathname === '/';
    const isAnidex = pathname?.startsWith('/anidex');
    const isMap = pathname === '/map';

    const unlockedTotal = observedCount ?? allSpecies.filter((s) => isUnlocked(s.name)).length;
    const speciesTotal = totalCount ?? (allSpecies.length || 100);

    return (
        <header className="bg-[#F7F3E8] text-[#162A20] px-4 py-3 sticky top-0 z-50 border-b-2 border-[#162A20]">
            <div className="flex justify-between items-center mx-auto max-w-6xl gap-3">
                {/* Brand Mark */}
                <Link href="/" className="flex items-center gap-3 group">
                    <div className="w-11 h-11 rounded-xl bg-[#162A20] border-2 border-[#162A20] tactile-shadow-sm flex items-center justify-center overflow-hidden p-1 group-hover:-translate-y-0.5 transition-transform">
                        <img
                            src="/animl-logo.png"
                            alt="aniML Logo"
                            className="w-full h-full object-contain"
                        />
                    </div>
                    <div className="flex flex-col leading-none">
                        <div className="flex items-baseline gap-1.5">
                            <span className="font-editorial text-2xl sm:text-3xl tracking-tight text-[#162A20]">
                                aniML
                            </span>
                            <span className="font-mono-tech text-[11px] font-extrabold uppercase tracking-widest px-2 py-0.5 rounded-full bg-[#F4DE9B] border border-[#162A20] text-[#162A20]">
                                Field Press
                            </span>
                        </div>
                        <p className="text-[10px] text-[#5A655E] font-mono-tech font-bold tracking-wider mt-0.5 hidden sm:block">
                            SPOT • STAMP • COLLECT NORTH AMERICAN WILDLIFE
                        </p>
                    </div>
                </Link>

                {/* Desktop Tactile 3-Tab Pill Navigation (Option 31 / 39 Chrome) */}
                <nav className="hidden md:flex items-center gap-2">
                    <Link
                        href="/"
                        className={`tactile-btn px-4 py-2 rounded-full text-xs font-extrabold uppercase tracking-wider flex items-center gap-2 ${
                            isIdentify
                                ? 'bg-[#C84B31] text-[#FBF8F1]'
                                : 'bg-[#FBF8F1] text-[#162A20] hover:bg-[#F4DE9B]'
                        }`}
                    >
                        <Camera size={15} />
                        <span>Identify</span>
                    </Link>
                    <Link
                        href="/anidex"
                        className={`tactile-btn px-4 py-2 rounded-full text-xs font-extrabold uppercase tracking-wider flex items-center gap-2 ${
                            isAnidex
                                ? 'bg-[#162A20] text-[#F4DE9B]'
                                : 'bg-[#FBF8F1] text-[#162A20] hover:bg-[#F4DE9B]'
                        }`}
                    >
                        <BookOpen size={15} />
                        <span>AniDex</span>
                        <span
                            className={`px-1.5 py-0.5 rounded-full text-[10px] font-mono-tech ${
                                isAnidex
                                    ? 'bg-[#C84B31] text-[#FBF8F1]'
                                    : 'bg-[#EAE3D2] text-[#162A20]'
                            }`}
                        >
                            {unlockedTotal}/{speciesTotal}
                        </span>
                    </Link>
                    <Link
                        href="/map"
                        className={`tactile-btn px-4 py-2 rounded-full text-xs font-extrabold uppercase tracking-wider flex items-center gap-2 ${
                            isMap
                                ? 'bg-[#E8B84B] text-[#162A20]'
                                : 'bg-[#FBF8F1] text-[#162A20] hover:bg-[#F4DE9B]'
                        }`}
                    >
                        <Map size={15} />
                        <span>Park & Pin Atlas</span>
                    </Link>
                </nav>

                {/* Right: Collection Counter (Mobile) + Auth */}
                <div className="flex items-center gap-2.5">
                    {isAnidex && (
                        <div className="md:hidden text-[11px] bg-[#F4DE9B] border-2 border-[#162A20] tactile-shadow-sm px-2.5 py-1 rounded-full font-mono-tech font-bold text-[#162A20] flex items-center gap-1.5">
                            <span className="w-2 h-2 bg-[#C84B31] rounded-full"></span>
                            <span>{unlockedTotal}</span>
                            <span className="opacity-50">/</span>
                            <span>{speciesTotal}</span>
                            {filter && filter !== 'Total' && (
                                <span className="hidden xs:inline">({filter.toUpperCase()})</span>
                            )}
                        </div>
                    )}
                    <UserHeader />
                </div>
            </div>
        </header>
    );
}
