'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import UserHeader from './UserHeader';

interface AppHeaderProps {
    observedCount?: number;
    totalCount?: number;
    filter?: string;
}

export default function AppHeader({ observedCount, totalCount, filter }: AppHeaderProps) {
    const pathname = usePathname();
    const isAnidex = pathname?.startsWith('/anidex');

    return (
        <header className="bg-stone-900 text-stone-50 p-4 shadow-lg sticky top-0 z-50 border-b border-stone-800">
            <div className="flex justify-between items-center mx-auto max-w-6xl">
                {/* Logo */}
                <Link href="/" className="flex items-center gap-4 hover:opacity-90 transition-opacity">
                    <div className="relative w-14 h-14 flex items-center justify-center">
                        <img
                            src="/animl-logo.png"
                            alt="AniML Logo"
                            className="w-full h-full object-contain"
                        />
                    </div>
                    <div className="flex flex-col leading-none justify-center">
                        <h1 className="text-3xl font-black tracking-tighter text-white">
                            Ani<span className="text-emerald-400 font-mono">ML</span>
                        </h1>
                        <span className="text-[10px] font-bold text-emerald-500 tracking-[0.2em] uppercase mt-0.5">
                            Computer Vision
                        </span>
                    </div>
                </Link>

                {/* Progress Counter (only on anidex) */}
                {isAnidex && observedCount !== undefined && totalCount !== undefined && (
                    <div className="text-xs bg-stone-800 border border-stone-700 px-3 py-1.5 rounded-full font-mono text-stone-300 shadow-sm flex items-center gap-2">
                        <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>
                        <span className="text-white font-bold">{observedCount}</span>
                        <span className="text-stone-500 mx-1">/</span>
                        {totalCount} {filter?.toUpperCase() || 'TOTAL'}
                    </div>
                )}

                {/* User Auth */}
                <UserHeader />
            </div>
        </header>
    );
}
