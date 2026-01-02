'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Camera, PawPrint, Map } from 'lucide-react';

export default function BottomNav() {
    const pathname = usePathname();
    const isAnidex = pathname?.startsWith('/anidex');
    const isMap = pathname === '/map';

    return (
        <nav className="fixed bottom-0 left-0 right-0 bg-white border-t border-stone-200 z-50 shadow-lg">
            <div className="flex justify-center items-center max-w-md mx-auto">
                <Link
                    href="/"
                    className={`flex-1 flex flex-col items-center gap-1 py-3 transition-all ${pathname === '/'
                        ? 'text-emerald-600'
                        : 'text-stone-400 hover:text-stone-600'
                        }`}
                >
                    <Camera size={24} />
                    <span className="text-xs font-bold">Identify</span>
                </Link>
                <Link
                    href="/anidex"
                    className={`flex-1 flex flex-col items-center gap-1 py-3 transition-all ${isAnidex
                        ? 'text-emerald-600'
                        : 'text-stone-400 hover:text-stone-600'
                        }`}
                >
                    <PawPrint size={24} />
                    <span className="text-xs font-bold">AniDex</span>
                </Link>
                <Link
                    href="/map"
                    className={`flex-1 flex flex-col items-center gap-1 py-3 transition-all ${isMap
                        ? 'text-emerald-600'
                        : 'text-stone-400 hover:text-stone-600'
                        }`}
                >
                    <Map size={24} />
                    <span className="text-xs font-bold">My World</span>
                </Link>
            </div>
        </nav>
    );
}
