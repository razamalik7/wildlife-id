'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Camera, BookOpen, Map } from 'lucide-react';

export default function BottomNav() {
    const pathname = usePathname();
    const isIdentify = pathname === '/';
    const isAnidex = pathname?.startsWith('/anidex');
    const isMap = pathname === '/map';

    return (
        <nav className="md:hidden fixed bottom-0 left-0 right-0 bg-[#F7F3E8] border-t-2 border-[#162A20] z-50 px-3 py-2 shadow-[0_-4px_16px_rgba(22,42,32,0.1)]">
            <div className="flex justify-center items-center gap-2.5 max-w-md mx-auto">
                <Link
                    href="/"
                    className={`flex-1 tactile-btn py-2 px-3 rounded-full flex items-center justify-center gap-1.5 text-xs font-extrabold uppercase tracking-wider ${
                        isIdentify
                            ? 'bg-[#C84B31] text-[#FBF8F1]'
                            : 'bg-[#FBF8F1] text-[#162A20]'
                    }`}
                >
                    <Camera size={16} />
                    <span>Identify</span>
                </Link>
                <Link
                    href="/anidex"
                    className={`flex-1 tactile-btn py-2 px-3 rounded-full flex items-center justify-center gap-1.5 text-xs font-extrabold uppercase tracking-wider ${
                        isAnidex
                            ? 'bg-[#162A20] text-[#F4DE9B]'
                            : 'bg-[#FBF8F1] text-[#162A20]'
                    }`}
                >
                    <BookOpen size={16} />
                    <span>AniDex</span>
                </Link>
                <Link
                    href="/map"
                    className={`flex-1 tactile-btn py-2 px-3 rounded-full flex items-center justify-center gap-1.5 text-xs font-extrabold uppercase tracking-wider ${
                        isMap
                            ? 'bg-[#E8B84B] text-[#162A20]'
                            : 'bg-[#FBF8F1] text-[#162A20]'
                    }`}
                >
                    <Map size={16} />
                    <span>Atlas</span>
                </Link>
            </div>
        </nav>
    );
}
