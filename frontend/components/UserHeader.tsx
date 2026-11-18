'use client';

import { useState, useEffect, useRef } from 'react';
import { supabase } from '@/lib/supabase';
import { User } from '@supabase/supabase-js';
import AuthModal from './AuthModal';

export default function UserHeader() {
    const [user, setUser] = useState<User | null>(null);
    const [showAuthModal, setShowAuthModal] = useState(false);
    const [showDropdown, setShowDropdown] = useState(false);
    const buttonRef = useRef<HTMLButtonElement>(null);
    const [dropdownPosition, setDropdownPosition] = useState({ top: 0, right: 0 });

    useEffect(() => {
        supabase.auth.getSession().then(({ data: { session } }) => {
            setUser(session?.user ?? null);
        });

        const {
            data: { subscription },
        } = supabase.auth.onAuthStateChange((_event, session) => {
            setUser(session?.user ?? null);
        });

        return () => subscription.unsubscribe();
    }, []);

    const handleSignOut = async () => {
        await supabase.auth.signOut();
        setShowDropdown(false);
    };

    const toggleDropdown = () => {
        if (!showDropdown && buttonRef.current) {
            const rect = buttonRef.current.getBoundingClientRect();
            setDropdownPosition({
                top: rect.bottom + 8,
                right: window.innerWidth - rect.right,
            });
        }
        setShowDropdown(!showDropdown);
    };

    if (!user) {
        return (
            <>
                <button
                    onClick={() => setShowAuthModal(true)}
                    className="tactile-btn px-4 py-2 bg-[#E8B84B] hover:bg-[#F4DE9B] text-[#162A20] text-xs font-extrabold uppercase tracking-wider rounded-full cursor-pointer"
                >
                    Sign In
                </button>
                <AuthModal
                    isOpen={showAuthModal}
                    onClose={() => setShowAuthModal(false)}
                    onSuccess={() => setShowAuthModal(false)}
                />
            </>
        );
    }

    return (
        <>
            <button
                ref={buttonRef}
                onClick={toggleDropdown}
                className="tactile-btn flex items-center gap-2 px-3 py-1.5 bg-[#FBF8F1] rounded-full cursor-pointer"
            >
                <div className="w-7 h-7 bg-[#C84B31] border border-[#162A20] rounded-full flex items-center justify-center text-[#FBF8F1] text-xs font-extrabold">
                    {user.email?.[0].toUpperCase() || '?'}
                </div>
                <span className="text-[#162A20] text-xs font-bold max-w-[110px] truncate hidden sm:inline">
                    {user.email?.split('@')[0]}
                </span>
                <span className="text-[#162A20] text-[10px]">▼</span>
            </button>

            {showDropdown && (
                <>
                    <div
                        className="fixed inset-0"
                        style={{ zIndex: 9998 }}
                        onClick={() => setShowDropdown(false)}
                    />
                    <div
                        className="fixed w-56 bg-[#FBF8F1] border-2 border-[#162A20] tactile-shadow rounded-2xl overflow-hidden"
                        style={{
                            zIndex: 9999,
                            top: dropdownPosition.top,
                            right: dropdownPosition.right,
                        }}
                    >
                        <div className="px-4 py-3 border-b-2 border-[#162A20] bg-[#F7F3E8]">
                            <p className="text-[10px] font-mono-tech uppercase tracking-wider text-[#5A655E] font-bold">
                                Ranger Pass
                            </p>
                            <p className="text-xs font-bold text-[#162A20] truncate">{user.email}</p>
                        </div>
                        <button
                            onClick={handleSignOut}
                            className="w-full px-4 py-3 text-left text-xs font-extrabold uppercase tracking-wider text-[#C84B31] hover:bg-[#F4DE9B]/40 transition-colors flex items-center gap-2 cursor-pointer"
                        >
                            <span>🚪</span> Sign Out
                        </button>
                    </div>
                </>
            )}
        </>
    );
}
