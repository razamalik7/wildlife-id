'use client'

import { useState, useEffect, useRef } from 'react'
import { supabase } from '@/lib/supabase'
import { User } from '@supabase/supabase-js'
import AuthModal from './AuthModal'

export default function UserHeader() {
    const [user, setUser] = useState<User | null>(null)
    const [showAuthModal, setShowAuthModal] = useState(false)
    const [showDropdown, setShowDropdown] = useState(false)
    const buttonRef = useRef<HTMLButtonElement>(null)
    const [dropdownPosition, setDropdownPosition] = useState({ top: 0, right: 0 })

    useEffect(() => {
        supabase.auth.getSession().then(({ data: { session } }) => {
            setUser(session?.user ?? null)
        })

        const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
            setUser(session?.user ?? null)
        })

        return () => subscription.unsubscribe()
    }, [])

    const handleSignOut = async () => {
        await supabase.auth.signOut()
        setShowDropdown(false)
    }

    const toggleDropdown = () => {
        if (!showDropdown && buttonRef.current) {
            const rect = buttonRef.current.getBoundingClientRect()
            setDropdownPosition({
                top: rect.bottom + 8,
                right: window.innerWidth - rect.right
            })
        }
        setShowDropdown(!showDropdown)
    }

    if (!user) {
        return (
            <>
                <button
                    onClick={() => setShowAuthModal(true)}
                    className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-emerald-500 to-teal-600 text-white font-semibold rounded-xl hover:from-emerald-600 hover:to-teal-700 transition-all shadow-lg"
                >
                    Sign In
                </button>
                <AuthModal
                    isOpen={showAuthModal}
                    onClose={() => setShowAuthModal(false)}
                    onSuccess={() => setShowAuthModal(false)}
                />
            </>
        )
    }

    return (
        <>
            <button
                ref={buttonRef}
                onClick={toggleDropdown}
                className="flex items-center gap-2 px-4 py-2 bg-stone-800 border border-stone-600 rounded-xl hover:bg-stone-700 transition-all"
            >
                <div className="w-8 h-8 bg-gradient-to-br from-emerald-400 to-teal-500 rounded-full flex items-center justify-center text-white font-bold">
                    {user.email?.[0].toUpperCase() || '?'}
                </div>
                <span className="text-stone-200 max-w-[120px] truncate">
                    {user.email?.split('@')[0]}
                </span>
                <span className="text-stone-400">▼</span>
            </button>

            {showDropdown && (
                <>
                    {/* Backdrop */}
                    <div
                        className="fixed inset-0"
                        style={{ zIndex: 9998 }}
                        onClick={() => setShowDropdown(false)}
                    />
                    {/* Dropdown - Fixed position to escape stacking context */}
                    <div
                        className="fixed w-56 bg-stone-800 border border-stone-600 rounded-xl shadow-2xl overflow-hidden"
                        style={{
                            zIndex: 9999,
                            top: dropdownPosition.top,
                            right: dropdownPosition.right
                        }}
                    >
                        <div className="px-4 py-3 border-b border-stone-600">
                            <p className="text-xs text-stone-400">Signed in as</p>
                            <p className="text-sm text-white truncate">{user.email}</p>
                        </div>
                        <button
                            onClick={handleSignOut}
                            className="w-full px-4 py-3 text-left text-red-400 hover:bg-stone-700 transition-colors flex items-center gap-2"
                        >
                            <span>🚪</span> Sign Out
                        </button>
                    </div>
                </>
            )}
        </>
    )
}
