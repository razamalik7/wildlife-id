'use client';

import { useState } from 'react';
import { supabase } from '@/lib/supabase';

interface AuthModalProps {
    isOpen: boolean;
    onClose: () => void;
    onSuccess: () => void;
}

export default function AuthModal({ isOpen, onClose, onSuccess }: AuthModalProps) {
    const [isLogin, setIsLogin] = useState(true);
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [message, setMessage] = useState<string | null>(null);

    if (!isOpen) return null;

    const handleAuth = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setMessage(null);

        try {
            if (isLogin) {
                const { error } = await supabase.auth.signInWithPassword({
                    email,
                    password,
                });
                if (error) throw error;
                onSuccess();
                onClose();
            } else {
                const { error } = await supabase.auth.signUp({
                    email,
                    password,
                });
                if (error) throw error;
                setMessage('Check your email for confirmation link!');
            }
        } catch (err: unknown) {
            setError(err instanceof Error ? err.message : 'An error occurred');
        } finally {
            setLoading(false);
        }
    };

    const handleGoogleLogin = async () => {
        const { error } = await supabase.auth.signInWithOAuth({
            provider: 'google',
            options: {
                redirectTo: window.location.origin,
            },
        });
        if (error) setError(error.message);
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#162A20]/70 backdrop-blur-xs p-4">
            <div className="bg-[#FBF8F1] rounded-2xl p-7 w-full max-w-md border-2 border-[#162A20] tactile-shadow-lg text-[#162A20]">
                <div className="flex justify-between items-center mb-5 pb-3 border-b-2 border-[#162A20]/15">
                    <div>
                        <span className="font-mono-tech text-[10px] font-bold uppercase tracking-widest text-[#C84B31] block">
                            RANGER FIELD PASS
                        </span>
                        <h2 className="font-editorial text-3xl text-[#162A20] leading-none mt-0.5">
                            {isLogin ? 'Welcome Back' : 'Issue Field Pass'}
                        </h2>
                    </div>
                    <button
                        onClick={onClose}
                        className="w-8 h-8 rounded-full border-2 border-[#162A20] bg-[#F7F3E8] hover:bg-[#F4DE9B] font-bold flex items-center justify-center cursor-pointer"
                    >
                        ✕
                    </button>
                </div>

                {error && (
                    <div className="bg-[#C84B31]/15 border-2 border-[#C84B31] text-[#C84B31] text-xs font-bold px-4 py-2.5 rounded-xl mb-4">
                        {error}
                    </div>
                )}

                {message && (
                    <div className="bg-[#234733]/15 border-2 border-[#234733] text-[#162A20] text-xs font-bold px-4 py-2.5 rounded-xl mb-4">
                        {message}
                    </div>
                )}

                <form onSubmit={handleAuth} className="space-y-4">
                    <div>
                        <label className="block text-[#162A20] text-xs font-extrabold uppercase tracking-wider mb-1.5">
                            Email
                        </label>
                        <input
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            className="w-full px-4 py-2.5 bg-[#F7F3E8] border-2 border-[#162A20] rounded-xl text-[#162A20] text-sm font-semibold focus:outline-none focus:bg-white"
                            placeholder="ranger@fieldpress.org"
                            required
                        />
                    </div>

                    <div>
                        <label className="block text-[#162A20] text-xs font-extrabold uppercase tracking-wider mb-1.5">
                            Password
                        </label>
                        <input
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            className="w-full px-4 py-2.5 bg-[#F7F3E8] border-2 border-[#162A20] rounded-xl text-[#162A20] text-sm font-semibold focus:outline-none focus:bg-white"
                            placeholder="••••••••"
                            required
                            minLength={6}
                        />
                    </div>

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full py-3 bg-[#C84B31] hover:bg-[#b23f27] text-[#FBF8F1] font-extrabold text-xs uppercase tracking-wider rounded-full tactile-btn disabled:opacity-50 cursor-pointer"
                    >
                        {loading ? 'Stamping Pass...' : isLogin ? 'Sign In to Atlas' : 'Create Field Pass'}
                    </button>
                </form>

                <div className="relative my-5">
                    <div className="absolute inset-0 flex items-center">
                        <div className="w-full border-t border-[#162A20]/20"></div>
                    </div>
                    <div className="relative flex justify-center text-xs">
                        <span className="px-3 bg-[#FBF8F1] text-[#5A655E] font-mono-tech uppercase">or</span>
                    </div>
                </div>

                <button
                    onClick={handleGoogleLogin}
                    className="w-full py-2.5 bg-[#F7F3E8] hover:bg-[#F4DE9B] text-[#162A20] font-extrabold text-xs uppercase tracking-wider rounded-full tactile-btn flex items-center justify-center gap-2 cursor-pointer"
                >
                    <svg className="w-4 h-4" viewBox="0 0 24 24">
                        <path
                            fill="#4285F4"
                            d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                        />
                        <path
                            fill="#34A853"
                            d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                        />
                        <path
                            fill="#FBBC05"
                            d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                        />
                        <path
                            fill="#EA4335"
                            d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                        />
                    </svg>
                    Continue with Google
                </button>

                <p className="text-center text-xs text-[#5A655E] mt-5">
                    {isLogin ? "Don't have a Field Pass yet? " : 'Already have a Field Pass? '}
                    <button
                        onClick={() => setIsLogin(!isLogin)}
                        className="text-[#C84B31] font-extrabold underline cursor-pointer"
                    >
                        {isLogin ? 'Sign Up' : 'Sign In'}
                    </button>
                </p>
            </div>
        </div>
    );
}
