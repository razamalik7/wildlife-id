'use client';

import { useState, useEffect } from 'react';
import { Camera, Brain, Map, PawPrint } from 'lucide-react';

export default function WelcomeOverlay() {
    const [step, setStep] = useState(0);
    const [show, setShow] = useState(false);
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
        const hasSeen = localStorage.getItem('hasSeenWelcome');
        if (!hasSeen) {
            setTimeout(() => setShow(true), 500); // Small delay for smooth entrance
        }
    }, []);

    const handleNext = () => {
        if (step < 3) {
            setStep(step + 1);
        } else {
            handleClose();
        }
    };

    const handleClose = () => {
        localStorage.setItem('hasSeenWelcome', 'true');
        setShow(false);
    };

    if (!mounted || !show) return null;

    const slides = [
        {
            icon: <Camera size={64} className="text-emerald-500" />,
            title: "Snap a Photo",
            desc: "Take a picture of any North American wildlife you spot."
        },
        {
            icon: <Brain size={64} className="text-violet-500" />,
            title: "AI Identifies It",
            desc: "Our advanced AI identifies the species in seconds."
        },
        {
            icon: <PawPrint size={64} className="text-amber-500" />,
            title: "Build Your AniDex",
            desc: "Track your findings and try to collect all 100 species."
        },
        {
            icon: <Map size={64} className="text-blue-500" />,
            title: "See Your World",
            desc: " visualize your discoveries on an interactive map."
        }
    ];

    return (
        <div className="fixed inset-0 z-[2000] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-300">
            <div className="bg-white rounded-3xl shadow-2xl p-8 max-w-sm w-full text-center relative overflow-hidden animate-in zoom-in-95 duration-300">

                {/* Background Decor */}
                <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-emerald-400 via-violet-400 to-amber-400" />

                {/* Content */}
                <div className="flex flex-col items-center gap-6 min-h-[300px] justify-center">
                    <div className="p-6 bg-stone-50 rounded-full shadow-inner animate-in zoom-in duration-500 delay-100">
                        {slides[step].icon}
                    </div>

                    <div className="space-y-2 animate-in slide-in-from-bottom-4 duration-500">
                        <h2 className="text-2xl font-black text-stone-800">{slides[step].title}</h2>
                        <p className="text-stone-500 leading-relaxed font-medium">
                            {slides[step].desc}
                        </p>
                    </div>
                </div>

                {/* Progress Indicators */}
                <div className="flex justify-center gap-2 mb-8">
                    {slides.map((_, i) => (
                        <div
                            key={i}
                            className={`h-2 rounded-full transition-all duration-300 ${i === step ? 'w-8 bg-black' : 'w-2 bg-stone-200'
                                }`}
                        />
                    ))}
                </div>

                {/* Action Button */}
                <button
                    onClick={handleNext}
                    className="w-full py-4 bg-black text-white rounded-xl font-bold text-lg hover:scale-105 active:scale-95 transition-all shadow-lg"
                >
                    {step === 3 ? "Start Exploring" : "Next"}
                </button>

            </div>
        </div>
    );
}
