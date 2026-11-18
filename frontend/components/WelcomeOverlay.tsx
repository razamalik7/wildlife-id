'use client';

import { useState, useEffect } from 'react';
import { Camera, Brain, Map, BookOpen } from 'lucide-react';

export default function WelcomeOverlay() {
    const [step, setStep] = useState(0);
    const [show, setShow] = useState(false);
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
        if (typeof window !== 'undefined' && window.location.search.includes('nowelcome=1')) {
            return;
        }
        const hasSeen = localStorage.getItem('hasSeenWelcome');
        if (!hasSeen) {
            setTimeout(() => setShow(true), 400);
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
            icon: <Camera size={48} className="text-[#C84B31]" />,
            tag: 'STEP 01 · FIELD CAPTURE',
            title: 'Snap Wild Fauna',
            desc: 'Photograph any North American mammal, bird, reptile, or amphibian you spot on the trail.',
        },
        {
            icon: <Brain size={48} className="text-[#162A20]" />,
            tag: 'STEP 02 · OOGWAY CV ENGINE',
            title: 'OOGWAY Identifies It',
            desc: 'Our custom dual-network OOGWAY vision architecture & live iNaturalist range check verify your sighting in seconds.',
        },
        {
            icon: <BookOpen size={48} className="text-[#C84B31]" />,
            tag: 'STEP 03 · 3-STAGE COLLECTOR DECK',
            title: 'Ink Your Collector Cards',
            desc: 'Turn un-inked woodcut ghosts into full-color Spot-Ink Collector Cards and flip them to view your Personal Polaroid Wall.',
        },
        {
            icon: <Map size={48} className="text-[#234733]" />,
            tag: 'STEP 04 · PARK & BIOME ATLAS',
            title: 'Stamp Parks & Ecosystems',
            desc: 'Search 248 National & State Parks, toggle North American ecosystem overlays, and map your sightings.',
        },
    ];

    return (
        <div className="fixed inset-0 z-[2000] flex items-center justify-center p-4 bg-[#162A20]/70 backdrop-blur-xs">
            <div className="bg-[#FBF8F1] rounded-2xl border-2 border-[#162A20] tactile-shadow-lg p-7 max-w-sm w-full text-center relative overflow-hidden">
                <button
                    onClick={handleClose}
                    className="absolute top-3 right-3 text-xs font-mono-tech font-bold text-[#5A655E] hover:text-[#162A20] px-2 py-1"
                >
                    SKIP ✕
                </button>

                <div className="flex flex-col items-center gap-4 min-h-[260px] justify-center pt-2">
                    <span className="font-mono-tech text-[10px] font-extrabold uppercase tracking-widest px-3 py-1 rounded-full bg-[#F4DE9B] border border-[#162A20] text-[#162A20]">
                        {slides[step].tag}
                    </span>
                    <div className="p-5 bg-[#F7F3E8] rounded-2xl border-2 border-[#162A20] tactile-shadow-sm">
                        {slides[step].icon}
                    </div>
                    <div className="space-y-1.5">
                        <h2 className="font-editorial text-3xl text-[#162A20]">{slides[step].title}</h2>
                        <p className="text-[#5A655E] text-sm leading-relaxed font-medium">
                            {slides[step].desc}
                        </p>
                    </div>
                </div>

                <div className="flex justify-center gap-2 my-5">
                    {slides.map((_, i) => (
                        <div
                            key={i}
                            className={`h-2.5 rounded-full border border-[#162A20] transition-all duration-300 ${
                                i === step ? 'w-8 bg-[#C84B31]' : 'w-2.5 bg-[#EAE3D2]'
                            }`}
                        />
                    ))}
                </div>

                <button
                    onClick={handleNext}
                    className="w-full py-3.5 bg-[#162A20] hover:bg-[#234733] text-[#F4DE9B] rounded-full font-extrabold text-xs uppercase tracking-widest tactile-btn cursor-pointer"
                >
                    {step === 3 ? 'Open Field Press →' : 'Next Page →'}
                </button>
            </div>
        </div>
    );
}
