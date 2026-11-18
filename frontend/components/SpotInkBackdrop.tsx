'use client';

import { EcosystemId } from '@/lib/ecosystems';

interface SpotInkBackdropProps {
    ecosystemId: EcosystemId;
    className?: string;
    variant?: 'card' | 'cover';
}

/**
 * Renders the bespoke 2-tone geometric National Park / Spot-Ink SVG habitat backdrop
 * (Option 27 / A3 geometry in Option 32's Spruce #162A20, Rust #C84B31, and Sunlit Gold #E8B84B palette).
 */
export default function SpotInkBackdrop({
    ecosystemId,
    className = 'w-full h-full',
    variant = 'card',
}: SpotInkBackdropProps) {
    if (ecosystemId === 'coastal') {
        return (
            <svg viewBox="0 0 300 200" preserveAspectRatio="none" className={className}>
                <rect width="300" height="200" fill="#DDEAEF" />
                {/* Sunlit Goldenrod Orb */}
                <circle cx="220" cy="68" r={variant === 'cover' ? 54 : 44} fill="#E8B84B" />
                <circle cx="220" cy="68" r={variant === 'cover' ? 66 : 54} fill="none" stroke="#E8B84B" strokeWidth="1.5" strokeDasharray="4 4" />
                {/* Distant Headland */}
                <path d="M0 135 L95 88 L185 135 Z" fill="#235365" opacity="0.85" />
                <path d="M140 140 L240 96 L300 135 L300 200 L0 200 Z" fill="#162A20" />
                {/* Layered Tidewater Waves */}
                <path d="M0 152 Q38 142 75 152 T150 152 T225 152 T300 152 L300 200 L0 200 Z" fill="#235365" />
                <path d="M0 168 Q38 160 75 168 T150 168 T225 168 T300 168" fill="none" stroke="#FBF8F1" strokeWidth="2.5" opacity="0.7" />
                <path d="M0 184 Q38 176 75 184 T150 184 T225 184 T300 184" fill="none" stroke="#E8B84B" strokeWidth="2" opacity="0.75" />
            </svg>
        );
    }

    if (ecosystemId === 'alpine') {
        return (
            <svg viewBox="0 0 300 200" preserveAspectRatio="none" className={className}>
                <rect width="300" height="200" fill="#EDE6DA" />
                {/* Pale Gold + Rust Sun */}
                <circle cx="150" cy="70" r={variant === 'cover' ? 56 : 46} fill="#F4DE9B" />
                <circle cx="150" cy="70" r={variant === 'cover' ? 40 : 32} fill="#C84B31" opacity="0.88" />
                {/* Sharp High-Altitude Crags */}
                <polygon points="0,175 78,62 158,175" fill="#523F2D" />
                <polygon points="78,62 112,175 158,175" fill="#162A20" opacity="0.45" />
                <polygon points="115,180 212,52 300,180" fill="#1E4631" />
                <polygon points="212,52 248,180 300,180" fill="#162A20" opacity="0.6" />
                {/* Foreground Ridge & Contour Lines */}
                <path d="M0 164 Q150 138 300 168 L300 200 L0 200 Z" fill="#162A20" />
                <path d="M0 182 Q150 156 300 184" fill="none" stroke="#E8B84B" strokeWidth="1.5" strokeDasharray="5 4" opacity="0.65" />
            </svg>
        );
    }

    if (ecosystemId === 'wetland') {
        return (
            <svg viewBox="0 0 300 200" preserveAspectRatio="none" className={className}>
                <rect width="300" height="200" fill="#E3EFE7" />
                {/* Warm Golden Sun */}
                <circle cx="82" cy="66" r={variant === 'cover' ? 50 : 42} fill="#E8B84B" />
                {/* Cypress / Marsh Horizon */}
                <path d="M0 132 Q85 102 175 130 T300 124 L300 200 L0 200 Z" fill="#2D5A3C" />
                <path d="M0 152 Q110 138 210 154 T300 148 L300 200 L0 200 Z" fill="#162A20" />
                {/* Cattail / Reed Motifs */}
                <line x1="36" y1="165" x2="36" y2="102" stroke="#162A20" strokeWidth="2.5" />
                <rect x="33" y="108" width="6" height="18" rx="3" fill="#C84B31" />
                <line x1="262" y1="168" x2="262" y2="106" stroke="#162A20" strokeWidth="2.5" />
                <rect x="259" y="112" width="6" height="18" rx="3" fill="#C84B31" />
                {/* Water Ripples */}
                <line x1="55" y1="172" x2="185" y2="172" stroke="#E8B84B" strokeWidth="2" strokeLinecap="round" opacity="0.8" />
                <line x1="95" y1="184" x2="235" y2="184" stroke="#FBF8F1" strokeWidth="1.5" strokeLinecap="round" opacity="0.65" />
            </svg>
        );
    }

    if (ecosystemId === 'meadow') {
        return (
            <svg viewBox="0 0 300 200" preserveAspectRatio="none" className={className}>
                <rect width="300" height="200" fill="#F5ECD7" />
                {/* Radiant Sunburst */}
                <circle cx="150" cy="74" r={variant === 'cover' ? 60 : 48} fill="#E8B84B" />
                <circle cx="150" cy="74" r={variant === 'cover' ? 74 : 60} fill="none" stroke="#C84B31" strokeWidth="1.5" strokeDasharray="4 4" opacity="0.6" />
                {/* Sweeping Prairie Hills */}
                <path d="M0 136 Q95 96 205 132 T300 122 L300 200 L0 200 Z" fill="#C84B31" opacity="0.85" />
                <path d="M0 152 Q140 120 300 150 L300 200 L0 200 Z" fill="#234733" />
                <path d="M0 170 Q160 148 300 172 L300 200 L0 200 Z" fill="#162A20" />
                <path d="M0 184 Q150 165 300 185" fill="none" stroke="#F4DE9B" strokeWidth="1.5" strokeDasharray="6 4" opacity="0.7" />
            </svg>
        );
    }

    if (ecosystemId === 'arid') {
        return (
            <svg viewBox="0 0 300 200" preserveAspectRatio="none" className={className}>
                <rect width="300" height="200" fill="#F5E2DC" />
                {/* Desert Sun */}
                <circle cx="210" cy="64" r={variant === 'cover' ? 52 : 44} fill="#E8B84B" />
                {/* Red-Rock Butte & Canyon Layers */}
                <polygon points="18,155 42,92 118,92 142,155" fill="#C84B31" />
                <polygon points="155,158 178,108 248,108 272,158" fill="#9E3822" />
                <path d="M0 148 L110 134 L215 148 L300 136 L300 200 L0 200 Z" fill="#523F2D" />
                <path d="M0 166 Q150 152 300 168 L300 200 L0 200 Z" fill="#162A20" />
                <path d="M0 183 Q150 170 300 184" fill="none" stroke="#E8B84B" strokeWidth="1.5" strokeDasharray="5 4" opacity="0.75" />
            </svg>
        );
    }

    // Default: Temperate Forest ('forest')
    return (
        <svg viewBox="0 0 300 200" preserveAspectRatio="none" className={className}>
            <rect width="300" height="200" fill="#E8E2D1" />
            {/* Terracotta & Sunlit Gold Rising Sun */}
            <circle cx="150" cy="72" r={variant === 'cover' ? 56 : 46} fill="#E8B84B" />
            <circle cx="150" cy="72" r={variant === 'cover' ? 68 : 56} fill="none" stroke="#C84B31" strokeWidth="1.5" strokeDasharray="4 4" opacity="0.65" />
            {/* Layered Appalachian / Forest Ridges */}
            <polygon points="0,155 75,86 165,155" fill="#C84B31" opacity="0.88" />
            <polygon points="110,158 212,78 300,158" fill="#234733" />
            {/* Conifer Silhouettes & Deep Spruce Foreground */}
            <path d="M0 146 Q150 116 300 146 L300 200 L0 200 Z" fill="#162A20" />
            <path d="M0 172 Q150 144 300 172" fill="none" stroke="#E8B84B" strokeWidth="1.5" strokeDasharray="5 4" opacity="0.55" />
        </svg>
    );
}
