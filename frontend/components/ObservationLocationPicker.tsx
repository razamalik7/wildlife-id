"use client";

import { MapContainer, TileLayer, Marker, useMapEvents, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { useState, useEffect, useRef } from "react";
import { MapPin, Check } from "lucide-react";

// Custom pin icon with a modern look
const pinIcon = L.divIcon({
    className: "custom-pin-icon",
    html: `
    <div style="
      width: 40px;
      height: 40px;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
    ">
      <div style="
        position: absolute;
        width: 60px;
        height: 60px;
        background: rgba(16, 185, 129, 0.2);
        border-radius: 50%;
        animation: pulse 2s infinite;
      "></div>
      <div style="
        width: 20px;
        height: 20px;
        background: #10b981;
        border: 3px solid white;
        border-radius: 50%;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      "></div>
    </div>
    <style>
      @keyframes pulse {
        0% { transform: scale(0.8); opacity: 1; }
        50% { transform: scale(1.2); opacity: 0.5; }
        100% { transform: scale(0.8); opacity: 1; }
      }
    </style>
  `,
    iconSize: [40, 40],
    iconAnchor: [20, 20],
});

interface Props {
    initialLat?: number;
    initialLng?: number;
    photoPreview?: string;
    isConfirmed: boolean;  // Controlled by parent
    onConfirm: (lat: number, lng: number) => void;
    onLocationChange?: () => void;  // Called when pin moves (invalidates confirmation)
}

// Fix gray map bug
function MapResizer() {
    const map = useMap();
    useEffect(() => {
        setTimeout(() => map.invalidateSize(), 100);
    }, [map]);
    return null;
}

// Draggable marker handler
function DraggableMarker({
    position,
    setPosition,
    onMove
}: {
    position: L.LatLng;
    setPosition: (pos: L.LatLng) => void;
    onMove: () => void;
}) {
    // Also update on map click
    useMapEvents({
        click(e) {
            setPosition(e.latlng);
            onMove();
        },
    });

    return (
        <Marker
            position={position}
            icon={pinIcon}
            draggable={true}
            eventHandlers={{
                dragend: (e) => {
                    const marker = e.target;
                    if (marker) {
                        setPosition(marker.getLatLng());
                        onMove();
                    }
                },
            }}
        />
    );
}

export default function ObservationLocationPicker({
    initialLat,
    initialLng,
    photoPreview,
    isConfirmed,
    onConfirm,
    onLocationChange
}: Props) {
    // Default to center of US if no initial location
    const defaultLat = initialLat ?? 39.8283;
    const defaultLng = initialLng ?? -98.5795;

    const [position, setPosition] = useState<L.LatLng>(
        new L.LatLng(defaultLat, defaultLng)
    );

    const handlePinMove = () => {
        // Notify parent that pin moved (invalidates confirmation)
        if (onLocationChange) {
            onLocationChange();
        }
    };

    const handleConfirm = () => {
        onConfirm(position.lat, position.lng);
    };

    return (
        <div className="h-full w-full flex flex-col bg-stone-900 relative">
            {/* Header */}
            <div className="absolute top-0 left-0 right-0 z-[1000] bg-gradient-to-b from-stone-900/95 via-stone-900/80 to-transparent p-4 pb-8">
                <div className="flex items-center gap-3">
                    <div className={`${isConfirmed ? 'bg-emerald-500' : 'bg-amber-500'} p-2 rounded-xl shadow-lg transition-colors`}>
                        {isConfirmed ? <Check className="w-5 h-5 text-white" /> : <MapPin className="w-5 h-5 text-white" />}
                    </div>
                    <div>
                        <h2 className="text-white font-bold text-lg">
                            {isConfirmed ? 'Location Confirmed!' : 'Where did you spot this?'}
                        </h2>
                        <p className="text-stone-400 text-xs">
                            {isConfirmed
                                ? `${position.lat.toFixed(4)}, ${position.lng.toFixed(4)}`
                                : 'Drag the pin or tap the map • Approximate is fine!'}
                        </p>
                    </div>
                </div>
            </div>

            {/* Photo Preview Badge */}
            {photoPreview && (
                <div className="absolute top-20 left-4 z-[1000] bg-white/10 backdrop-blur-lg rounded-xl p-1.5 border border-white/20 shadow-xl">
                    <img
                        src={photoPreview}
                        alt="Your photo"
                        className="w-16 h-16 object-cover rounded-lg"
                    />
                </div>
            )}

            {/* Map */}
            <div className="flex-1 relative z-0">
                <MapContainer
                    center={[defaultLat, defaultLng]}
                    zoom={initialLat ? 10 : 4}
                    style={{ height: "100%", width: "100%" }}
                    scrollWheelZoom={true}
                >
                    <MapResizer />
                    <TileLayer
                        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
                        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                    />
                    <DraggableMarker
                        position={position}
                        setPosition={setPosition}
                        onMove={handlePinMove}
                    />
                </MapContainer>
            </div>

            {/* Confirm Button - Only show if NOT confirmed */}
            {!isConfirmed && (
                <div className="absolute bottom-6 left-4 right-4 z-[1000]">
                    <button
                        onClick={handleConfirm}
                        className="w-full py-4 bg-emerald-600 hover:bg-emerald-500 text-white rounded-2xl font-bold text-lg shadow-2xl flex justify-center items-center gap-3 transition-all active:scale-95 border border-emerald-400/30"
                    >
                        <Check className="w-5 h-5" />
                        Confirm Location
                    </button>
                    <p className="text-center text-stone-500 text-xs mt-2">
                        {position.lat.toFixed(4)}, {position.lng.toFixed(4)}
                    </p>
                </div>
            )}
        </div>
    );
}
