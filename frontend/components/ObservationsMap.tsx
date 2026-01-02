'use client';

import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { useEffect, useState } from 'react';
import L from 'leaflet';
import { Loader2 } from 'lucide-react';

// Fix Leaflet Default Icon
const icon = L.icon({
    iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
    iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
    shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
});

export default function ObservationsMap({ observations }: { observations: any[] }) {
    if (!observations) return null;

    // Center map on the most recent observation, or default to US
    const center = observations.length > 0
        ? [observations[0].lat, observations[0].lng]
        : [39.8283, -98.5795]; // US Center

    return (
        <MapContainer
            center={center as [number, number]}
            zoom={observations.length > 0 ? 5 : 4}
            style={{ height: '100%', width: '100%' }}
            className="z-0"
        >
            <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
                url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
            />

            {observations.map((obs) => (
                <Marker
                    key={obs.id}
                    position={[obs.lat, obs.lng]}
                    icon={icon}
                >
                    <Popup className="custom-popup">
                        <div className="flex flex-col items-center gap-2 min-w-[150px]">
                            <div className="w-full h-24 bg-stone-100 rounded-md overflow-hidden relative">
                                <img
                                    src={obs.image_url}
                                    alt={obs.common_name}
                                    className="w-full h-full object-cover"
                                />
                            </div>
                            <div className="text-center">
                                <h3 className="font-bold text-sm text-stone-800">{obs.common_name}</h3>
                                <p className="text-xs text-stone-500 font-mono italic">{obs.scientific_name}</p>
                                <p className="text-[10px] text-stone-400 mt-1">
                                    {new Date(obs.created_at).toLocaleDateString()}
                                </p>
                            </div>
                        </div>
                    </Popup>
                </Marker>
            ))}
        </MapContainer>
    );
}
