"use client";

import { MapContainer, TileLayer, Marker, useMapEvents } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { useState } from "react";

// Use standard icons from a CDN hosting service
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// This invisible component listens for clicks on the map
function MapClickHandler({ onLocationSelect }: { onLocationSelect: (lat: number, lng: number) => void }) {
  useMapEvents({
    click(e) {
      onLocationSelect(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
}

export default function LocationMap({ onLocationSelect }: { onLocationSelect: (lat: number, lng: number) => void }) {
  const [position, setPosition] = useState<{ lat: number; lng: number } | null>(null);

  const handleSelect = (lat: number, lng: number) => {
    setPosition({ lat, lng });
    onLocationSelect(lat, lng);
  };

  return (
    <div className="h-64 w-full rounded-xl overflow-hidden border border-gray-300 relative z-0">
      <MapContainer 
        center={[40.7128, -74.0060]} // Default: New York
        zoom={2} 
        style={{ height: "100%", width: "100%" }}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; OpenStreetMap contributors'
        />
        
        {/* The Click Listener */}
        <MapClickHandler onLocationSelect={handleSelect} />

        {/* Show marker if user clicked */}
        {position && <Marker position={[position.lat, position.lng]} />} 
      </MapContainer>

      {/* Helper Text Overlay */}
      {!position && (
        <div className="absolute bottom-2 left-2 bg-white/80 px-2 py-1 text-xs rounded z-[1000] pointer-events-none">
          Click map to set location
        </div>
      )}
    </div>
  );
}