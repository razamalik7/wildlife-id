"use client";

import { MapContainer, TileLayer, Marker, useMapEvents, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { useState, useEffect } from "react";

const icon = L.icon({
  iconUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
});

interface Props {
  onLocationSelect: (lat: number, lng: number) => void;
}

// Helper to fix the "Gray Map" bug
function MapResizer() {
  const map = useMap();
  useEffect(() => {
    // Wait 100ms for modal to open, then force map to fit
    setTimeout(() => {
      map.invalidateSize();
    }, 100);
  }, [map]);
  return null;
}

function ClickHandler({ onLocationSelect }: Props) {
  const [position, setPosition] = useState<L.LatLng | null>(null);
  useMapEvents({
    click(e) {
      setPosition(e.latlng);
      onLocationSelect(e.latlng.lat, e.latlng.lng);
    },
  });
  return position ? <Marker position={position} icon={icon} /> : null;
}

export default function LocationPicker({ onLocationSelect }: Props) {
  return (
    <div className="h-full w-full relative z-0"> 
      <MapContainer 
        center={[39.8283, -98.5795]} 
        zoom={4} 
        style={{ height: "100%", width: "100%" }}
      >
        <MapResizer /> {/* Keeps the map form going gray */}
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <ClickHandler onLocationSelect={onLocationSelect} />
      </MapContainer>
    </div>
  );
}