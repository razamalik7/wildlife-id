"use client";

import { MapContainer, TileLayer, Marker, useMapEvents, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { useState, useEffect } from "react";
import { MapPin, Check, Search } from "lucide-react";
import { ParkRecord } from "@/lib/ecosystems";

const pinIcon = L.divIcon({
    className: "custom-pin-icon",
    html: `
    <div style="
      width: 42px;
      height: 42px;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
    ">
      <div style="
        position: absolute;
        width: 58px;
        height: 58px;
        background: rgba(200, 75, 49, 0.25);
        border-radius: 50%;
        animation: pulse 2s infinite;
      "></div>
      <div style="
        width: 22px;
        height: 22px;
        background: #C84B31;
        border: 3px solid #FBF8F1;
        outline: 2px solid #162A20;
        border-radius: 50%;
        box-shadow: 2px 2px 0px #162A20;
      "></div>
    </div>
    <style>
      @keyframes pulse {
        0% { transform: scale(0.8); opacity: 1; }
        50% { transform: scale(1.25); opacity: 0.45; }
        100% { transform: scale(0.8); opacity: 1; }
      }
    </style>
  `,
    iconSize: [42, 42],
    iconAnchor: [21, 21],
});

interface Props {
    initialLat?: number;
    initialLng?: number;
    photoPreview?: string;
    isConfirmed: boolean;
    onConfirm: (lat: number, lng: number) => void;
    onLocationChange?: () => void;
}

function MapController({ flyTarget }: { flyTarget: [number, number] | null }) {
    const map = useMap();
    useEffect(() => {
        setTimeout(() => map.invalidateSize(), 100);
    }, [map]);

    useEffect(() => {
        if (flyTarget) {
            map.flyTo(flyTarget, 10, { duration: 1.1 });
        }
    }, [flyTarget, map]);

    return null;
}

function DraggableMarker({
    position,
    setPosition,
    onMove,
}: {
    position: L.LatLng;
    setPosition: (pos: L.LatLng) => void;
    onMove: () => void;
}) {
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
    onLocationChange,
}: Props) {
    const defaultLat = initialLat ?? 39.8283;
    const defaultLng = initialLng ?? -98.5795;

    const [position, setPosition] = useState<L.LatLng>(
        new L.LatLng(defaultLat, defaultLng)
    );
    const [parks, setParks] = useState<ParkRecord[]>([]);
    const [parkQuery, setParkQuery] = useState("");
    const [flyTarget, setFlyTarget] = useState<[number, number] | null>(null);

    useEffect(() => {
        fetch("/parks.json")
            .then((r) => r.json())
            .then((data) => setParks(data))
            .catch(() => {});
    }, []);

    const matchingParks =
        parkQuery.trim().length > 1
            ? parks
                  .filter(
                      (p) =>
                          p.name.toLowerCase().includes(parkQuery.toLowerCase()) ||
                          p.state.toLowerCase().includes(parkQuery.toLowerCase())
                  )
                  .slice(0, 6)
            : [];

    const handlePinMove = () => {
        if (onLocationChange) {
            onLocationChange();
        }
    };

    const handleSelectPark = (park: ParkRecord) => {
        const nextPos = new L.LatLng(park.lat, park.lng);
        setPosition(nextPos);
        setFlyTarget([park.lat, park.lng]);
        setParkQuery("");
        handlePinMove();
    };

    const handleConfirm = () => {
        onConfirm(position.lat, position.lng);
    };

    return (
        <div className="h-full w-full flex flex-col bg-[#FBF8F1] relative border-2 border-[#162A20] rounded-2xl overflow-hidden tactile-shadow">
            {/* Top Ranger Bar + Quick Park Search */}
            <div className="bg-[#F7F3E8] border-b-2 border-[#162A20] p-3.5 z-20 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="flex items-center gap-3">
                    {photoPreview && (
                        <img
                            src={photoPreview}
                            alt="Field capture"
                            className="w-12 h-12 object-cover rounded-lg border-2 border-[#162A20] tactile-shadow-sm shrink-0"
                        />
                    )}
                    <div>
                        <div className="flex items-center gap-2">
                            <span
                                className={`px-2 py-0.5 rounded-full text-[10px] font-mono-tech font-extrabold uppercase border border-[#162A20] ${
                                    isConfirmed
                                        ? "bg-[#234733] text-[#F4DE9B]"
                                        : "bg-[#F4DE9B] text-[#162A20]"
                                }`}
                            >
                                {isConfirmed ? "✓ STAMP LOCKED" : "STEP 02 · PIN SIGHTING"}
                            </span>
                            <span className="font-mono-tech text-xs font-bold text-[#5A655E]">
                                {position.lat.toFixed(3)}°N, {Math.abs(position.lng).toFixed(3)}°W
                            </span>
                        </div>
                        <h2 className="font-editorial text-2xl text-[#162A20] leading-tight mt-0.5">
                            {isConfirmed
                                ? "Observation Coordinates Locked"
                                : "Where did you spot this animal?"}
                        </h2>
                    </div>
                </div>

                {/* Quick Park Search Box */}
                <div className="relative w-full sm:w-64">
                    <Search
                        size={14}
                        className="absolute left-3 top-1/2 -translate-y-1/2 text-[#5A655E]"
                    />
                    <input
                        type="text"
                        value={parkQuery}
                        onChange={(e) => setParkQuery(e.target.value)}
                        placeholder="Jump to a National/State Park..."
                        className="w-full pl-8 pr-3 py-1.5 bg-[#FBF8F1] border-2 border-[#162A20] rounded-full text-xs font-bold text-[#162A20] placeholder:text-[#5A655E] focus:outline-none"
                    />
                    {matchingParks.length > 0 && (
                        <div className="absolute left-0 right-0 top-full mt-1 bg-[#FBF8F1] border-2 border-[#162A20] rounded-xl tactile-shadow overflow-hidden z-50">
                            {matchingParks.map((p) => (
                                <button
                                    key={p.id}
                                    type="button"
                                    onClick={() => handleSelectPark(p)}
                                    className="w-full text-left px-3 py-2 text-xs font-bold text-[#162A20] hover:bg-[#F4DE9B] border-b border-[#162A20]/10 last:border-none flex items-center justify-between cursor-pointer"
                                >
                                    <span>🏞️ {p.name}</span>
                                    <span className="font-mono-tech text-[10px] text-[#5A655E]">
                                        {p.state}
                                    </span>
                                </button>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            {/* Leaflet Map */}
            <div className="flex-1 relative z-0">
                <MapContainer
                    center={[defaultLat, defaultLng]}
                    zoom={initialLat ? 9 : 4}
                    style={{ height: "100%", width: "100%" }}
                    scrollWheelZoom={true}
                >
                    <MapController flyTarget={flyTarget} />
                    <TileLayer
                        attribution='Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ, TomTom, Intermap, iPC, USGS, FAO, NPS, NRCAN, GeoBase'
                        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
                    />
                    <DraggableMarker
                        position={position}
                        setPosition={setPosition}
                        onMove={handlePinMove}
                    />
                </MapContainer>
            </div>

            {/* Confirm Button Overlay */}
            {!isConfirmed && (
                <div className="absolute bottom-4 left-4 right-4 z-10 flex justify-center pointer-events-none">
                    <button
                        type="button"
                        onClick={handleConfirm}
                        className="pointer-events-auto px-6 py-3 bg-[#234733] hover:bg-[#162A20] text-[#F4DE9B] rounded-full font-extrabold text-xs uppercase tracking-widest tactile-btn flex items-center gap-2 cursor-pointer"
                    >
                        <Check className="w-4 h-4" />
                        <span>Lock Sighting Location ({position.lat.toFixed(2)}°, {position.lng.toFixed(2)}°)</span>
                    </button>
                </div>
            )}
        </div>
    );
}
