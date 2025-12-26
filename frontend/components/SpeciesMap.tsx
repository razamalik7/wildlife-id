"use client";

import { MapContainer, TileLayer, Marker, Popup, LayersControl } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { useEffect, useState } from "react";

// Standard Blue Pin for "You are Here"
const currentLocIcon = L.icon({
  iconUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
});

interface Props {
  taxonId: number;
  userLat?: number;  // Optional - only show marker if provided
  userLng?: number;  // Optional - only show marker if provided
  animalName: string;
  observations?: Array<{ lat: number, lng: number, date: Date }>;
}

export default function SpeciesMap({ taxonId, userLat, userLng, animalName, observations = [] }: Props) {
  // Use observation location or default to center US for map center
  const centerLat = userLat ?? (observations[0]?.lat ?? 39.8);
  const centerLng = userLng ?? (observations[0]?.lng ?? -98.5);
  const [hasIUCNRange, setHasIUCNRange] = useState(false);

  // Check if IUCN range data exists for this taxon
  useEffect(() => {
    if (taxonId <= 0) return;

    setHasIUCNRange(false); // Reset on taxon change

    // Try to fetch a single tile to see if range data exists
    const checkIUCN = async () => {
      try {
        const response = await fetch(
          `https://api.inaturalist.org/v1/taxon_ranges/${taxonId}/5/8/11.png`,
          { method: 'HEAD' } // Just check if it exists, don't download
        );
        // If we get a 200 with content, range exists
        const contentLength = response.headers.get('content-length');
        setHasIUCNRange(response.ok && contentLength !== null && parseInt(contentLength) > 1000);
      } catch {
        setHasIUCNRange(false);
      }
    };

    checkIUCN();
  }, [taxonId]);
  // Dynamic icon for user observations
  const getAnimalIcon = () => {
    const slug = animalName.toLowerCase().replace(/ /g, "_");
    return L.icon({
      iconUrl: `/icons/${slug}.png`,
      iconSize: [40, 40],
      iconAnchor: [20, 20],
      className: "drop-shadow-lg opacity-50 grayscale",
    });
  };

  return (
    <div className="h-full w-full rounded-xl overflow-hidden border border-stone-700 shadow-2xl relative z-0">
      <MapContainer
        center={[centerLat, centerLng]}
        zoom={5}
        style={{ height: "100%", width: "100%" }}
        scrollWheelZoom={true}
      >
        <LayersControl position="topright">

          {/* BASE LAYERS */}
          <LayersControl.BaseLayer checked name="Dark">
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
              url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            />
          </LayersControl.BaseLayer>

          <LayersControl.BaseLayer name="Light">
            <TileLayer
              url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
              attribution='&copy; OSM'
            />
          </LayersControl.BaseLayer>

          <LayersControl.BaseLayer name="Satellite">
            <TileLayer
              url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
              attribution='Tiles &copy; Esri'
            />
          </LayersControl.BaseLayer>

          {/* SPECIES OVERLAY LAYERS */}
          {taxonId > 0 && (
            <>
              {/* Native Range - Green Grid */}
              <LayersControl.Overlay checked name="🟩 Native Range">
                <TileLayer
                  opacity={0.6}
                  url={`https://api.inaturalist.org/v1/grid/{z}/{x}/{y}.png?taxon_id=${taxonId}&color=%2316a34a&captive=false&quality_grade=research&native=true&verifiable=true`}
                />
              </LayersControl.Overlay>

              {/* Introduced Range - Red Grid */}
              <LayersControl.Overlay checked name="🟥 Introduced Range">
                <TileLayer
                  opacity={0.6}
                  url={`https://api.inaturalist.org/v1/grid/{z}/{x}/{y}.png?taxon_id=${taxonId}&color=%23dc2626&captive=false&quality_grade=research&native=false&verifiable=true`}
                />
              </LayersControl.Overlay>

              {/* Heatmap View */}
              <LayersControl.Overlay name="🔥 Heatmap">
                <TileLayer
                  opacity={0.7}
                  url={`https://api.inaturalist.org/v1/colored_heatmap/{z}/{x}/{y}.png?taxon_id=${taxonId}&color=%23f97316&captive=false&quality_grade=research&verifiable=true`}
                />
              </LayersControl.Overlay>

              {/* Exact Points */}
              <LayersControl.Overlay name="📍 Exact Sightings">
                <TileLayer
                  opacity={1.0}
                  url={`https://api.inaturalist.org/v1/points/{z}/{x}/{y}.png?taxon_id=${taxonId}&color=%23000000&captive=false&quality_grade=research&verifiable=true`}
                />
              </LayersControl.Overlay>

              {/* IUCN Range - Only show if data exists */}
              {hasIUCNRange && (
                <LayersControl.Overlay name="🗺️ IUCN Range">
                  <TileLayer
                    opacity={0.5}
                    url={`https://api.inaturalist.org/v1/taxon_ranges/${taxonId}/{z}/{x}/{y}.png`}
                  />
                </LayersControl.Overlay>
              )}
            </>
          )}

        </LayersControl>

        {/* Current Location - Only show if provided */}
        {userLat !== undefined && userLng !== undefined && (
          <Marker position={[userLat, userLng]} icon={currentLocIcon}>
            <Popup>You are here</Popup>
          </Marker>
        )}

        {/* Your Observations */}
        {observations.map((obs, i) => (
          <Marker key={i} position={[obs.lat, obs.lng]} icon={getAnimalIcon()}>
            <Popup>
              <div className="text-center">
                <strong>{animalName}</strong><br />
                Sighted: {obs.date.toLocaleDateString()}
              </div>
            </Popup>
          </Marker>
        ))}

      </MapContainer>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-black/80 p-3 rounded-lg text-xs text-white z-[1000] border border-stone-600 shadow-xl">
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <div className="w-4 h-3 bg-green-500/60 border border-green-700"></div>
            <span>Native Range</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-3 bg-red-500/60 border border-red-700"></div>
            <span>Introduced</span>
          </div>
          {userLat !== undefined && userLng !== undefined && (
            <div className="flex items-center gap-2">
              <img src="https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png" className="w-3 h-5" alt="pin" />
              <span>You</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}