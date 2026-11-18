"use client";

import { MapContainer, TileLayer, Marker, Popup, LayersControl } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { useEffect, useState } from "react";
import { speciesIconUrl } from "@/lib/species";

const currentLocIcon = L.icon({
  iconUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
});

interface Props {
  taxonId: number;
  userLat?: number;
  userLng?: number;
  animalName: string;
  observations?: Array<{ lat: number; lng: number; date: Date }>;
}

export default function SpeciesMap({
  taxonId,
  userLat,
  userLng,
  animalName,
  observations = [],
}: Props) {
  const centerLat = observations[0]?.lat ?? userLat ?? 39.8;
  const centerLng = observations[0]?.lng ?? userLng ?? -98.5;
  const [hasIUCNRange, setHasIUCNRange] = useState(false);

  useEffect(() => {
    if (taxonId <= 0) return;

    setHasIUCNRange(false);

    const checkIUCN = async () => {
      try {
        const response = await fetch(
          `https://api.inaturalist.org/v1/taxon_ranges/${taxonId}/5/8/11.png`,
          { method: "HEAD" }
        );
        const contentLength = response.headers.get("content-length");
        setHasIUCNRange(
          response.ok && contentLength !== null && parseInt(contentLength) > 1000
        );
      } catch {
        setHasIUCNRange(false);
      }
    };

    checkIUCN();
  }, [taxonId]);

  const getAnimalIcon = () => {
    return L.icon({
      iconUrl: speciesIconUrl(animalName),
      iconSize: [50, 50],
      iconAnchor: [25, 25],
      className: "cutout-animal-art",
    });
  };

  return (
    <div className="h-full w-full overflow-hidden relative z-0">
      <MapContainer
        center={[centerLat, centerLng]}
        zoom={4}
        style={{ height: "100%", width: "100%" }}
        scrollWheelZoom={true}
      >
        <LayersControl position="topright">
          <LayersControl.BaseLayer checked name="Topographic Atlas">
            <TileLayer
              url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
              attribution="Tiles &copy; Esri, USGS, NPS"
            />
          </LayersControl.BaseLayer>

          <LayersControl.BaseLayer name="OpenStreetMap">
            <TileLayer
              url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution="&copy; OpenStreetMap contributors"
            />
          </LayersControl.BaseLayer>

          <LayersControl.BaseLayer name="Satellite">
            <TileLayer
              url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
              attribution="Tiles &copy; Esri"
            />
          </LayersControl.BaseLayer>

          {taxonId > 0 && (
            <>
              <LayersControl.Overlay checked name="🟩 Native Range">
                <TileLayer
                  opacity={0.6}
                  url={`https://api.inaturalist.org/v1/grid/{z}/{x}/{y}.png?taxon_id=${taxonId}&color=%23234733&captive=false&quality_grade=research&native=true&verifiable=true`}
                />
              </LayersControl.Overlay>

              <LayersControl.Overlay checked name="🟥 Introduced Range">
                <TileLayer
                  opacity={0.6}
                  url={`https://api.inaturalist.org/v1/grid/{z}/{x}/{y}.png?taxon_id=${taxonId}&color=%23C84B31&captive=false&quality_grade=research&native=false&verifiable=true`}
                />
              </LayersControl.Overlay>

              <LayersControl.Overlay name="🔥 Density Heatmap">
                <TileLayer
                  opacity={0.7}
                  url={`https://api.inaturalist.org/v1/colored_heatmap/{z}/{x}/{y}.png?taxon_id=${taxonId}&color=%23E8B84B&captive=false&quality_grade=research&verifiable=true`}
                />
              </LayersControl.Overlay>

              <LayersControl.Overlay name="📍 Exact Sightings">
                <TileLayer
                  opacity={1.0}
                  url={`https://api.inaturalist.org/v1/points/{z}/{x}/{y}.png?taxon_id=${taxonId}&color=%23162A20&captive=false&quality_grade=research&verifiable=true`}
                />
              </LayersControl.Overlay>

              {hasIUCNRange && (
                <LayersControl.Overlay name="🗺️ IUCN Range Polygon">
                  <TileLayer
                    opacity={0.5}
                    url={`https://api.inaturalist.org/v1/taxon_ranges/${taxonId}/{z}/{x}/{y}.png`}
                  />
                </LayersControl.Overlay>
              )}
            </>
          )}
        </LayersControl>

        {userLat !== undefined && userLng !== undefined && (
          <Marker position={[userLat, userLng]} icon={currentLocIcon}>
            <Popup>Your Current Location</Popup>
          </Marker>
        )}

        {observations.map((obs, i) => (
          <Marker key={i} position={[obs.lat, obs.lng]} icon={getAnimalIcon()}>
            <Popup>
              <div className="text-center p-1">
                <strong className="text-[#162A20] font-extrabold">{animalName}</strong>
                <br />
                <span className="font-mono-tech text-[11px] text-[#5A655E]">
                  Spotted: {obs.date.toLocaleDateString()}
                </span>
              </div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>

      {/* Archival Legend */}
      <div className="absolute bottom-4 left-4 bg-[#FBF8F1]/95 p-3 rounded-xl text-xs text-[#162A20] z-[1000] border-2 border-[#162A20] tactile-shadow-sm font-mono-tech">
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center gap-2">
            <div className="w-4 h-3 bg-[#234733]/75 border border-[#162A20]"></div>
            <span className="font-bold text-[11px]">Native Range</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-3 bg-[#C84B31]/75 border border-[#162A20]"></div>
            <span className="font-bold text-[11px]">Introduced / Invasive</span>
          </div>
          {observations.length > 0 && (
            <div className="flex items-center gap-2">
              <span className="text-xs">🐾</span>
              <span className="font-bold text-[11px]">
                Your Sightings ({observations.length})
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
