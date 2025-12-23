"use client";

import { MapContainer, TileLayer, Marker, Popup, LayersControl } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";

// 1. Standard Blue Pin for "You are Here"
const currentLocIcon = L.icon({
  iconUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
});

interface Props {
  taxonId: number;
  userLat: number;
  userLng: number;
  animalName: string;
  observations?: Array<{ lat: number, lng: number, date: Date }>;
}

export default function SpeciesMap({ taxonId, userLat, userLng, animalName, observations = [] }: Props) {

  // 2. DYNAMIC ICON GENERATOR
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
        center={[userLat, userLng]}
        zoom={5}
        style={{ height: "100%", width: "100%" }}
        scrollWheelZoom={true}
      >
        <LayersControl position="topright">

          <LayersControl.BaseLayer checked name="Dark Context">
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
              url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            />
          </LayersControl.BaseLayer>

          <LayersControl.BaseLayer name="Satellite">
            <TileLayer
              url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
              attribution='Tiles &copy; Esri'
            />
          </LayersControl.BaseLayer>

          {/* THE UNIFIED HEATMAP */}
          {taxonId > 0 && (
            <LayersControl.Overlay checked name="Population Density">
              <TileLayer
                opacity={0.7}
                // ADDED: &captive=false (No Zoos) &quality_grade=research (No blurry/unconfirmed photos)
                url={`https://api.inaturalist.org/v1/heatmap/{z}/{x}/{y}.png?taxon_id=${taxonId}&color=%23ff5733&captive=false&quality_grade=research`}
              />
            </LayersControl.Overlay>
          )}

        </LayersControl>

        {/* Current Location */}
        <Marker position={[userLat, userLng]} icon={currentLocIcon}>
          <Popup>You are here</Popup>
        </Marker>

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

      {/* 5. CLEANED UP LEGEND */}
      <div className="absolute bottom-4 left-4 bg-black/80 p-3 rounded-lg text-xs text-white z-[1000] border border-stone-600 shadow-xl">
        <div className="flex flex-col gap-2">

          {/* Density Gradient */}
          <div className="flex items-center gap-2">
            <div className="w-12 h-3 rounded-full bg-gradient-to-r from-yellow-500 via-orange-500 to-red-600 border border-white/20"></div>
            <span>Population Density</span>
          </div>

          {/* Current Location Pin */}
          <div className="flex items-center gap-2">
            <img src="https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png" className="w-3 h-5" alt="pin" />
            <span>Current Location</span>
          </div>

          {/* Your Observations (Animal Icon) */}
          <div className="flex items-center gap-2">
            <img
              src={`/icons/${animalName.toLowerCase().replace(/ /g, "_")}_locked.png`}
              className="w-5 h-5 object-contain opacity-90"
              alt="icon"
            />
            <span>Your Observations</span>
          </div>

        </div>
      </div>
    </div>
  );
}