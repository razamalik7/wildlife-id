'use client';

import {
    MapContainer,
    TileLayer,
    Marker,
    Popup,
    Polygon,
    Tooltip,
    useMap,
} from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { useEffect } from 'react';
import L from 'leaflet';
import Link from 'next/link';
import { speciesIconUrl } from '@/lib/species';
import {
    ECOSYSTEMS,
    NORTH_AMERICA_BIOME_ZONES,
    ParkRecord,
    formatLocationStamp,
} from '@/lib/ecosystems';

interface ObservationsMapProps {
    observations: any[];
    parks: ParkRecord[];
    showEcosystems: boolean;
    showParks: boolean;
    selectedPark: ParkRecord | null;
    flyTarget: { lat: number; lng: number; zoom?: number } | null;
    onSelectPark: (park: ParkRecord) => void;
}

function MapFlyController({
    flyTarget,
}: {
    flyTarget: { lat: number; lng: number; zoom?: number } | null;
}) {
    const map = useMap();

    useEffect(() => {
        setTimeout(() => map.invalidateSize(), 100);
    }, [map]);

    useEffect(() => {
        if (flyTarget) {
            map.flyTo([flyTarget.lat, flyTarget.lng], flyTarget.zoom ?? 9, {
                duration: 1.2,
            });
        }
    }, [flyTarget, map]);

    return null;
}

function createAnimalPinIcon(speciesName: string) {
    const imgUrl = speciesIconUrl(speciesName);
    return L.divIcon({
        className: 'custom-animal-pin',
        html: `
            <div style="
                width: 48px;
                height: 48px;
                background: #FBF8F1;
                border: 2.5px solid #162A20;
                border-radius: 14px;
                box-shadow: 3px 3px 0px #162A20;
                display: flex;
                align-items: center;
                justify-content: center;
                position: relative;
                overflow: visible;
            ">
                <img
                    src="${imgUrl}"
                    alt="${speciesName}"
                    onerror="this.src='/icons/default.png'"
                    style="width: 42px; height: 42px; object-fit: contain; transform: translateY(-3px);"
                />
                <div style="
                    position: absolute;
                    bottom: -6px;
                    left: 50%;
                    transform: translateX(-50%) rotate(45deg);
                    width: 10px;
                    height: 10px;
                    background: #C84B31;
                    border-right: 2px solid #162A20;
                    border-bottom: 2px solid #162A20;
                "></div>
            </div>
        `,
        iconSize: [48, 54],
        iconAnchor: [24, 52],
        popupAnchor: [0, -48],
    });
}

function createParkPatchIcon(
    park: ParkRecord,
    isSelected: boolean,
    hasSighting: boolean
) {
    if (isSelected) {
        const shortLabel = park.name.replace(' NP', '');
        return L.divIcon({
            className: 'custom-park-patch-selected',
            html: `
                <div style="
                    display: inline-flex;
                    align-items: center;
                    gap: 5px;
                    padding: 4px 10px;
                    background: #C84B31;
                    color: #FBF8F1;
                    border: 2px dashed #F4DE9B;
                    outline: 2.5px solid #162A20;
                    border-radius: 999px;
                    box-shadow: 3px 3px 0px #162A20;
                    font-family: monospace;
                    font-size: 11px;
                    font-weight: 800;
                    white-space: nowrap;
                    transform: translate(-50%, -50%);
                ">
                    <span>🏞️</span>
                    <span>${shortLabel}</span>
                </div>
            `,
            iconSize: [140, 30],
            iconAnchor: [70, 15],
        });
    }

    const bg = hasSighting ? '#162A20' : '#F4DE9B';
    const border = hasSighting ? '#F4DE9B' : '#162A20';

    return L.divIcon({
        className: 'custom-park-patch-dot',
        html: `
            <div style="
                width: 24px;
                height: 24px;
                background: ${bg};
                border: 2px dashed ${border};
                outline: 2px solid #162A20;
                border-radius: 50%;
                box-shadow: 2px 2px 0px #162A20;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 11px;
            ">
                🌲
            </div>
        `,
        iconSize: [24, 24],
        iconAnchor: [12, 12],
    });
}

export default function ObservationsMap({
    observations,
    parks,
    showEcosystems,
    showParks,
    selectedPark,
    flyTarget,
    onSelectPark,
}: ObservationsMapProps) {
    const center: [number, number] =
        observations.length > 0 && observations[0].lat && observations[0].lng
            ? [observations[0].lat, observations[0].lng]
            : [39.8283, -98.5795];

    const visibleParks = parks.filter((p) => {
        if (selectedPark && p.id === selectedPark.id) return true;
        if (p.name.endsWith('NP')) return true;
        return false;
    });

    return (
        <MapContainer
            center={center}
            zoom={observations.length > 0 ? 5 : 4}
            style={{ height: '100%', width: '100%' }}
            className="z-0"
        >
            <MapFlyController flyTarget={flyTarget} />

            {/* Esri World Topographic Atlas — Zero API Key Watermark, True National Park Terrain Contours */}
            <TileLayer
                attribution="Tiles &copy; Esri &mdash; Esri, USGS, NPS"
                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
            />

            {/* ============================================================
                OPTIONAL NORTH AMERICAN ECOSYSTEM OVERLAY (STEP D)
               ============================================================ */}
            {showEcosystems &&
                NORTH_AMERICA_BIOME_ZONES.map((zone, idx) => {
                    const eco = ECOSYSTEMS[zone.id];
                    return (
                        <Polygon
                            key={`${zone.id}-${idx}`}
                            positions={zone.positions}
                            pathOptions={{
                                color: eco.borderColor,
                                fillColor: eco.color,
                                fillOpacity: 0.28,
                                weight: 2.5,
                                dashArray: '6 4',
                            }}
                        >
                            <Tooltip sticky>
                                <div className="font-mono-tech text-xs font-extrabold text-[#162A20]">
                                    {eco.icon} {zone.regionName} ({eco.name})
                                </div>
                            </Tooltip>
                        </Polygon>
                    );
                })}

            {/* ============================================================
                M2 STITCHED NATIONAL PARK PATCH MEDALLIONS
               ============================================================ */}
            {showParks &&
                visibleParks.map((park) => {
                    const isSel = selectedPark?.id === park.id;
                    const hasSighting = observations.some(
                        (o) =>
                            Math.abs((o.lat || 0) - park.lat) < 0.45 &&
                            Math.abs((o.lng || 0) - park.lng) < 0.45
                    );
                    return (
                        <Marker
                            key={`park-${park.id}`}
                            position={[park.lat, park.lng]}
                            icon={createParkPatchIcon(park, isSel, hasSighting)}
                            eventHandlers={{
                                click: () => onSelectPark(park),
                            }}
                        >
                            <Tooltip direction="top" offset={[0, -10]}>
                                <span className="font-mono-tech text-xs font-extrabold text-[#162A20]">
                                    🏞️ {park.name} ({park.state}) — Click to inspect
                                </span>
                            </Tooltip>
                        </Marker>
                    );
                })}

            {/* ============================================================
                USER OBSERVATION CUTOUT ANIMAL PINS + POLAROID POPUPS
               ============================================================ */}
            {observations
                .filter((obs) => obs.lat && obs.lng)
                .map((obs, idx) => {
                    const speciesName = obs.common_name || obs.species || 'Wildlife';
                    const stamp = formatLocationStamp(obs.lat, obs.lng, parks);
                    return (
                        <Marker
                            key={obs.id || idx}
                            position={[obs.lat, obs.lng]}
                            icon={createAnimalPinIcon(speciesName)}
                        >
                            <Popup className="custom-popup">
                                <div className="flex flex-col gap-2 min-w-[185px] p-1">
                                    <div className="w-full h-28 bg-[#162A20] rounded-lg overflow-hidden border-2 border-[#162A20] relative">
                                        {obs.image_url ? (
                                            <img
                                                src={obs.image_url}
                                                alt={speciesName}
                                                className="w-full h-full object-cover"
                                            />
                                        ) : (
                                            <div className="w-full h-full bg-[#F7F3E8] flex items-center justify-center p-2">
                                                <img
                                                    src={speciesIconUrl(speciesName)}
                                                    alt={speciesName}
                                                    className="h-20 w-20 object-contain"
                                                />
                                            </div>
                                        )}
                                    </div>
                                    <div>
                                        <span className="inline-block font-mono-tech text-[9px] font-extrabold uppercase text-[#C84B31] border border-[#C84B31] px-1.5 py-0.5 rounded bg-[#C84B31]/10 mb-1">
                                            {stamp}
                                        </span>
                                        <h3 className="font-extrabold text-sm text-[#162A20] leading-tight">
                                            {speciesName}
                                        </h3>
                                        {obs.scientific_name && (
                                            <p className="text-[10px] text-[#5A655E] font-mono-tech italic">
                                                {obs.scientific_name}
                                            </p>
                                        )}
                                        <div className="mt-2 pt-1.5 border-t border-[#162A20]/15 flex items-center justify-between">
                                            <span className="text-[10px] font-mono-tech text-[#5A655E]">
                                                {new Date(
                                                    obs.created_at || obs.observed_at || Date.now()
                                                ).toLocaleDateString()}
                                            </span>
                                            <Link
                                                href={`/anidex/${encodeURIComponent(
                                                    speciesName
                                                )}?side=back`}
                                                className="text-[10px] font-mono-tech font-extrabold text-[#C84B31] underline"
                                            >
                                                Polaroid Card →
                                            </Link>
                                        </div>
                                    </div>
                                </div>
                            </Popup>
                        </Marker>
                    );
                })}
        </MapContainer>
    );
}
