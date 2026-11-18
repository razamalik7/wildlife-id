import { SpeciesInfo } from './AppContext';

export type EcosystemId =
    | 'forest'
    | 'coastal'
    | 'alpine'
    | 'wetland'
    | 'meadow'
    | 'arid';

export interface EcosystemMeta {
    id: EcosystemId;
    name: string;
    shortName: string;
    icon: string;
    color: string;
    bgTint: string;
    borderColor: string;
    accentColor: string;
    description: string;
}

export const ECOSYSTEMS: Record<EcosystemId, EcosystemMeta> = {
    forest: {
        id: 'forest',
        name: 'Temperate Forest',
        shortName: 'Forest',
        icon: '🌲',
        color: '#1E4631',
        bgTint: '#E6EFEA',
        borderColor: '#2A5C43',
        accentColor: '#C84B31',
        description: 'Hardwood ridges, mixed conifer woods & woodland understory'
    },
    coastal: {
        id: 'coastal',
        name: 'Coastal & Tidewater',
        shortName: 'Coastal',
        icon: '🌊',
        color: '#235365',
        bgTint: '#E3EEF2',
        borderColor: '#2B5B6C',
        accentColor: '#E8B84B',
        description: 'Rocky intertidal shores, estuaries, kelp beds & pelagic waters'
    },
    alpine: {
        id: 'alpine',
        name: 'Boreal & Alpine',
        shortName: 'Alpine',
        icon: '🏔️',
        color: '#523F2D',
        bgTint: '#EFE9E1',
        borderColor: '#6D543D',
        accentColor: '#C84B31',
        description: 'High peaks, subalpine krummholz, tundra & northern spruce taiga'
    },
    wetland: {
        id: 'wetland',
        name: 'Freshwater & Wetland',
        shortName: 'Wetland',
        icon: '🐸',
        color: '#2D5A3C',
        bgTint: '#E4F0E8',
        borderColor: '#3B6E4C',
        accentColor: '#E8B84B',
        description: 'Rivers, marshes, vernal pools, swamps & riparian banks'
    },
    meadow: {
        id: 'meadow',
        name: 'Grassland & Meadow',
        shortName: 'Meadow',
        icon: '🌾',
        color: '#8C6218',
        bgTint: '#F6EEDB',
        borderColor: '#B58326',
        accentColor: '#C84B31',
        description: 'Open prairies, upland meadows, savannas & agricultural edges'
    },
    arid: {
        id: 'arid',
        name: 'Arid & Canyon',
        shortName: 'Arid',
        icon: '🌵',
        color: '#9E3822',
        bgTint: '#F6E5E0',
        borderColor: '#C84B31',
        accentColor: '#E8B84B',
        description: 'Desert basins, red-rock canyons, scrubland & dry mesas'
    }
};

export const ECOSYSTEM_LIST: EcosystemMeta[] = Object.values(ECOSYSTEMS);

/**
 * Deterministically assigns a primary and optional secondary ecosystem to a species
 * based on its name, description, hotspots, and taxonomy.
 */
export function getSpeciesEcosystem(species?: SpeciesInfo | { name: string; description?: string; hotspots?: string[]; taxonomy?: any }): EcosystemMeta {
    if (!species) return ECOSYSTEMS.forest;

    const text = `${species.name} ${species.description || ''} ${(species.hotspots || []).join(' ')} ${species.taxonomy?.class || ''} ${species.taxonomy?.family || ''}`.toLowerCase();

    // 1. Coastal & Marine keywords
    if (
        text.includes('seal') ||
        text.includes('sea lion') ||
        text.includes('whale') ||
        text.includes('dolphin') ||
        text.includes('orca') ||
        text.includes('walrus') ||
        text.includes('puffin') ||
        text.includes('pelican') ||
        text.includes('gull') ||
        text.includes('tern') ||
        text.includes('albatross') ||
        text.includes('cormorant') ||
        text.includes('crab') ||
        text.includes('lobster') ||
        text.includes('sea otter') ||
        text.includes('sea turtle') ||
        text.includes('manatee') ||
        text.includes('shark') ||
        text.includes('ocean') ||
        text.includes('coast') ||
        text.includes('marine') ||
        text.includes('channel islands') ||
        text.includes('biscayne') ||
        text.includes('kenai fjords')
    ) {
        return ECOSYSTEMS.coastal;
    }

    // 2. Arid & Canyon keywords
    if (
        text.includes('desert') ||
        text.includes('gila') ||
        text.includes('roadrunner') ||
        text.includes('rattlesnake') ||
        text.includes('sidewinder') ||
        text.includes('scorpion') ||
        text.includes('tarantula') ||
        text.includes('chuckwalla') ||
        text.includes('horned lizard') ||
        text.includes('collared lizard') ||
        text.includes('tortoise') ||
        text.includes('javelina') ||
        text.includes('peccary') ||
        text.includes('ringtail') ||
        text.includes('kit fox') ||
        text.includes('pronghorn') ||
        text.includes('kangaroo rat') ||
        text.includes('cactus') ||
        text.includes('joshua tree') ||
        text.includes('saguaro') ||
        text.includes('death valley') ||
        text.includes('big bend') ||
        text.includes('zion') ||
        text.includes('grand canyon') ||
        text.includes('arches')
    ) {
        return ECOSYSTEMS.arid;
    }

    // 3. Freshwater & Wetland keywords
    if (
        text.includes('frog') ||
        text.includes('toad') ||
        text.includes('salamander') ||
        text.includes('newt') ||
        text.includes('alligator') ||
        text.includes('crocodile') ||
        text.includes('turtle') ||
        text.includes('slider') ||
        text.includes('cooter') ||
        text.includes('terrapin') ||
        text.includes('watersnake') ||
        text.includes('cottonmouth') ||
        text.includes('beaver') ||
        text.includes('muskrat') ||
        text.includes('nutria') ||
        text.includes('river otter') ||
        text.includes('mink') ||
        text.includes('heron') ||
        text.includes('egret') ||
        text.includes('ibis') ||
        text.includes('spoonbill') ||
        text.includes('stork') ||
        text.includes('crane') ||
        text.includes('duck') ||
        text.includes('mallard') ||
        text.includes('goose') ||
        text.includes('swan') ||
        text.includes('loon') ||
        text.includes('grebe') ||
        text.includes('kingfisher') ||
        text.includes('osprey') ||
        text.includes('everglades') ||
        text.includes('congaree') ||
        text.includes('marsh') ||
        text.includes('swamp') ||
        text.includes('wetland') ||
        text.includes('pond') ||
        species.taxonomy?.class === 'Amphibia'
    ) {
        return ECOSYSTEMS.wetland;
    }

    // 4. Boreal & Alpine keywords
    if (
        text.includes('mountain goat') ||
        text.includes('bighorn') ||
        text.includes('dall sheep') ||
        text.includes('pika') ||
        text.includes('marmot') ||
        text.includes('wolverine') ||
        text.includes('lynx') ||
        text.includes('caribou') ||
        text.includes('moose') ||
        text.includes('polar bear') ||
        text.includes('grizzly') ||
        text.includes('snowy owl') ||
        text.includes('ptarmigan') ||
        text.includes('marten') ||
        text.includes('fisher') ||
        text.includes('ermine') ||
        text.includes('snowshoe hare') ||
        text.includes('arctic') ||
        text.includes('tundra') ||
        text.includes('alpine') ||
        text.includes('denali') ||
        text.includes('glacier national') ||
        text.includes('rocky mountain')
    ) {
        return ECOSYSTEMS.alpine;
    }

    // 5. Grassland & Meadow keywords
    if (
        text.includes('bison') ||
        text.includes('prairie') ||
        text.includes('badger') ||
        text.includes('ferret') ||
        text.includes('coyote') ||
        text.includes('jackrabbit') ||
        text.includes('cottontail') ||
        text.includes('ground squirrel') ||
        text.includes('meadowlark') ||
        text.includes('kestrel') ||
        text.includes('burrowing owl') ||
        text.includes('barn owl') ||
        text.includes('red-tailed hawk') ||
        text.includes('pheasant') ||
        text.includes('quail') ||
        text.includes('grouse') ||
        text.includes('monarch') ||
        text.includes('bumblebee') ||
        text.includes('grassland') ||
        text.includes('badlands') ||
        text.includes('wind cave') ||
        text.includes('theodore roosevelt')
    ) {
        return ECOSYSTEMS.meadow;
    }

    // Default: Temperate Forest
    return ECOSYSTEMS.forest;
}

export interface ParkRecord {
    id: number;
    name: string;
    state: string;
    country: string;
    lat: number;
    lng: number;
}

export function haversineKm(lat1: number, lng1: number, lat2: number, lng2: number): number {
    const R = 6371;
    const dLat = ((lat2 - lat1) * Math.PI) / 180;
    const dLng = ((lng2 - lng1) * Math.PI) / 180;
    const a =
        Math.sin(dLat / 2) * Math.sin(dLat / 2) +
        Math.cos((lat1 * Math.PI) / 180) *
            Math.cos((lat2 * Math.PI) / 180) *
            Math.sin(dLng / 2) *
            Math.sin(dLng / 2);
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

export function findNearestPark(
    lat?: number,
    lng?: number,
    parks: ParkRecord[] = [],
    maxDistKm = 45
): { park: ParkRecord; distKm: number } | null {
    if (lat === undefined || lng === undefined || !parks.length) return null;
    let best: ParkRecord | null = null;
    let bestDist = Infinity;

    for (const p of parks) {
        const d = haversineKm(lat, lng, p.lat, p.lng);
        if (d < bestDist) {
            bestDist = d;
            best = p;
        }
    }
    if (best && bestDist <= maxDistKm) {
        return { park: best, distKm: bestDist };
    }
    return null;
}

export function formatLocationStamp(lat?: number, lng?: number, parks: ParkRecord[] = []): string {
    const match = findNearestPark(lat, lng, parks, 45);
    if (match) {
        return `${match.park.name.toUpperCase()} · ${match.park.state}`;
    }
    if (lat !== undefined && lng !== undefined && (lat !== 0 || lng !== 0)) {
        const ns = lat >= 0 ? 'N' : 'S';
        const ew = lng >= 0 ? 'E' : 'W';
        return `${Math.abs(lat).toFixed(2)}°${ns}, ${Math.abs(lng).toFixed(2)}°${ew}`;
    }
    return 'FIELD OBSERVATION';
}

/**
 * North American Macro-Ecosystem Overlay Regions for the Leaflet Map
 */
export interface EcosystemZone {
    id: EcosystemId;
    regionName: string;
    positions: [number, number][];
}

export const NORTH_AMERICA_BIOME_ZONES: EcosystemZone[] = [
    {
        id: 'forest',
        regionName: 'Eastern & Appalachian Temperate Forest',
        positions: [
            [47.5, -92.0],
            [47.0, -67.5],
            [39.0, -74.0],
            [32.5, -81.0],
            [33.0, -94.0],
            [41.0, -93.5],
        ],
    },
    {
        id: 'forest',
        regionName: 'Pacific Northwest Conifer Forest',
        positions: [
            [49.0, -124.5],
            [49.0, -117.0],
            [40.0, -120.0],
            [40.0, -124.5],
        ],
    },
    {
        id: 'alpine',
        regionName: 'Rocky Mountain & Boreal Cordillera',
        positions: [
            [62.0, -150.0],
            [60.0, -112.0],
            [48.5, -113.0],
            [37.0, -105.0],
            [37.0, -113.0],
            [48.0, -120.5],
            [58.0, -138.0],
        ],
    },
    {
        id: 'meadow',
        regionName: 'Great Plains & Central Prairie',
        positions: [
            [51.0, -112.0],
            [50.0, -95.0],
            [33.0, -95.0],
            [29.5, -100.5],
            [37.0, -104.5],
            [48.5, -112.5],
        ],
    },
    {
        id: 'arid',
        regionName: 'Southwest Canyons, Sonoran & Mojave Deserts',
        positions: [
            [40.0, -119.5],
            [37.0, -104.5],
            [28.5, -101.0],
            [26.0, -112.0],
            [33.0, -118.0],
        ],
    },
    {
        id: 'wetland',
        regionName: 'Gulf Coast, Everglades & Mississippi Delta Wetlands',
        positions: [
            [33.0, -93.5],
            [32.5, -80.0],
            [24.8, -80.2],
            [25.5, -83.0],
            [29.5, -89.0],
            [28.5, -96.5],
        ],
    },
    {
        id: 'coastal',
        regionName: 'Atlantic & Acadian Tidewater Shelf',
        positions: [
            [46.5, -66.0],
            [43.5, -65.0],
            [34.5, -74.0],
            [35.5, -76.5],
            [43.5, -70.5],
        ],
    },
    {
        id: 'coastal',
        regionName: 'Pacific Kelp & Channel Shelf',
        positions: [
            [48.5, -126.0],
            [48.5, -123.5],
            [32.5, -117.0],
            [32.0, -120.5],
            [40.0, -125.5],
        ],
    },
];
