"use client";

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { ArrowLeft, Lock, MapPin, Calendar, X } from 'lucide-react';

interface Sighting {
  id: number;
  species_name: string;
  region: string;
  timestamp: string;
  image_path: string;
  latitude: number;
  longitude: number;
}

interface AnidexEntry {
  id: number;
  name: string;
  fact: string;
  unlocked: boolean;
  sightings: Sighting[];
}

export default function AnidexPage() {
  const [entries, setEntries] = useState<AnidexEntry[]>([]);
  const [selectedAnimal, setSelectedAnimal] = useState<AnidexEntry | null>(null);

  useEffect(() => {
    // 1. Fetch from the NEW endpoint
    fetch('http://127.0.0.1:8000/api/anidex')
      .then(res => res.json())
      .then(data => {
        // 2. Set the data to 'entries' (matching the backend response)
        setEntries(data.anidex); 
      })
      .catch(err => console.error("Failed to load Anidex:", err));
  }, []);

  return (
    <div className="min-h-screen bg-[#1a1a1a] p-8 font-sans text-white">
      <div className="max-w-6xl mx-auto">
        
        {/* Header */}
        <div className="flex items-center gap-4 mb-8">
          <Link href="/" className="p-2 bg-white/10 rounded-full hover:bg-white/20 transition-colors">
            <ArrowLeft className="w-6 h-6" />
          </Link>
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-[#4f824f]">ANIDEX</h1>
            <p className="text-gray-400 text-sm">
              {entries.filter(e => e.unlocked).length} / {entries.length} Species Discovered
            </p>
          </div>
        </div>

        {/* THE GRID */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {entries.map((animal) => (
            <button
              key={animal.id}
              onClick={() => animal.unlocked && setSelectedAnimal(animal)}
              disabled={!animal.unlocked}
              className={`
                relative aspect-[3/4] rounded-xl overflow-hidden border-2 transition-all duration-300 group
                ${animal.unlocked 
                  ? 'border-[#4f824f] cursor-pointer hover:scale-105 hover:shadow-[0_0_20px_rgba(79,130,79,0.4)]' 
                  : 'border-gray-800 bg-gray-900 opacity-70 cursor-not-allowed'}
              `}
            >
              {animal.unlocked ? (
                // UNLOCKED STATE
                <>
                  <img 
                    src={`http://127.0.0.1:8000/uploads/${animal.sightings[0].image_path}`} 
                    className="absolute inset-0 w-full h-full object-cover"
                    alt={animal.name}
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-transparent to-transparent flex flex-col justify-end p-4">
                    <h3 className="font-bold text-lg">{animal.name}</h3>
                    <p className="text-xs text-[#4f824f] font-mono">
                      {animal.sightings.length} SIGHTINGS
                    </p>
                  </div>
                </>
              ) : (
                // LOCKED STATE
                <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-700">
                  <Lock className="w-12 h-12 mb-2" />
                  <span className="text-xs font-mono uppercase tracking-widest">Undiscovered</span>
                  <div className="absolute bottom-4 font-bold text-gray-800 text-xl opacity-50">
                    ???
                  </div>
                </div>
              )}
            </button>
          ))}
        </div>

        {/* DETAIL MODAL (Pop-up when clicked) */}
        {selectedAnimal && (
          <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="bg-gray-900 w-full max-w-2xl max-h-[90vh] rounded-2xl border border-gray-700 overflow-hidden flex flex-col">
              
              {/* Modal Header */}
              <div className="p-6 border-b border-gray-800 flex justify-between items-start">
                <div>
                  <h2 className="text-3xl font-bold text-white mb-1">{selectedAnimal.name}</h2>
                  <p className="text-gray-400 text-sm italic">{selectedAnimal.fact}</p>
                </div>
                <button 
                  onClick={() => setSelectedAnimal(null)}
                  className="p-2 hover:bg-gray-800 rounded-full"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              {/* Scrollable History List */}
              <div className="overflow-y-auto p-6 space-y-4">
                <h3 className="text-sm font-bold text-[#4f824f] uppercase tracking-wider mb-4">
                  Sighting History
                </h3>
                
                {selectedAnimal.sightings.map((sighting) => (
                  <div key={sighting.id} className="flex gap-4 bg-black/20 p-4 rounded-xl border border-gray-800">
                    {/* Thumbnail */}
                    <img 
                      src={`http://127.0.0.1:8000/uploads/${sighting.image_path}`} 
                      className="w-24 h-24 object-cover rounded-lg bg-gray-800"
                    />
                    
                    {/* Details */}
                    <div className="flex-1">
                      <div className="flex items-center gap-2 text-sm text-gray-300 mb-1">
                        <Calendar className="w-4 h-4 text-[#4f824f]" />
                        {new Date(sighting.timestamp).toLocaleDateString()}
                      </div>
                      <div className="flex items-center gap-2 text-sm text-gray-300 mb-2">
                        <MapPin className="w-4 h-4 text-[#4f824f]" />
                        {sighting.region}
                      </div>
                    </div>
                  </div>
                ))}
              </div>

            </div>
          </div>
        )}

      </div>
    </div>
  );
}