'use client';

import { useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';
import AppHeader from '@/components/AppHeader';

interface Log {
    id: number;
    created_at: string;
    species_prediction: string;
    confidence: number;
    family: string;
    class_name: string;
    filename: string;
}

export default function AdminPage() {
    const [logs, setLogs] = useState<Log[]>([]);
    const [loading, setLoading] = useState(true);

    const fetchLogs = async () => {
        setLoading(true);
        const { data, error } = await supabase
            .from('prediction_logs')
            .select('*')
            .order('created_at', { ascending: false })
            .limit(50);

        if (data) {
            setLogs(data);
        }
        if (error) {
            console.error("Error fetching logs:", error);
        }
        setLoading(false);
    };

    useEffect(() => {
        fetchLogs();

        // Realtime subscription (Optional, enables live updates)
        const channel = supabase
            .channel('realtime logs')
            .on('postgres_changes', { event: 'INSERT', schema: 'public', table: 'prediction_logs' }, (payload) => {
                setLogs((prev) => [payload.new as Log, ...prev].slice(0, 50));
            })
            .subscribe();

        return () => {
            supabase.removeChannel(channel);
        };
    }, []);

    return (
        <main className="min-h-screen bg-stone-100 text-stone-900 pb-24">
            <AppHeader />

            <div className="max-w-6xl mx-auto p-4">
                <div className="flex justify-between items-center mb-6">
                    <h1 className="text-2xl font-bold text-stone-700">Live Prediction Logs</h1>
                    <button
                        onClick={fetchLogs}
                        className="px-4 py-2 bg-stone-200 hover:bg-stone-300 rounded-lg text-sm font-bold transition-colors"
                    >
                        Refresh
                    </button>
                </div>

                <div className="bg-white rounded-3xl shadow-sm border border-stone-200 overflow-hidden">
                    <div className="overflow-x-auto">
                        <table className="w-full text-left text-sm text-stone-600">
                            <thead className="bg-stone-50 text-stone-400 uppercase tracking-wider font-bold">
                                <tr>
                                    <th className="p-4">Time</th>
                                    <th className="p-4">Species</th>
                                    <th className="p-4">Conf</th>
                                    <th className="p-4">Family / Class</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-stone-100">
                                {loading ? (
                                    <tr>
                                        <td colSpan={4} className="p-8 text-center text-stone-400">Loading logs...</td>
                                    </tr>
                                ) : logs.length === 0 ? (
                                    <tr>
                                        <td colSpan={4} className="p-8 text-center text-stone-400">No predictions logged yet.</td>
                                    </tr>
                                ) : (
                                    logs.map((log) => (
                                        <tr key={log.id} className="hover:bg-stone-50/50 transition-colors">
                                            <td className="p-4 font-mono text-xs opacity-70">
                                                {new Date(log.created_at).toLocaleTimeString()}
                                            </td>
                                            <td className="p-4 font-bold text-stone-800">
                                                {log.species_prediction}
                                            </td>
                                            <td className="p-4">
                                                <span
                                                    className={`px-2 py-1 rounded-md text-xs font-bold ${log.confidence > 80 ? 'bg-emerald-100 text-emerald-700' :
                                                            log.confidence > 40 ? 'bg-amber-100 text-amber-700' :
                                                                'bg-red-100 text-red-700'
                                                        }`}
                                                >
                                                    {log.confidence.toFixed(1)}%
                                                </span>
                                            </td>
                                            <td className="p-4 text-xs">
                                                <span className="block font-bold text-stone-600">{log.family}</span>
                                                <span className="block opacity-60">{log.class_name}</span>
                                            </td>
                                        </tr>
                                    ))
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </main>
    );
}
