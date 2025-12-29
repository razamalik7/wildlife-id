import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

export const supabase = createClient(supabaseUrl, supabaseAnonKey)

// Types for our database
export interface Observation {
    id: string
    user_id: string
    image_url: string
    species: string
    confidence: number
    family: string | null
    class: string | null
    latitude: number | null
    longitude: number | null
    observed_at: string
    created_at: string
}

export interface UserProfile {
    id: string
    email: string
    display_name: string | null
    created_at: string
}
