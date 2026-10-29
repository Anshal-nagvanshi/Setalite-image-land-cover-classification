import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'https://eiolnvhtjzwijtkqabpe.supabase.co';
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || 'sb_publishable_z-Eeb0Uu3uBpNlAovprOHA_d_l3BwXK';

export const supabase = createClient(supabaseUrl, supabaseAnonKey);
