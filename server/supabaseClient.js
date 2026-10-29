import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';

dotenv.config();

const supabaseUrl = process.env.SUPABASE_URL || 'https://eiolnvhtjzwijtkqabpe.supabase.co';
const supabaseKey = process.env.SUPABASE_KEY || 'sb_publishable_z-Eeb0Uu3uBpNlAovprOHA_d_l3BwXK';

export const supabase = createClient(supabaseUrl, supabaseKey);
