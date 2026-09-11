-- ============================================================================
-- Cierra el acceso directo a visitor_sessions y heatmap_snapshots.
--
-- CORRER DESPUÉS de que el Cerebro entregue por la app (LEANRETAIL_INGEST_URL en
-- su .env) y ya se vean filas nuevas con branch_id = 'principal'. Antes de eso,
-- corta el guardado de visitas: el Cerebro las retiene en su cola en disco, pero
-- no llegan.
--
-- POR QUÉ
-- El Cerebro escribía con la anon key compartida y un branch_id de texto libre:
-- quien tuviera esa clave podía insertar visitas a nombre de cualquier sucursal
-- de cualquier organización. Y la anon key no es secreta: además de la Pi, la ve
-- cualquier navegador que abra la app (NEXT_PUBLIC_SUPABASE_ANON_KEY).
--
-- Ahora escribe la app, con la service role, que no pasa por estos permisos. El
-- panel también lee con la service role, del lado del servidor y después de
-- validar la membresía (lib/visitor-traffic.ts, lib/heatmap-history.ts), así que
-- ni `anon` ni `authenticated` necesitan tocar estas tablas.
--
-- Se revoca a nivel de GRANT y no política por política porque las políticas de
-- visitor_sessions se crearon a mano y no tienen nombre conocido. Sin el grant,
-- ninguna política vieja vuelve a abrir la puerta.
-- ============================================================================

revoke all on public.visitor_sessions from anon, authenticated;
revoke all on public.heatmap_snapshots from anon, authenticated;

drop policy if exists "el cerebro puede guardar fotos" on public.heatmap_snapshots;
