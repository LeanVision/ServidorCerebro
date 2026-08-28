-- ============================================================================
-- Columna que consume procesar_y_enviar_supabase() en ServerLocal.py
--
-- CORRER ESTO ANTES DE DESPLEGAR EL CÓDIGO.
-- Si el código sale primero, PostgREST rechaza cada sesión con
-- `400 column "zonas_tiempo" does not exist` y se dejan de guardar visitas.
-- Al revés no pasa nada: la columna es nullable y el código viejo la ignora.
--
-- QUÉ GUARDA
-- Los segundos que cada visitante pasó en cada zona de negocio, por ejemplo:
--   {"Entrada": 12, "Pasillo": 41, "Zona.mesas": 180}
--
-- El Cerebro ya venía calculando esto al cerrar cada sesión y lo descartaba:
-- nadie lo leía y no viajaba a ningún lado. De acá salen tanto "cuánta gente
-- pasó por la entrada" (cuántas sesiones tienen esa zona) como "cuánto se
-- quedaron" (la suma de sus segundos).
--
-- LA CLAVE ES EL NOMBRE DE LA ZONA
-- Porque es lo que devuelve _zona_en_punto(). Renombrar una zona en el
-- calibrador parte su historial en dos: lo anterior queda con el nombre viejo.
-- Si en algún momento las zonas pasan a identificarse por id, esto hay que
-- migrarlo.
-- ============================================================================

alter table public.visitor_sessions
  add column if not exists zonas_tiempo jsonb;

-- Para filtrar por zona sin recorrer toda la tabla cuando haya volumen.
create index if not exists visitor_sessions_zonas_tiempo_idx
  on public.visitor_sessions using gin (zonas_tiempo);
