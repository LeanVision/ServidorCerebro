-- ============================================================================
-- Tabla que consume reloj_fotos_heatmap_background() en ServerLocal.py
--
-- POR QUÉ
-- El heatmap del Cerebro es un acumulador en memoria sin dimensión temporal:
-- no se puede preguntar "la semana pasada" porque esa información nunca se
-- guardó, y se pierde entera en cada reinicio. Guardando fotos con timestamp,
-- el tránsito de cualquier período pasa a ser la resta entre dos fotos.
--
-- CÓMO SE LEE — la regla que no se puede olvidar
-- El acumulador vuelve a CERO en cada arranque del servidor, así que restar
-- dos fotos de `instancia` distinta da negativo. La primera foto de una
-- instancia nueva es una BASE, no se resta contra la anterior. Lo que se
-- acumuló entre la última foto y el reinicio se perdió: el intervalo de fotas
-- es el techo de esa pérdida.
--
-- DÓNDE VA
-- Esta es una propuesta como tabla propia. Puede tener más sentido dentro de
-- `store_events` del esquema de la plataforma (hoy vacía), donde el local se
-- identifica por uuid en vez de por el `branch_id` de texto libre que el
-- Cerebro viene escribiendo desde antes de que existiera `stores`.
-- Decisión pendiente.
-- ============================================================================

create table if not exists public.heatmap_snapshots (
  id           bigint generated always as identity primary key,
  branch_id    text not null,
  -- Arranque del servidor que generó esta foto. Sin esto no se puede saber
  -- dónde se reinició el contador. Ver la regla de arriba.
  instancia    text not null,
  captured_at  timestamptz not null,
  celda_px     real not null,
  celda_metros real,
  max          real not null,
  -- [{x, y, value}] en píxeles del plano. Sólo celdas con registro: las vacías
  -- no viajan, por eso una foto son ~100 filas y no 640x480.
  celdas       jsonb not null,
  created_at   timestamptz not null default now()
);

create index if not exists heatmap_snapshots_branch_capturado_idx
  on public.heatmap_snapshots (branch_id, captured_at desc);

-- Hace seguro el reintento del outbox. Las fotos se entregan desde la cola en
-- disco, que es entrega AL MENOS UNA VEZ: si el POST llega pero la respuesta se
-- corta, el reintento vuelve a mandar la misma foto. Con este índice, ese
-- reintento es rechazado con 409 / 23505 y el Cerebro lo trata como entrega
-- exitosa, porque el hecho ya está guardado.
--
-- Mismo criterio que visitor_sessions: INSERT plano y no upsert, porque el
-- upsert exige política de UPDATE y `anon` no la tiene ni debe tenerla.
create unique index if not exists heatmap_snapshots_identidad_uniq
  on public.heatmap_snapshots (branch_id, instancia, captured_at);

alter table public.heatmap_snapshots enable row level security;

-- El Cerebro escribe con la anon key y NO debe poder leer: si esa clave se
-- filtra desde la Pi, no sirve para sacar datos.
drop policy if exists "el cerebro puede guardar fotos" on public.heatmap_snapshots;
create policy "el cerebro puede guardar fotos"
  on public.heatmap_snapshots for insert to anon
  with check (true);

-- El panel lee sólo con sesión iniciada. Cuando haya más de una empresa esto
-- tiene que cruzar contra las membresías, igual que visitor_sessions.
drop policy if exists "usuarios logueados leen fotos" on public.heatmap_snapshots;
create policy "usuarios logueados leen fotos"
  on public.heatmap_snapshots for select to authenticated
  using (true);

-- ============================================================================
-- RETENCIÓN — definirla ahora, no en seis meses
-- Con fotos cada 15 minutos son ~96 filas por día por local, de unos pocos KB
-- cada una: unos pocos MB al mes. No es urgente, pero crece para siempre si
-- nadie lo mira. Lo razonable es agregar a diario y podar el detalle fino
-- pasadas unas semanas.
-- ============================================================================
