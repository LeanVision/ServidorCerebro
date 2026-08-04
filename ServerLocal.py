import cv2
import numpy as np
import time
import base64
import asyncio
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
import uvicorn
import requests

import torch
import torch.nn.functional as F
from torchreid.utils import FeatureExtractor
import onnxruntime as ort 

# --- CONEXIÓN DIRECTA A SUPABASE ---
SUPABASE_URL = "https://butoxtgngmbnkmgueavf.supabase.co/rest/v1/visitor_sessions"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ1dG94dGduZ21ibmttZ3VlYXZmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODMxNjUwODcsImV4cCI6MjA5ODc0MTA4N30.nFYb_11mK353SWQMCNQIjM3IcbhIrpbD9M59iMHWkaM"
SUPABASE_HEADERS = {
    "apikey": SUPABASE_KEY, 
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json", 
    "Prefer": "return=representation"
}

IP_CAMARA = "http://172.31.99.7:8002"

# =========================================================
# 🦅 CALIBRACIÓN BIRD'S EYE VIEW (Vista de Pájaro)
# =========================================================
PUNTOS_CAMARA = np.float32([
    [120, 250],  # Arriba Izquierda
    [360, 250],  # Arriba Derecha
    [480, 640],  # Abajo Derecha
    [0, 640]     # Abajo Izquierda
])

PUNTOS_PLANO = np.float32([
    [0, 0],       
    [480, 0],     
    [480, 640],   
    [0, 640]      
])

MATRIZ_HOMOGRAFIA = cv2.getPerspectiveTransform(PUNTOS_CAMARA, PUNTOS_PLANO)
# =========================================================

app = FastAPI(title="Cerebro Central Director + Web Dashboard (Re-ID IBN Pro)")

# 🚀 MEJORA DE RE-ID: Usamos OSNet-IBN (Mayor robustez ante cambios de luz y ángulo)
print("🧠 Cargando modelo avanzado IA Re-ID (OSNet-IBN Omni-Scale)...")
extractor_ia = FeatureExtractor(model_name='osnet_ibn_x1_0', device='cpu')
print("✅ ¡Modelo OSNet-IBN listo para producción!")

print("🧠 Buscando modelo ONNX de Edad/Género (Hugging Face)...")
try:
    session_age_gender = ort.InferenceSession("demografia.onnx")
    tiene_onnx = True
    print("✅ ¡Modelo ONNX cargado correctamente!")
except Exception as e:
    session_age_gender = None
    tiene_onnx = False
    print("⚠️ Archivo 'demografia.onnx' no encontrado.")

# 🎯 AJUSTES FINOS DE RE-ID
UMBRAL_SIMILITUD = 0.38       # Ligeramente más flexible para tolerar cambios de sombra
MAX_FOTOS_ALBUM = 8           # Ampliamos el álbum para capturar más ángulos (antes 5)
TIEMPO_TELETRANSPORTACION = 3 
TIEMPO_INACTIVIDAD_SEGUNDOS = 60 

clientes_globales = {}  
traductor_camaras = {}  
contador_global_ids = 1

puntos_heatmap = []
MAX_PUNTOS_HEATMAP = 20000

@app.post("/identificar")
async def identificar_persona(
    file: UploadFile = File(...), 
    zona: str = Form("Desconocida"),
    tracker_id: str = Form(None),
    camara_id: str = Form("camara_default"),
    branch_id: str = Form("SUC-001"),
    pos_x: str = Form("0"),
    pos_y: str = Form("0") 
):
    global contador_global_ids
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    imagen_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    imagen_rgb = cv2.cvtColor(imagen_cv2, cv2.COLOR_BGR2RGB)
    foto_b64 = base64.b64encode(contents).decode('utf-8')
    
    # 🦅 Transformar coordenadas al mapa plano
    try:
        x_val = float(pos_x)
        y_val = float(pos_y)
        if x_val > 0 or y_val > 0:
            punto_original = np.array([[[x_val, y_val]]], dtype=np.float32)
            punto_transformado = cv2.perspectiveTransform(punto_original, MATRIZ_HOMOGRAFIA)
            nx, ny = punto_transformado[0][0]
            
            if 0 <= nx <= 480 and 0 <= ny <= 640:
                puntos_heatmap.append({"x": int(nx), "y": int(ny), "value": 1})
                if len(puntos_heatmap) > MAX_PUNTOS_HEATMAP:
                    puntos_heatmap.pop(0) 
    except ValueError:
        pass

    # Extracción de características con OSNet-IBN
    with torch.no_grad():
        huellas_batch = extractor_ia([imagen_rgb]) 
        huella_bruta = huellas_batch[0]            
        huella_nueva = F.normalize(huella_bruta, p=2, dim=0)

    ahora = time.time()
    id_local_camara = f"{camara_id}_{tracker_id}" if tracker_id else None

    # Actualización directa si viene del mismo tracker local
    if id_local_camara and id_local_camara in traductor_camaras:
        id_global = traductor_camaras[id_local_camara]["id_global"]
        if id_global in clientes_globales:
            if ahora - clientes_globales[id_global].get("ultimo_update_album", 0) > 1.0:
                album = clientes_globales[id_global]["historial"]
                if len(album) >= MAX_FOTOS_ALBUM:
                    album.pop(0) 
                album.append(huella_nueva)
                clientes_globales[id_global]["ultimo_update_album"] = ahora
            
            clientes_globales[id_global].update({
                "foto_b64": foto_b64, 
                "zona_actual": zona, 
                "timestamp": ahora, 
                "hora_legible": time.strftime("%H:%M:%S")
            })
            traductor_camaras[id_local_camara]["ultimo_update"] = ahora
            return {"status": "ok", "id_asignado": id_global}

    personas_activas_en_esta_camara = set()
    for tk, info_tk in traductor_camaras.items():
        if tk.startswith(f"{camara_id}_") and tk != id_local_camara:
            if (ahora - info_tk["ultimo_update"]) < 3.0: 
                personas_activas_en_esta_camara.add(info_tk["id_global"])

    mejor_id_global = None
    max_similitud = -1.0
    
    # 🧠 MATCHER INTELIGENTE POR CENTROIDE Y ÁLBUM
    for persona_id, datos in clientes_globales.items():
        if branch_id != datos.get("branch_id", branch_id): continue
        if zona != datos["zona_actual"] and (ahora - datos["timestamp"]) < TIEMPO_TELETRANSPORTACION: continue
        if persona_id in personas_activas_en_esta_camara: 
            continue

        # 1. Similitud contra cada foto individual del historial
        similitudes_album = [F.cosine_similarity(huella_nueva.unsqueeze(0), h.unsqueeze(0)).item() for h in datos["historial"]]
        max_sim_foto = max(similitudes_album) if similitudes_album else 0.0
        
        # 2. Similitud contra el Centroide (Promedio de identidad global de la persona)
        huella_centroid = torch.stack(datos["historial"]).mean(dim=0)
        huella_centroid = F.normalize(huella_centroid, p=2, dim=0)
        sim_centroid = F.cosine_similarity(huella_nueva.unsqueeze(0), huella_centroid.unsqueeze(0)).item()
        
        # Tomamos el mejor puntaje entre el centroide y las fotos individuales
        mejor_sim_persona = max(max_sim_foto, sim_centroid)
                
        if mejor_sim_persona > max_similitud:
            max_similitud = mejor_sim_persona
            mejor_id_global = persona_id
            
    # Asignación final basada en similitud mejorada
    if max_similitud >= UMBRAL_SIMILITUD:
        id_global = mejor_id_global
        album = clientes_globales[id_global]["historial"]
        if len(album) >= MAX_FOTOS_ALBUM:
            album.pop(0)
        album.append(huella_nueva)
        clientes_globales[id_global]["ultimo_update_album"] = ahora
    else:
        id_global = f"Cliente_Global_{contador_global_ids}"
        contador_global_ids += 1
        clientes_globales[id_global] = {
            "historial": [huella_nueva], "ultimo_update_album": ahora, 
            "hora_entrada": ahora, "zona_entrada": zona, "branch_id": branch_id 
        }
        
    if id_local_camara:
        traductor_camaras[id_local_camara] = {"id_global": id_global, "ultimo_update": ahora}

    clientes_globales[id_global].update({
        "foto_b64": foto_b64, "zona_actual": zona, "similitud": max_similitud if max_similitud >= 0 else 0,
        "timestamp": ahora, "hora_legible": time.strftime("%H:%M:%S"), "branch_id": branch_id 
    })
        
    return {"status": "ok", "id_asignado": id_global}


def procesar_y_enviar_supabase(pid, datos, tiempo_adentro):
    sucursal_final = datos.get("branch_id", "SUC-001")
    genero_calculado = "No definido"
    rango_edad_calculado = "No definido"
    
    if tiene_onnx and session_age_gender is not None:
        try:
            img_bytes = base64.b64decode(datos["foto_b64"])
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            rostros = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(rostros) > 0:
                x, y, w, h = rostros[0]
                imagen_ia = img_cv[y:y+h, x:x+w] 
            else:
                imagen_ia = img_cv
                
            img_rgb = cv2.cvtColor(imagen_ia, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224)) 
            img_float = img_resized.astype(np.float32) / 255.0
            
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_normalized = (img_float - mean) / std
            
            img_transpuesta = np.transpose(img_normalized, (2, 0, 1))
            input_onnx = np.expand_dims(img_transpuesta, axis=0)
            nombre_entrada = session_age_gender.get_inputs()[0].name

            salidas = session_age_gender.run(None, {nombre_entrada: input_onnx.astype(np.float32)})
            logits = salidas[0][0]
            
            edad_exacta = min(max(int(round(logits[0])), 0), 100)
            prob_genero = logits[1]
            genero_calculado = "Mujer" if prob_genero >= 0.5 else "Hombre"
            
            if edad_exacta < 18: rango_edad_calculado = "-18"
            elif 18 <= edad_exacta <= 25: rango_edad_calculado = "18-25"
            elif 26 <= edad_exacta <= 35: rango_edad_calculado = "26-35"
            elif 36 <= edad_exacta <= 45: rango_edad_calculado = "36-45"
            else: rango_edad_calculado = "46+"
        except Exception as e:
            pass
    
    payload = {
        "branch_id": sucursal_final,  
        "tracker_id": pid,
        "gender": genero_calculado,
        "age_range": rango_edad_calculado,
        "entered_at": datetime.fromtimestamp(datos["hora_entrada"], timezone.utc).isoformat(),
        "exited_at": datetime.fromtimestamp(datos["timestamp"], timezone.utc).isoformat(),
        "dwell_time_seconds": tiempo_adentro
    }
    
    try:
        requests.post(SUPABASE_URL, json=payload, headers=SUPABASE_HEADERS)
    except:
        pass


async def reloj_limpiador_background():
    while True:
        await asyncio.sleep(20)
        ahora = time.time()
        borrados = [pid for pid, d in clientes_globales.items() if ahora - d["timestamp"] > TIEMPO_INACTIVIDAD_SEGUNDOS]
        
        for pid in borrados:
            datos = clientes_globales[pid]
            tiempo_adentro = int(datos["timestamp"] - datos["hora_entrada"])
            if tiempo_adentro > 5:
                procesar_y_enviar_supabase(pid, datos, tiempo_adentro)
            del clientes_globales[pid]
            trackers_a_borrar = [k for k, v in traductor_camaras.items() if v["id_global"] == pid]
            for t in trackers_a_borrar: del traductor_camaras[t]

@app.on_event("startup")
async def iniciar_servicios():
    asyncio.create_task(reloj_limpiador_background())

@app.get("/api/clientes")
def obtener_clientes():
    return {"clientes": [{"id": p, "zona": d["zona_actual"], "similitud": f"{d['similitud']*100:.1f}%", "ultima_vista": d["hora_legible"], "foto": d["foto_b64"]} for p, d in clientes_globales.items()]}

@app.get("/api/heatmap")
def obtener_heatmap():
    return {"puntos": puntos_heatmap}

@app.get("/api/zonas_camara")
def proxy_zonas():
    try:
        res = requests.get(f"{IP_CAMARA}/config", timeout=2)
        data = res.json()
        
        zonas_enderezadas = []
        for zone in data.get("zones", []):
            poligono_nuevo = []
            for pt in zone.get("polygon", []):
                pto = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
                pto_trans = cv2.perspectiveTransform(pto, MATRIZ_HOMOGRAFIA)
                nx, ny = pto_trans[0][0]
                poligono_nuevo.append([int(nx), int(ny)])
            
            zone["polygon"] = poligono_nuevo
            zonas_enderezadas.append(zone)
            
        return {"zones": zonas_enderezadas}
    except Exception as e:
        return {"zones": []}

@app.post("/api/reset")
def resetear_memoria():
    global contador_global_ids, puntos_heatmap
    clientes_globales.clear()
    traductor_camaras.clear()
    puntos_heatmap.clear() 
    contador_global_ids = 1
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
def panel_web():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LeanVision | Cerebro Total</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/heatmap.js/2.0.2/heatmap.min.js"></script>
        <style>
            body { font-family: 'Segoe UI', sans-serif; background-color: #1e1e2f; color: white; padding: 20px; margin: 0;}
            h1 { text-align: center; color: #00ff88; margin-bottom: 5px; }
            .header-bar { display: flex; justify-content: space-between; align-items: center; max-width: 1200px; margin: 0 auto 20px auto; background-color: #2a2a40; padding: 12px 20px; border-radius: 8px; }
            .btn-reset { background-color: #ff5252; color: white; border: none; padding: 10px 18px; border-radius: 6px; font-weight: bold; cursor: pointer; transition: 0.2s; }
            
            .dashboard-layout { display: flex; gap: 30px; max-width: 1200px; margin: 0 auto; justify-content: center; }
            
            .clientes-section { flex: 1; max-width: 650px; }
            .grid { display: flex; flex-wrap: wrap; gap: 20px; justify-content: flex-start; }
            .card { background-color: #2a2a40; border-radius: 10px; padding: 15px; width: 200px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.3); }
            .card img { max-width: 100%; border-radius: 6px; height: 180px; object-fit: cover; }
            .tag-id { font-size: 1.15em; font-weight: bold; margin: 10px 0 5px 0; color: #ffeb3b; }
            .tag-zona { background-color: #00ff88; color: black; padding: 3px 10px; border-radius: 15px; font-size: 0.85em; font-weight: bold; display: inline-block; }
            .tag-info { font-size: 0.8em; color: #aaa; margin-top: 8px; }

            .heatmap-section { width: 480px; display: flex; flex-direction: column; align-items: center; }
            .heatmap-title { text-align: center; margin-bottom: 10px; color: #ffeb3b; }
            
            .map-container {
                position: relative;
                width: 480px;
                height: 640px;
                border-radius: 8px;
                overflow: hidden;
                border: 2px solid #334155;
                box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            }

            .orientacion {
                position: absolute;
                color: rgba(0, 255, 136, 0.6);
                font-size: 12px;
                font-weight: bold;
                letter-spacing: 2px;
                z-index: 10;
                pointer-events: none;
                text-shadow: 0px 2px 4px rgba(0,0,0,0.8);
            }
            .orientacion.top { top: 10px; width: 100%; text-align: center; }
            .orientacion.bottom { bottom: 10px; width: 100%; text-align: center; }

            #heatmap-canvas { 
                position: absolute;
                top: 0; left: 0;
                width: 100%; 
                height: 100%; 
                background-color: #0f172a; 
                background-image: 
                    linear-gradient(rgba(0, 255, 136, 0.15) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(0, 255, 136, 0.15) 1px, transparent 1px);
                background-size: 40px 40px; 
                z-index: 1;
            }
            
            #zonas-overlay {
                position: absolute;
                top: 0; left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 5;
            }
        </style>
    </head>
    <body>
        <h1>🦅 Cerebro Central Re-ID (OSNet-IBN Pro)</h1>
        <div class="header-bar">
            <span>📊 Check-Out automático en 60s | Puerto 8081</span>
            <button class="btn-reset" onclick="borrarMemoria()">🧹 Resetear Sistema</button>
        </div>
        
        <div class="dashboard-layout">
            <div class="clientes-section">
                <h3 style="color: #00ff88; margin-top:0;">👥 Clientes Detectados</h3>
                <div class="grid" id="contenedor-clientes"></div>
            </div>

            <div class="heatmap-section">
                <h3 class="heatmap-title">🔥 Mapa Físico Enderezado</h3>
                <div class="map-container">
                    <div class="orientacion top">↑ FONDO DEL LOCAL (LEJOS) ↑</div>
                    <div class="orientacion bottom">↓ CÁMARA (CERCA) ↓</div>
                    <div id="heatmap-canvas"></div>
                    <canvas id="zonas-overlay" width="480" height="640"></canvas>
                </div>
            </div>
        </div>

        <script>
            var heatmapInstance = h337.create({
                container: document.getElementById('heatmap-canvas'),
                radius: 40,
                maxOpacity: .7,
                minOpacity: 0,
                blur: .8,
                gradient: { '.3': 'blue', '.5': 'green', '.8': 'yellow', '1': 'red' }
            });

            async function actualizarPanel() {
                try {
                    const response = await fetch('/api/clientes');
                    const data = await response.json();
                    const contenedor = document.getElementById('contenedor-clientes');
                    contenedor.innerHTML = ''; 
                    data.clientes.forEach(cliente => {
                        const div = document.createElement('div');
                        div.className = 'card';
                        div.innerHTML = `
                            <img src="data:image/jpeg;base64,${cliente.foto}">
                            <div class="tag-id">${cliente.id}</div>
                            <span class="tag-zona">📍 ${cliente.zona}</span>
                            <div class="tag-info">⏱️ Última vez: ${cliente.ultima_vista}</div>
                            <div class="tag-info">🧬 Similitud: ${cliente.similitud}</div>
                        `;
                        contenedor.appendChild(div);
                    });
                } catch (e) {}
            }

            async function actualizarHeatmap() {
                try {
                    const res = await fetch('/api/heatmap');
                    const json = await res.json();
                    const data = { max: 15, data: json.puntos };
                    heatmapInstance.setData(data);
                } catch(e) {}
            }
            
            async function dibujarZonas() {
                try {
                    const res = await fetch('/api/zonas_camara');
                    const data = await res.json();
                    
                    const canvas = document.getElementById('zonas-overlay');
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height); 
                    
                    if (data.zones && data.zones.length > 0) {
                        data.zones.forEach(zone => {
                            if (zone.polygon && zone.polygon.length > 2) {
                                ctx.beginPath();
                                ctx.moveTo(zone.polygon[0][0], zone.polygon[0][1]);
                                for (let i = 1; i < zone.polygon.length; i++) {
                                    ctx.lineTo(zone.polygon[i][0], zone.polygon[i][1]);
                                }
                                ctx.closePath();
                                
                                ctx.lineWidth = 2;
                                ctx.strokeStyle = zone.color;
                                ctx.stroke();
                                
                                ctx.fillStyle = zone.color + '40'; 
                                ctx.fill();
                                
                                ctx.fillStyle = 'white';
                                ctx.font = 'bold 15px Arial';
                                ctx.shadowColor = 'black';
                                ctx.shadowBlur = 5;
                                ctx.fillText(zone.name, zone.polygon[0][0] + 5, zone.polygon[0][1] - 5);
                                ctx.shadowBlur = 0;
                            }
                        });
                    }
                } catch (e) {}
            }

            async function borrarMemoria() {
                if (confirm("¿Borrar historial global y mapa de calor?")) { 
                    await fetch('/api/reset', { method: 'POST' }); 
                    actualizarPanel(); 
                    actualizarHeatmap();
                }
            }

            setInterval(actualizarPanel, 1500);
            setInterval(actualizarHeatmap, 1500);
            setInterval(dibujarZonas, 5000); 
            
            actualizarPanel();
            actualizarHeatmap();
            dibujarZonas();
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)