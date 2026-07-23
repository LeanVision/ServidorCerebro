import cv2
import numpy as np
import time
import base64
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
import uvicorn
import requests

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# --- CONEXIÓN DIRECTA A SUPABASE (Sin la librería problemática) ---
SUPABASE_URL = "https://butoxtgngmbnkmgueavf.supabase.co/rest/v1/visitor_sessions"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ1dG94dGduZ21ibmttZ3VlYXZmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODMxNjUwODcsImV4cCI6MjA5ODc0MTA4N30.nFYb_11mK353SWQMCNQIjM3IcbhIrpbD9M59iMHWkaM"
SUPABASE_HEADERS = {
    "apikey": SUPABASE_KEY, 
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json", 
    "Prefer": "return=representation"
}

app = FastAPI(title="Cerebro Central Re-ID + Supabase")

print("🧠 Cargando modelo IA (ResNet50)...")
pesos = models.ResNet50_Weights.DEFAULT
modelo_ia = models.resnet50(weights=pesos)
modelo_ia = torch.nn.Sequential(*(list(modelo_ia.children())[:-1]))
modelo_ia.eval()

transformaciones = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print("✅ ¡Modelo IA listo!")

UMBRAL_SIMILITUD = 0.40       
MAX_FOTOS_ALBUM = 5           
TIEMPO_TELETRANSPORTACION = 3 
TIEMPO_EXPIRACION_SEGUNDOS = 120 # 2 minutos para el check-out

memoria_reid = {}         
mapeo_trackers = {}       
contador_ids = 1

@app.post("/identificar")
async def identificar_persona(
    file: UploadFile = File(...), 
    zona: str = Form("Desconocida"),
    tracker_id: str = Form(None),
    camara_id: str = Form("camara_default") # 🟢 Recibimos el origen
):
    global contador_ids
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    imagen_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    imagen_rgb = cv2.cvtColor(imagen_cv2, cv2.COLOR_BGR2RGB)
    imagen_pil = Image.fromarray(imagen_rgb)
    foto_b64 = base64.b64encode(contents).decode('utf-8')
    
    lote_entrada = transformaciones(imagen_pil).unsqueeze(0)
    with torch.no_grad():
        huella_bruta = modelo_ia(lote_entrada).flatten()
        huella_nueva = F.normalize(huella_bruta, p=2, dim=0)

    ahora = time.time()

    # 🟢 Armamos una llave única combinando la cámara y su tracker local
    llave_tracker_unico = f"{camara_id}_{tracker_id}" if tracker_id else None

    if llave_tracker_unico and llave_tracker_unico in mapeo_trackers:
        id_final = mapeo_trackers[llave_tracker_unico]
        if id_final in memoria_reid:
            if ahora - memoria_reid[id_final].get("ultimo_update_album", 0) > 3.0:
                album = memoria_reid[id_final]["historial"]
                if len(album) >= MAX_FOTOS_ALBUM:
                    album.pop(0) 
                album.append(huella_nueva)
                memoria_reid[id_final]["ultimo_update_album"] = ahora
            
            memoria_reid[id_final].update({
                "foto_b64": foto_b64,
                "zona_actual": zona,
                "timestamp": ahora,
                "hora_legible": time.strftime("%H:%M:%S")
            })
            return {"status": "ok", "id_asignado": id_final}

    # Búsqueda matemática de similitud habitual...
    mejor_id = None
    max_similitud = -1.0
    
    for persona_id, datos in memoria_reid.items():
        if zona != datos["zona_actual"] and (ahora - datos["timestamp"]) < TIEMPO_TELETRANSPORTACION:
            continue

        similitudes = [F.cosine_similarity(huella_nueva.unsqueeze(0), h.unsqueeze(0)).item() for h in datos["historial"]]
        mejor_sim_album = max(similitudes)
        
        if mejor_sim_album > max_similitud:
            max_similitud = mejor_sim_album
            mejor_id = persona_id
            
    if max_similitud >= UMBRAL_SIMILITUD:
        id_final = mejor_id
        if len(memoria_reid[id_final]["historial"]) >= MAX_FOTOS_ALBUM:
            memoria_reid[id_final]["historial"].pop(0)
        memoria_reid[id_final]["historial"].append(huella_nueva)
        memoria_reid[id_final]["ultimo_update_album"] = ahora
        mensaje = f"¡Reconocido! Es {id_final}"
    else:
        id_final = f"Cliente_{contador_ids}"
        contador_ids += 1
        mensaje = f"Nueva persona: {id_final}"
        
        memoria_reid[id_final] = {
            "historial": [huella_nueva],
            "ultimo_update_album": ahora,
            "hora_entrada": ahora,
            "zona_entrada": zona
        }
        
    if llave_tracker_unico:
        mapeo_trackers[llave_tracker_unico] = id_final

    memoria_reid[id_final].update({
        "foto_b64": foto_b64,
        "zona_actual": zona,
        "similitud": max_similitud if max_similitud >= 0 else 0,
        "timestamp": ahora,
        "hora_legible": time.strftime("%H:%M:%S")
    })
        
    print(f"🔍 {mensaje} desde [{camara_id}] en {zona}")
    return {"status": "ok", "id_asignado": id_final}

# --- CHECK-OUT Y ENVÍO A SUPABASE POR HTTP DIRECTO ---
def limpiar_memoria_inactiva():
    ahora = time.time()
    borrados = [pid for pid, d in memoria_reid.items() if ahora - d["timestamp"] > TIEMPO_EXPIRACION_SEGUNDOS]
    
    for pid in borrados:
        datos = memoria_reid[pid]
        tiempo_adentro = int(datos["timestamp"] - datos["hora_entrada"])
        
        # Solo mandamos a la BD si estuvo más de 5 segundos
        if tiempo_adentro > 5:
            print(f"🚀 Enviando a {pid} a Supabase... (Tiempo total: {tiempo_adentro}s)")
            
            payload = {
                "branch_id": "SUC-001", 
                "tracker_id": pid,
                "gender": "No definido",
                "age_range": "No definido",
                "entered_at": datetime.fromtimestamp(datos["hora_entrada"], timezone.utc).isoformat(),
                "exited_at": datetime.fromtimestamp(datos["timestamp"], timezone.utc).isoformat(),
                "dwell_time_seconds": tiempo_adentro
            }
            
            try:
                # Enviamos el paquete por requests, libre de errores de path
                respuesta = requests.post(SUPABASE_URL, json=payload, headers=SUPABASE_HEADERS)
                
                if respuesta.status_code in [200, 201]:
                    print("✅ Sesión guardada en la nube con éxito.")
                else:
                    print(f"⚠️ Rechazo de Supabase: {respuesta.text}")
            except Exception as e:
                print(f"⚠️ Error de red enviando a Supabase: {e}")

        del memoria_reid[pid]
        trackers_a_borrar = [k for k, v in mapeo_trackers.items() if v == pid]
        for t in trackers_a_borrar: del mapeo_trackers[t]

@app.get("/api/clientes")
def obtener_clientes():
    limpiar_memoria_inactiva()
    clientes = [{"id": p, "zona": d["zona_actual"], "similitud": f"{d['similitud']*100:.1f}%", "ultima_vista": d["hora_legible"], "foto": d["foto_b64"]} for p, d in memoria_reid.items()]
    return {"clientes": clientes}

@app.post("/api/reset")
def resetear_memoria():
    global contador_ids
    memoria_reid.clear()
    mapeo_trackers.clear()
    contador_ids = 1
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
def panel_web():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LeanVision | Cerebro Cloud</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; background-color: #1e1e2f; color: white; padding: 20px; }
            h1 { text-align: center; color: #00ff88; margin-bottom: 5px; }
            .header-bar { display: flex; justify-content: space-between; align-items: center; max-width: 900px; margin: 0 auto 20px auto; background-color: #2a2a40; padding: 12px 20px; border-radius: 8px; }
            .btn-reset { background-color: #ff5252; color: white; border: none; padding: 10px 18px; border-radius: 6px; font-weight: bold; cursor: pointer; transition: 0.2s; }
            .btn-reset:hover { background-color: #ff1744; transform: scale(1.03); }
            .grid { display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }
            .card { background-color: #2a2a40; border-radius: 10px; padding: 15px; width: 210px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.3); transition: 0.3s; }
            .card img { max-width: 100%; border-radius: 6px; height: 180px; object-fit: cover; }
            .tag-id { font-size: 1.15em; font-weight: bold; margin: 10px 0 5px 0; color: #ffeb3b; }
            .tag-zona { background-color: #00ff88; color: black; padding: 3px 10px; border-radius: 15px; font-size: 0.85em; font-weight: bold; display: inline-block; }
            .tag-info { font-size: 0.8em; color: #aaa; margin-top: 8px; }
        </style>
    </head>
    <body>
        <h1>🧠 Cerebro Central Re-ID + Supabase</h1>
        <div class="header-bar">
            <span>📊 Conectado por HTTP Directo | Puerto 8081</span>
            <button class="btn-reset" onclick="borrarMemoria()">🧹 Resetear Sistema</button>
        </div>
        <div class="grid" id="contenedor-clientes"></div>
        <script>
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
            async function borrarMemoria() {
                if (confirm("¿Borrar historial?")) { await fetch('/api/reset', { method: 'POST' }); actualizarPanel(); }
            }
            setInterval(actualizarPanel, 1500);
            actualizarPanel();
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    # PUERTO CAMBIADO AL 8081 PARA NO CHOCAR CON REACT
    uvicorn.run(app, host="0.0.0.0", port=8081)