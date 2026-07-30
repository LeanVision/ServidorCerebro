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

app = FastAPI(title="Cerebro Central Director + Web Dashboard (ONNX)")

print("🧠 Cargando modelo IA Re-ID (OSNet Omni-Scale)...")
extractor_ia = FeatureExtractor(model_name='osnet_x1_0', device='cpu')
print("✅ ¡Modelo OSNet listo!")

# 🟢 CARGA DEL MODELO DEMOGRÁFICO
print("🧠 Buscando modelo ONNX de Edad/Género...")
try:
    session_age_gender = ort.InferenceSession("demografia.onnx")
    tiene_onnx = True
    print("✅ ¡Modelo ONNX cargado correctamente!")
except Exception as e:
    session_age_gender = None
    tiene_onnx = False
    print("⚠️ Archivo 'demografia.onnx' no encontrado. Se enviará 'No definido' temporalmente.")

# --- 🛠️ AJUSTES DEL DIRECTOR DE ORQUESTA ---
UMBRAL_SIMILITUD = 0.32       
MAX_FOTOS_ALBUM = 5           
TIEMPO_TELETRANSPORTACION = 3 
TIEMPO_INACTIVIDAD_SEGUNDOS = 60 # 🟢 1 MINUTO PARA PRUEBA

clientes_globales = {}  
traductor_camaras = {}  
contador_global_ids = 1

@app.post("/identificar")
async def identificar_persona(
    file: UploadFile = File(...), 
    zona: str = Form("Desconocida"),
    tracker_id: str = Form(None),
    camara_id: str = Form("camara_default"),
    branch_id: str = Form("SUC-001") 
):
    global contador_global_ids
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    imagen_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    imagen_rgb = cv2.cvtColor(imagen_cv2, cv2.COLOR_BGR2RGB)
    foto_b64 = base64.b64encode(contents).decode('utf-8')
    
    with torch.no_grad():
        huellas_batch = extractor_ia([imagen_rgb]) 
        huella_bruta = huellas_batch[0]            
        huella_nueva = F.normalize(huella_bruta, p=2, dim=0)

    ahora = time.time()
    id_local_camara = f"{camara_id}_{tracker_id}" if tracker_id else None

    if id_local_camara and id_local_camara in traductor_camaras:
        id_global = traductor_camaras[id_local_camara]["id_global"]
        if id_global in clientes_globales:
            if ahora - clientes_globales[id_global].get("ultimo_update_album", 0) > 1.5:
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
            if (ahora - info_tk["ultimo_update"]) < 0.8:
                personas_activas_en_esta_camara.add(info_tk["id_global"])

    mejor_id_global = None
    max_similitud = -1.0
    
    for persona_id, datos in clientes_globales.items():
        if branch_id != datos.get("branch_id", branch_id): continue
        if zona != datos["zona_actual"] and (ahora - datos["timestamp"]) < TIEMPO_TELETRANSPORTACION: continue

        similitudes = [F.cosine_similarity(huella_nueva.unsqueeze(0), h.unsqueeze(0)).item() for h in datos["historial"]]
        mejor_sim_album = max(similitudes)
        
        if persona_id in personas_activas_en_esta_camara and mejor_sim_album < 0.45: continue
                
        if mejor_sim_album > max_similitud:
            max_similitud = mejor_sim_album
            mejor_id_global = persona_id
            
    if max_similitud >= UMBRAL_SIMILITUD:
        id_global = mejor_id_global
        if len(clientes_globales[id_global]["historial"]) >= MAX_FOTOS_ALBUM:
            clientes_globales[id_global]["historial"].pop(0)
        clientes_globales[id_global]["historial"].append(huella_nueva)
        clientes_globales[id_global]["ultimo_update_album"] = ahora
        mensaje = f"Fusión Multicámara: {id_local_camara} ahora es {id_global} (Sim: {max_similitud*100:.1f}%)"
    else:
        id_global = f"Cliente_Global_{contador_global_ids}"
        contador_global_ids += 1
        mensaje = f"Nuevo Ingreso: {id_global} (Mejor similitud: {max_similitud*100:.1f}%)"
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
        
    print(f"🔍 {mensaje} desde [{camara_id}]")
    return {"status": "ok", "id_asignado": id_global}


# --- 🟢 EL CHECK-OUT CON ROSTRO Y ONNX ---
def procesar_y_enviar_supabase(pid, datos, tiempo_adentro):
    sucursal_final = datos.get("branch_id", "SUC-001")
    print(f"🚀 [CHECK-OUT] Procesando a {pid} por inactividad. (Tiempo total: {tiempo_adentro}s)")
    
    genero_calculado = "No definido"
    rango_edad_calculado = "No definido"
    
    if tiene_onnx and session_age_gender is not None:
        try:
            print(f"🧬 Buscando rostro en la foto para analizar demografía...")
            
            img_bytes = base64.b64decode(datos["foto_b64"])
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # --- 🕵️ NUEVO: DETECTOR DE CARAS OPENCV ---
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            # Escanea la foto buscando patrones de rostros
            rostros = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(rostros) > 0:
                print("   ↳ 👤 ¡Cara encontrada! Recortando para máxima precisión ONNX...")
                x, y, w, h = rostros[0]
                imagen_ia = img_cv[y:y+h, x:x+w] # Se queda solo con el cuadrado de la cara
            else:
                print("   ↳ ⚠️ Cara no visible. ONNX usará el cuerpo entero como respaldo.")
                imagen_ia = img_cv
                
            img_rgb = cv2.cvtColor(imagen_ia, cv2.COLOR_BGR2RGB)
            
            # Pasamos la cara recortada a ONNX
            img_resized = cv2.resize(img_rgb, (128, 256)) 
            img_float = img_resized.astype(np.float32) / 255.0
            img_transpuesta = np.transpose(img_float, (2, 0, 1))
            input_onnx = np.expand_dims(img_transpuesta, axis=0)

            salidas = session_age_gender.run(None, {"input": input_onnx})
            prediccion_genero = salidas[0][0] 
            prediccion_edad = salidas[1][0]   
            
            generos_texto = ["Hombre", "Mujer"]
            edades_texto = ["18-25", "26-35", "36-45", "46+"]
            
            genero_calculado = generos_texto[np.argmax(prediccion_genero)]
            rango_edad_calculado = edades_texto[np.argmax(prediccion_edad)]
            print(f"   ↳ Resultado ONNX: {genero_calculado}, {rango_edad_calculado}")
            
        except Exception as e:
            print(f"⚠️ Error al procesar ONNX: {e}")
    
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
        respuesta = requests.post(SUPABASE_URL, json=payload, headers=SUPABASE_HEADERS)
        if respuesta.status_code in [200, 201]:
            print(f"✅ {pid} guardado en Supabase con éxito.")
        else:
            print(f"⚠️ Rechazo de Supabase: {respuesta.text}")
    except Exception as e:
        print(f"⚠️ Error enviando a Supabase: {e}")

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
    print(f"⏱️ Reloj activado. Expulsará clientes tras {TIEMPO_INACTIVIDAD_SEGUNDOS}s sin ser vistos.")

# --- RUTAS WEB ---
@app.get("/api/clientes")
def obtener_clientes():
    return {"clientes": [{"id": p, "zona": d["zona_actual"], "similitud": f"{d['similitud']*100:.1f}%", "ultima_vista": d["hora_legible"], "foto": d["foto_b64"]} for p, d in clientes_globales.items()]}

@app.post("/api/reset")
def resetear_memoria():
    global contador_global_ids
    clientes_globales.clear()
    traductor_camaras.clear()
    contador_global_ids = 1
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
def panel_web():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LeanVision | Cerebro Total</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; background-color: #1e1e2f; color: white; padding: 20px; }
            h1 { text-align: center; color: #00ff88; margin-bottom: 5px; }
            .header-bar { display: flex; justify-content: space-between; align-items: center; max-width: 900px; margin: 0 auto 20px auto; background-color: #2a2a40; padding: 12px 20px; border-radius: 8px; }
            .btn-reset { background-color: #ff5252; color: white; border: none; padding: 10px 18px; border-radius: 6px; font-weight: bold; cursor: pointer; transition: 0.2s; }
            .grid { display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }
            .card { background-color: #2a2a40; border-radius: 10px; padding: 15px; width: 210px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.3); }
            .card img { max-width: 100%; border-radius: 6px; height: 180px; object-fit: cover; }
            .tag-id { font-size: 1.15em; font-weight: bold; margin: 10px 0 5px 0; color: #ffeb3b; }
            .tag-zona { background-color: #00ff88; color: black; padding: 3px 10px; border-radius: 15px; font-size: 0.85em; font-weight: bold; display: inline-block; }
            .tag-info { font-size: 0.8em; color: #aaa; margin-top: 8px; }
        </style>
    </head>
    <body>
        <h1>🧠 Cerebro Central Re-ID (ONNX)</h1>
        <div class="header-bar">
            <span>📊 Check-Out automático en 60s | Puerto 8081</span>
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
                if (confirm("¿Borrar historial global?")) { await fetch('/api/reset', { method: 'POST' }); actualizarPanel(); }
            }
            setInterval(actualizarPanel, 1500);
            actualizarPanel();
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)