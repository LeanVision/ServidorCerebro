# Ergonomía con visión artificial (prototipo)

Asistente para el **Protocolo de Ergonomía de la Resolución SRT 886/2015**. A
partir de un video del puesto de trabajo, estima la postura del trabajador
(su "esqueleto") y arma un **borrador de evidencias** para las planillas:

- **2.F Posturas forzadas:** % del tiempo con el tronco, el cuello, los brazos, la muñeca y las rodillas en posturas de riesgo, más el puntaje **REBA** por cuadro.
- **2.E Movimientos repetitivos:** ciclos por minuto, duración del ciclo, % del tiempo en movimiento y un **Nivel de Actividad Manual estimado** (ecuación de Radwin, 2015).
- **2.D Bipedestación:** tiempo de pie.

La norma pide que la evaluación la haga un profesional con conocimientos en
ergonomía: el informe trae las mediciones y el espacio para que ese
profesional las revise, complete (carga, fuerza percibida, duración de la
tarea en la jornada) y firme.

## Cómo filmar

- **Cámara de costado al trabajador**, a la altura de la cintura o el pecho, con todo el cuerpo en cuadro. De frente los ángulos salen más chicos de lo que son.
- Una sola persona en el puesto (si hay varias se toma la más grande en la imagen).
- Varios ciclos completos de la tarea. Con un celular alcanza.

## Uso

```bash
pip install -r ergonomia/requirements.txt
python -m ergonomia.analizar video.mp4 --puesto "Embalaje línea 2" --carga-kg 8 --agarre regular
```

Deja en `video_ergonomia/`:

- `informe.md`: el informe para leer.
- `informe.json`: los mismos datos, para procesarlos.
- `angulos.csv`: los ángulos cuadro por cuadro.
- `esqueleto.mp4`, sólo con `--video-esqueleto`: el esqueleto sobre fondo negro, para mostrar el análisis sin mostrar a la persona.

Opciones útiles: `--lado izq|der` si la detección automática del lado que mira a la cámara falla, `--golpes`, `--un-pie`, `--cuadros-por-segundo`.

### Modelo de pose

- `--modelo yolo` (por defecto): Ultralytics YOLO-pose. Baja los pesos solo la primera vez. No detecta los dedos, así que la muñeca no se mide.
- `--modelo mediapipe --ruta-modelo pose_landmarker_full.task`: MediaPipe Pose Landmarker. Sí mide la muñeca. El archivo se baja de la [página de MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models).

Todo corre local: el video no sale de la máquina y sus cuadros no se guardan.

## Limitaciones

- Ángulos en 2D: no se miden giros ni inclinaciones laterales, abducción del brazo, hombro elevado ni desvío lateral de la muñeca. Esas partes del REBA no suman.
- El peso de la carga, el agarre y la fuerza percibida (escala de Borg) los informa quien evalúa.
- Factores que una cámara no mide (vibraciones, confort térmico) quedan fuera.
- Antes de usarlo en una evaluación real hay que **validarlo** contra la evaluación manual de un ergónomo en algunos puestos.

## Privacidad

Filmar trabajadores es tratamiento de datos personales (Ley 25.326) y un
control del empleador (art. 70 de la Ley de Contrato de Trabajo): avisar a
los trabajadores, filmar sólo lo necesario y borrar el video original después
del análisis. El informe y el CSV no contienen imágenes.

## Pruebas

```bash
python -m pytest ergonomia/tests
```

Usan esqueletos sintéticos con ángulos conocidos y un modelo de pose falso,
así que no necesitan bajar pesos.
