"""OSNet corriendo con NCNN en vez de PyTorch, con las mismas huellas.

NCNN es bastante más rápido en la CPU de la Raspberry Pi (lo mismo que se hizo
con el detector de EPP). El modelo no viaja en el repo: se exporta en la Pi,
desde los mismos pesos que carga PyTorch, la primera vez que arranca con
CEREBRO_REID_MOTOR=ncnn, y queda guardado para los arranques siguientes.

Los umbrales de similitud se calibraron con las huellas de PyTorch, así que
no alcanza con que "ande": durante los primeros recortes reales se calculan
las dos huellas y, si alguna difiere, se vuelve a PyTorch solo y se avisa en
el log y en /health. Por lo mismo NCNN corre en fp32: con fp16 (lo que usaría
por defecto en la Pi 5) las huellas cambian un poco.

Necesita los paquetes de pip ncnn y pnnx, que no están en requirements.txt a
propósito: si en alguna Pi no hubiera wheel, el auto-pull no se rompe y el
Cerebro sigue con PyTorch.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import shutil
import tempfile
import threading
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Coseno mínimo entre la huella de NCNN y la de PyTorch para el mismo recorte.
# En las pruebas da 1,0000; cualquier cosa por debajo de esto es un problema
# de conversión, no ruido numérico.
COSENO_MINIMO = 0.999


def huella_de_pesos(modelo: torch.nn.Module) -> str:
    """Identifica los pesos: si cambian, el modelo exportado ya no sirve."""
    digest = hashlib.sha1()
    for nombre, tensor in modelo.state_dict().items():
        digest.update(nombre.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:12]


def exportar_si_falta(
    modelo: torch.nn.Module, carpeta: Path, nombre: str, tamano: tuple[int, int]
) -> tuple[Path, Path]:
    """Devuelve (param, bin) del modelo en NCNN, exportándolo si no existe."""
    base = carpeta / f"{nombre}-{huella_de_pesos(modelo)}"
    param, pesos = base.with_suffix(".param"), base.with_suffix(".bin")
    if param.exists() and pesos.exists():
        return param, pesos

    import pnnx  # solo hace falta para exportar

    carpeta.mkdir(parents=True, exist_ok=True)
    logger.info("Exportando %s a NCNN (una sola vez, puede tardar un minuto)...", nombre)
    with tempfile.TemporaryDirectory() as temporal:
        destino = Path(temporal) / "osnet.pt"
        # Una copia: el modelo de PyTorch sigue en uso para validar.
        pnnx.export(copy.deepcopy(modelo).eval(), str(destino), (torch.zeros(1, 3, *tamano),))
        # Se mueven los dos juntos y el .param al final: un .param sin su .bin
        # al lado nunca queda a medio escribir.
        shutil.move(str(destino.with_suffix(".ncnn.bin")), pesos)
        shutil.move(str(destino.with_suffix(".ncnn.param")), param)
    logger.info("Modelo NCNN guardado en %s", param)
    return param, pesos


class ExtractorNCNN:
    """Misma interfaz que el FeatureExtractor de torchreid: lista de imágenes RGB
    (np.ndarray) -> tensor (N, 512). Usa su mismo preprocesado."""

    def __init__(
        self,
        extractor_torch,
        param: Path,
        pesos: Path,
        hilos: int,
        muestras_validacion: int = 50,
    ):
        import ncnn

        self._torch = extractor_torch
        self._ncnn = ncnn
        red = ncnn.Net()
        red.opt.num_threads = max(1, hilos)
        # fp32 en todo: las huellas tienen que dar iguales a las de PyTorch.
        red.opt.use_fp16_storage = False
        red.opt.use_fp16_packed = False
        red.opt.use_fp16_arithmetic = False
        red.opt.use_bf16_storage = False
        if red.load_param(str(param)) != 0 or red.load_model(str(pesos)) != 0:
            raise RuntimeError(f"No se pudo cargar el modelo NCNN de {param}")
        self._red = red
        self._lock = threading.Lock()
        self.pendientes_validacion = muestras_validacion
        self.validadas = 0
        self.coseno_minimo: float | None = None
        self.desactivado = False

    @property
    def motor(self) -> str:
        return "torch" if self.desactivado else "ncnn"

    def _preprocesar(self, imagenes: list[np.ndarray]) -> torch.Tensor:
        return torch.stack([self._torch.preprocess(self._torch.to_pil(img)) for img in imagenes])

    def _una(self, entrada: torch.Tensor) -> np.ndarray:
        # El Mat de NCNN apunta a la memoria del array sin copiarla: el array
        # tiene que seguir vivo hasta después de extract().
        datos = np.ascontiguousarray(entrada.numpy(), dtype=np.float32)
        extractor = self._red.create_extractor()
        extractor.input("in0", self._ncnn.Mat(datos))
        _, salida = extractor.extract("out0")
        return np.array(salida, dtype=np.float32).reshape(-1)

    def __call__(self, imagenes: list[np.ndarray]) -> torch.Tensor:
        if self.desactivado:
            return self._torch(imagenes)
        entradas = self._preprocesar(imagenes)
        huellas = torch.from_numpy(np.stack([self._una(x) for x in entradas]))
        if self.pendientes_validacion > 0:
            self._validar(entradas, huellas)
            if self.desactivado:
                return self._torch(entradas)
        return huellas

    def _validar(self, entradas: torch.Tensor, huellas: torch.Tensor) -> None:
        with torch.inference_mode():
            referencia = self._torch.model(entradas)
        cosenos = torch.nn.functional.cosine_similarity(huellas, referencia, dim=1)
        peor = float(cosenos.min())
        with self._lock:
            self.validadas += len(entradas)
            self.pendientes_validacion -= len(entradas)
            if self.coseno_minimo is None or peor < self.coseno_minimo:
                self.coseno_minimo = peor
            if peor < COSENO_MINIMO:
                self.desactivado = True
                logger.error(
                    "Las huellas de NCNN no coinciden con las de PyTorch (coseno %.4f). "
                    "Se vuelve a PyTorch.", peor,
                )
            elif self.pendientes_validacion <= 0:
                logger.info(
                    "NCNN validado contra PyTorch en %d recortes (coseno mínimo %.5f).",
                    self.validadas, self.coseno_minimo,
                )

    def estado(self) -> dict:
        with self._lock:
            return {
                "validadas": self.validadas,
                "validando": self.pendientes_validacion > 0 and not self.desactivado,
                "coseno_minimo": None if self.coseno_minimo is None else round(self.coseno_minimo, 5),
            }
