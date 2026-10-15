"""OSNet en NCNN tiene que dar las mismas huellas que en PyTorch, o volver a PyTorch."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("ncnn")
pytest.importorskip("pnnx")

from reid_ncnn import ExtractorNCNN, exportar_si_falta, huella_de_pesos
from torchreid.utils import FeatureExtractor


@pytest.fixture(scope="module")
def extractor_torch():
    """Un FeatureExtractor con osnet_x0_25 sin pesos descargados: rápido y offline."""
    import torchvision.transforms as T
    from torchreid import models

    torch.manual_seed(0)
    extractor = FeatureExtractor.__new__(FeatureExtractor)
    extractor.model = models.build_model(name="osnet_x0_25", num_classes=1, pretrained=False).eval()
    extractor.preprocess = T.Compose([
        T.Resize((256, 128)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    extractor.to_pil = T.ToPILImage()
    extractor.device = torch.device("cpu")
    return extractor


@pytest.fixture(scope="module")
def archivos(extractor_torch, tmp_path_factory):
    return exportar_si_falta(extractor_torch.model, tmp_path_factory.mktemp("ncnn"), "osnet_x0_25", (256, 128))


def _imagenes(n):
    rng = np.random.default_rng(1)
    return [rng.integers(0, 255, (180 + 10 * i, 70, 3), dtype=np.uint8) for i in range(n)]


def test_las_huellas_coinciden_con_pytorch(extractor_torch, archivos):
    ncnn = ExtractorNCNN(extractor_torch, *archivos, hilos=2, muestras_validacion=5)
    imagenes = _imagenes(5)
    huellas = ncnn(imagenes)
    referencia = extractor_torch(imagenes)
    assert huellas.shape == referencia.shape
    assert float(torch.nn.functional.cosine_similarity(huellas, referencia).min()) > 0.999
    assert ncnn.motor == "ncnn"
    estado = ncnn.estado()
    assert estado["validadas"] == 5 and not estado["validando"] and estado["coseno_minimo"] > 0.999


def test_no_vuelve_a_exportar_si_ya_existe(extractor_torch, archivos):
    param, _ = archivos
    assert exportar_si_falta(extractor_torch.model, param.parent, "osnet_x0_25", (256, 128)) == archivos
    assert huella_de_pesos(extractor_torch.model) in param.name


def test_si_las_huellas_no_coinciden_vuelve_a_pytorch(extractor_torch, archivos):
    ncnn = ExtractorNCNN(extractor_torch, *archivos, hilos=2)
    ncnn._una = lambda entrada: np.random.default_rng(0).standard_normal(512).astype(np.float32)
    imagenes = _imagenes(2)
    huellas = ncnn(imagenes)
    assert ncnn.motor == "torch"
    assert torch.allclose(huellas, extractor_torch(imagenes), atol=1e-5)
