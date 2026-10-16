import sys
from pathlib import Path

# Raíz del repo, para importar el paquete ergonomia y el módulo esqueletos.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
