import torch
import torch.nn as nn

print("Generando modelo ONNX estructural...")

class ModeloDemografia(nn.Module):
    def __init__(self):
        super().__init__()
        # Red neuronal súper básica para dejar armada la tubería
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        # Salidas: 2 géneros (Hombre/Mujer) y 4 rangos de edad
        self.clasificador_genero = nn.Linear(16, 2) 
        self.clasificador_edad = nn.Linear(16, 4)   

    def forward(self, x):
        x = self.features(x)
        genero = self.clasificador_genero(x)
        edad = self.clasificador_edad(x)
        return genero, edad

modelo = ModeloDemografia()
modelo.eval()

# Simulamos una foto del mismo tamaño que usa tu Cerebro (256x128)
dummy_input = torch.randn(1, 3, 256, 128) 

# Exportamos el archivo mágico
torch.onnx.export(modelo, dummy_input, "demografia.onnx", 
                  input_names=['input'], output_names=['genero', 'edad'])

print("✅ Archivo 'demografia.onnx' creado con éxito en tu carpeta.")