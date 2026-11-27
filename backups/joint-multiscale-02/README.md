# Detector de referencia de cuatro clases

Época seleccionada 18 de 24 mediante mAP50 de validación. El entrenamiento completó 9.792 actualizaciones y utilizó únicamente pesos del proyecto.

| Medida | Resultado |
|---|---:|
| mAP50 de cuatro clases en 5 fotos de validación | 81,43 % |
| mAP50 de cuatro clases en 6 fotos de reserva revisadas | 86,89 % |
| Cabezas sin casco detectadas en 345 fotos de prueba | 245 de 383 |
| Falsos positivos sin casco con confianza e IoU de 0,5 | 52 |
| Precisión sin casco | 82,49 % |
| Sensibilidad sin casco | 63,97 % |

Desde la raíz de la rama, con el entorno del proyecto, el checkpoint se carga así:

```python
import sys
import torch
sys.path.insert(0, "training")
from models import build_detector

model = build_detector(5, 8, 4)
checkpoint = torch.load("backups/joint-multiscale-02/best.pt", map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model"])
model.eval()
```

Para reproducir la geometría de inferencia, convertir la imagen a RGB, reducir proporcionalmente el lado mayor a un máximo de 640 píxeles y pasar un tensor CHW con valores entre cero y uno. Las etiquetas de salida 1 a 4 corresponden a motocicleta, ocupante, cabeza con casco y cabeza sin casco.
