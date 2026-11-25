# Detector de cuatro clases de referencia

El checkpoint evaluado corresponde a la última actualización (1950) de detector-clean-full-01. En las cinco imágenes de validación obtuvo 0,744912 de mAP50. Los pesos provienen del preentrenamiento propio con EdgeVision limpio y del ajuste con 70 fotos revisadas.

Para cargar los pesos desde la raíz del repositorio con el entorno del proyecto:

```python
import sys
import torch
sys.path.insert(0, "training")
from models import build_detector
ck = torch.load("backups/detector-clean-full-01/detector.pt", map_location="cpu", weights_only=True)
model = build_detector(5, 8, 4, deterministic_roi=True)
model.load_state_dict(ck["model"])
model.eval()
```

El archivo conserva los tensores exactos del detector final. El especialista empleado para la evaluación de coincidencia está en backups/detector70-especialista/especialista-best.pt. Las métricas por clase se encuentran en metrics.json.
