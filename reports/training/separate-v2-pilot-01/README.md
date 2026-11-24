# Primer entrenamiento del detector de cuatro clases

Completado: 20 épocas, RTX 3060, 14 imágenes train y 5 val. No se usó test. Semilla 20261124, AdamW lr 0.0003, batch 2, sin workers, planificación coseno. Duración del bucle: 27.53 segundos. Se trata de una prueba del flujo con un dataset muy pequeño, no de un modelo listo para conteo.

## Inicialización y datos

Todos los pesos se crearon aleatoriamente; no se cargaron checkpoints anteriores ni pesos externos. Se reprodujo la huella SHA256 inicial creando nuevamente el modelo con la misma semilla. La huella final es distinta. Configuración, hashes de código, versiones, lock del dataset y checkpoints se registran en los JSON adjuntos. `checkpoint-verification.json` contiene las comprobaciones y hashes de los archivos de pesos.

El cargador verifica las rutas y clases y usa las cajas revisadas, con el escalado de imagen aplicado también a las coordenadas. Etiquetas del detector: fondo 0, motocicleta 1, persona ocupante 2, cabeza con casco 3 y cabeza sin casco 4. La salida tiene cinco categorías. Solo entrenamiento usa aumento por reflexión horizontal. Los archivos bloqueados del dataset se verificaron antes de entrenar.

## Resultado

El checkpoint seleccionado por mAP@0.50 en validación corresponde a la época 18.

| Clase | AP@0.50 | Cajas de validación |
|---|---:|---:|
| Motocicleta | 42.57 % | 7 |
| Persona ocupante | 35.30 % | 9 |
| Cabeza con casco | 0 % | 3 |
| Cabeza sin casco | 0 % | 6 |
| Promedio | 19.47 % | 25 |

AP usa interpolación de 101 puntos a IoU 0.50; no es exactitud ni COCO mAP50:95. Se calcula sobre las salidas del detector, no solo las que superan confianza 0.5. AP cero en cabezas significa que no hubo aciertos de esa clase al IoU evaluado; no prueba que no existan propuestas de cabeza. No es comparable directamente con el baseline antiguo, que tenía otra taxonomía y otra validación.

La pérdida de entrenamiento fue 1.508 en la primera época y 1.144 en la última. Con cinco imágenes de validación, estos resultados son muy variables; no se afirma generalización ni suficiencia de datos.

## Verificación e inferencia

25 pruebas unitarias aprobadas, incluyendo cobertura de la cuarta clase en métricas, dimensiones de cajas escaladas y cinco salidas del detector. Se cargó `best.pt` con `weights_only=True` y se ejecutó inferencia en Image_01474: ninguna predicción alcanzó confianza 0.5. `prediction-example.json` y `.jpg` documentan ese resultado, sin seleccionar un ejemplo favorable.

La inferencia reconoce la nueva taxonomía. La asociación antigua de moto+ocupante no se aplica a estas cuatro clases: se devuelve asociación vacía y conteo deshabilitado. Aún hace falta implementar y evaluar la cadena cabeza→persona→moto para este detector.

## Reproducir

```powershell
.venv/Scripts/python.exe training/train.py --task detector --separate-classes --dataset data/prepared/helmet-separate-v2 --epochs 20 --batch-size 2 --workers 0 --seed 20261124 --run detector-separate-v2-pilot-02
```

El nombre debe ser nuevo; no existe reanudación implícita. Las operaciones GPU no prometen reproducción bit a bit de toda la trayectoria. El checkpoint de esta ejecución está en `runs/detector-separate-v2-pilot-01/best.pt`; `last.pt` corresponde a la época 20. El siguiente trabajo debe centrarse en diagnóstico de cabezas y ampliación de ejemplos revisados, antes de afirmar que el sistema identifica casco o habilitar conteo/video.
