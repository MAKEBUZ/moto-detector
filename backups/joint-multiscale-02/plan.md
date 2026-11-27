# Plan de mejora del detector conjunto

El control es la época 9 de joint-clean640-01. Obtuvo 78,47 % de mAP50 en las cinco imágenes revisadas de validación, 74,92 % en las seis de reserva y F1 de cabezas sin casco medido con confianza e IoU de 0,5 en EdgeVision. El conjunto de prueba histórico ya se ha consultado. Servirá de comparación y quedará pendiente una evaluación externa nueva.

1. Conservar las dos salidas de entrenamiento para las etiquetas de EdgeVision y las cuatro clases revisadas. Inicializar el detector con los pesos propios del control.
2. Entrenar 24 épocas completas sobre las 1.627 imágenes etiquetadas de EdgeVision, intercalando lotes de las 73 imágenes revisadas. Usar precisión mixta, carga en paralelo y promedio exponencial de los pesos.
3. Variar el lado largo entre 576, 640 y 768 píxeles durante el entrenamiento. Evaluar siempre a 640 píxeles.
4. Intercalar desde la primera época ejemplos de las 7.222 fotos elegibles de las seis fuentes. Un detector maestro congelado del proyecto proporciona una señal débil de consistencia de clasificación. Mantener una contribución reducida frente a la supervisión con cajas humanas.
5. Elegir la época por mAP50 en validación. Comparar una sola vez el candidato elegido con el control en reserva revisada y en las 345 fotos de prueba. Para promoverlo exigir mejoras en mAP50 de validación y reserva, y F1 sin casco a umbral 0,5 al menos igual en la prueba amplia. Registrar precisión y sensibilidad.

## Revisión del plan antes de entrenar

El ajuste fijo a 960 píxeles rindió peor en experimentos previos. La nueva prueba mantiene el tamaño de inferencia del control. La consistencia aplicada como etapa final alcanzó 77,50 % frente al 78,47 % del entrenamiento conjunto. En esta corrida se mezcla con supervisión desde el principio, con peso 0,1. La comparación mide el efecto del paquete completo. La reserva pequeña y su uso previo limitan las conclusiones.
