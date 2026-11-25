# Detector conjunto de cuatro clases

El checkpoint best.pt de la época 9 alcanzó 78,47 % de mAP50 sobre cinco fotos de validación y 74,92 % sobre seis fotos de reserva revisadas. La comparación incluye el control anterior y la evaluación específica de cabezas sin casco en EdgeVision.

Para inferencia, construir build_detector(5, 8, 4), cargar la clave model del checkpoint con load_state_dict y llamar a eval(). El entrenamiento parte de pesos propios.
