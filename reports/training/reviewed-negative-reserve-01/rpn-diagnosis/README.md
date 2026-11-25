# RPN congelado: localización de la pérdida de propuestas

Control frente a reserva de negativos; mismos checkpoints last.pt a 1950 pasos. Evaluación determinista en lotes de 2, orden original. Sin entrenamiento ni test. Dataset verificado por SHA256. Pesos, buffers y archivos de checkpoints intactos antes/después. Reconstrucción de las propuestas finales exactamente igual a torchvision y cobertura igual al diagnóstico anterior.

## Resultado

Las tres detecciones perdidas todavía tienen cajas ajustadas con IoU ≥0.5 antes del filtrado. Todas desaparecen en la selección de las 2000 propuestas de mayor objectness. No alcanzan NMS. Es un problema observable de posición relativa en el ranking; este diagnóstico no separa causalmente los cambios de puntuación de los cambios en la identidad y geometría de los candidatos.

| Cabeza | Cajas cercanas antes de selección, control → variante | Mejor puesto cercano, control → variante | Cercanas finales, control → variante |
|---|---:|---:|---:|
| Image_01474 / 2 | 16 → 11 | 280 → 2095 | 4 → 0 |
| Image_01087 / 1 | 7 → 3 | 412 → 615 | 2 → 1 |
| Image_01087 / 6 | 0 → 0 | None → None | 0 → 0 |
| Image_02284 / 1 | 104 → 87 | 1354 → 2435 | 1 → 0 |
| Image_01756 / 1 | 125 → 107 | 1857 → 2132 | 2 → 0 |
| Image_01756 / 3 | 73 → 104 | 2369 → 3110 | 0 → 0 |

Los puestos son de base 1, calculados como uno más el número de logits estrictamente mayores; la selección real se reconstruyó con el mismo top-k del RPN. Índices de caja de base cero. Cercana significa IoU ≥0.5.

## Separación de etapas

- Anclas: las tres cabezas perdidas tienen 8, 29 y 23 anclas cercanas en ambos modelos. La geometría base ofrece cobertura.
- Ajuste RPN: quedan 11, 87 y 107 cajas cercanas en la variante, frente a 16, 104 y 125 en el control. La regresión cambia la cobertura, pero no explica por sí sola su desaparición completa.
- Puntuación y selección top-2000: elimina todas las cajas cercanas de esas tres cabezas en la variante; mejores puestos 2095, 2435 y 2132.
- Recorte, tamaño mínimo y umbral de puntuación: no explican pérdidas adicionales de cobertura de estas cabezas después del top-k.
- NMS y límite final de 800: no son el punto de pérdida de esas tres cabezas. En el control NMS reduce duplicados y conserva cobertura.

## Otros casos y decisión

La cabeza Image_01087/1 mantiene cobertura: puesto 412 → 615, dos propuestas finales → una. Image_01087/6 no tiene anclas ni cajas ajustadas con IoU ≥0.5 en ninguno; es un problema distinto. Image_01756/3 ya quedaba fuera del top-2000 en el control (2369) y sigue fuera en la variante (3110).

Mantener el control. Un siguiente experimento congelado puede ampliar únicamente el presupuesto previo a NMS para comprobar si las cajas recuperadas llegan a detección correcta y cuánto aumentan los falsos positivos; no se ejecutó ni activó ese cambio aquí. Recuperar propuestas no garantiza recuperar detecciones. Una semilla y seis cabezas limitan la conclusión.

diagnosis.json conserva todas las etapas, candidatos cercanos, scores, puestos y supervivencia por candidato. Script reproducible: training/diagnose_negative_reservation_rpn.py (requiere destino nuevo).
