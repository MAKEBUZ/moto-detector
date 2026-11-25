# Perspectiva alternativa: localizar frente a clasificar

Prueba congelada con las mismas nueve cajas reales de cabeza de validación en ambos modelos. Se omite RPN y se introducen esas cajas directamente en RoIAlign; se conservan imagen completa, backbone, lotes de dos y geometría habitual. Sin entrenamiento, test ni modificación de pesos/buffers; hashes comprobados. Es un diagnóstico con localización conocida, no inferencia utilizable ni estimación de precisión del sistema.

## Resultado

En ambos modelos, fondo es la clase dominante en 7 de las 9 cajas reales: cuatro de seis cabezas sin casco y las tres con casco. Solo dos cabezas sin casco tienen su clase correcta como ganadora entre las cinco salidas. Por tanto, ampliar propuestas no resuelve por sí solo el reconocimiento de cabezas.

Si se compara exclusivamente casco contra sin casco, ignorando fondo, moto y persona, el control elige la clase correcta en 6/9 y la variante en 7/9. Esto es un contrafactual condicionado a que la caja ya es una cabeza: no autoriza quitar fondo en inferencia, porque también convertiría ropa y piezas en detecciones. Tampoco significa que todos los casos con fondo dominante se descarten: la inferencia usa puntuaciones por clase, no solo argmax.

Las cajas reales no necesariamente producen mejores puntuaciones que propuestas más amplias o desplazadas; el contexto y el pooling cambian. Los resultados sugieren investigar sensibilidad a la caja y reconocimiento cabeza/fondo además del ranking RPN, sin concluir todavía una causa única.

## Nueva línea de trabajo propuesta

1. Diagnóstico congelado de recortes con distintos márgenes alrededor de las cabezas, en entrenamiento y validación por separado, para medir sensibilidad al contexto y tamaño. No seleccionar márgenes por la validación como si fuese test independiente.
2. Preparar un experimento controlado desde cero que separe detección de cabeza de clasificación casco/sin casco: localizar cabeza como una clase y clasificar sus recortes con una red pequeña. Conservar negativos de fondo y evaluar errores de la cadena completa; las cajas reales solo sirven para diagnóstico del clasificador.
3. Ampliar ejemplos revisados y una validación independiente con suficientes cabezas de ambas clases. Mantener separación por imagen/grupo al generar recortes para evitar fuga.

No se ha cambiado la arquitectura ni entrenado otra red. Antes de asumir el coste de la nueva arquitectura, conviene comprobar con recortes si esta separación es viable. Comparación actual limitada a nueve cabezas y dos checkpoints de una semilla.

Evidencia completa: diagnosis.json. Script: training/diagnose_oracle_head_classification.py.
