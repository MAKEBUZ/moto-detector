# Comparación del piloto revisado

De 11 vínculos humanos confirmados, 5 reciben el candidato correcto. En 6 se localizan ambos extremos.

Confianza ≥ 0.5 e IoU ≥ 0.5; correspondencia uno a uno por confianza. El casco no condiciona la correspondencia de cabezas.

| Imagen | Cabeza → moto | Resultado | Clase de casco correcta |
|---|---|---|---|
| Image_00146 | h0 → m0 | Vínculo candidato correcto | Sí |
| Image_01898 | h0 → m0 | Vínculo candidato correcto | Sí |
| Image_01575 | h0 → m0 | Cabeza no localizada | No evaluable |
| Image_01575 | h1 → m1 | Vínculo candidato correcto | No |
| Image_01575 | h2 → m2 | Moto+ocupante no localizada | No |
| Image_01720 | h0 → m0 | Vínculo candidato correcto | Sí |
| Image_00135 | h0 → m0 | Vínculo candidato correcto | Sí |
| Image_00135 | h1 → m1 | Cabeza no localizada | No evaluable |
| Image_02097 | h0 → m0 | Cabeza no localizada | No evaluable |
| Image_02097 | h1 → m1 | Moto+ocupante no localizada | Sí |
| Image_02354 | h0 → m0 | Abstención por clases opuestas | Sí |

## Alcance

La clase del detector es moto con ocupante. Se usa la caja de contexto original como referencia de identidad; las nuevas cajas de moto aislada no se comparan directamente con esa clase. No hay detector independiente de personas, por lo que no se evalúa la cadena completa cabeza→persona→moto.

Se excluyen 2 vínculos rechazados/inciertos. Hay 4 propuestas de cabeza adicionales fuera de los vínculos puntuados: no se consideran automáticamente falsos positivos. Rechazar una anotación no certifica un ejemplo negativo.

Este piloto fue seleccionado por errores y no representa la precisión global. No se modificaron umbrales, pesos, anotaciones, particiones ni test. Los vínculos del sistema siguen siendo candidatos; el conteo permanece deshabilitado.

Reproducir: `.venv/Scripts/python.exe -m training.evaluate_relations`. El JSON incluye resultados por vínculo y hashes de entradas y código.
