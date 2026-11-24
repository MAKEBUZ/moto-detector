# Revisión de relaciones: piloto v2

Este lote separado contiene 7 imágenes de **validación**, 37 entidades y 13 relaciones propuestas. No cambia las etiquetas de v1, sus particiones, sus métricas ni los checkpoints. No se ha reentrenado el modelo.

## Qué se encontró

| Imagen | Propuesta pendiente de revisión humana |
|---|---|
| Image_00146 | Las cajas originales de cabeza y moto con ocupante tienen clases intercambiadas. La cabeza parece no llevar casco. |
| Image_01898 | Revisar cambio de Helmet a NoHelmet: se observa cabello. |
| Image_01575 | Conservar NoHelmet en la cabeza izquierda: lleva un pañuelo. Revisar cambio a Helmet de la cabeza del motociclista más lejano. Separar las tres motos. |
| Image_01720 | Separar el conductor sentado de la persona de pie del fondo; esta última no debe vincularse a la moto. |
| Image_00135 | Proponer vínculos distintos para las dos motos y sus cabezas. |
| Image_02097 | Dos vínculos propuestos; mantener incierta la relación de la cabeza con casco parcialmente ocluida. |
| Image_02354 | Una sola cabeza y un solo ocupante, aunque el detector produzca clases opuestas superpuestas. |

Son observaciones del asistente, **no etiquetas humanas verificadas**. Las cajas de cabeza proceden de las anotaciones existentes y, en un caso de fondo, del detector propio entrenado desde cero. Las cajas originales BikeWithRider se usan solamente como contexto: no representan una moto aislada. Quedan 24 cajas de moto/persona por dibujar o declarar inciertas/rechazadas. No se incorpora ningún modelo externo ni se usan estas propuestas para entrenar.

## Revisar y guardar

1. Abrir `data/annotations/relations-pilot-v2/review.html` en un navegador. Funciona sin conexión; las imágenes están incorporadas.
2. Seleccionar cada entidad. Dibujar una caja ajustada a la parte visible de la moto o persona correspondiente, sin inventar contornos ocluidos. Revisar las cajas de cabeza y su estado de casco. La caja gris discontinua solo sirve de contexto.
3. Marcar `confirmed` cuando la caja y la identidad sean claras; `uncertain` cuando la oclusión impida decidir; `rejected` para una propuesta incorrecta. No confirmar por defecto. El piloto permite editar las entidades existentes; nuevas entidades requieren ampliar el lote.
4. Revisar cabeza → persona → moto. Una persona de fondo debe tener moto `none`. Varias personas pueden compartir moto, pero una persona no debe recibir dos cabezas. No se infiere conductor o pasajero.
5. Marcar la escena como revisada y repetir en las siete imágenes. Usar **Guardar borrador JSON** antes de cerrar si falta trabajo. Es posible cargar ese JSON para continuar.
6. Introducir el identificador del revisor y usar **Validar y exportar revisión**. Guardar el archivo descargado dentro del proyecto; después ejecutar, desde la raíz:

```powershell
.venv/Scripts/python.exe -m training.validate_annotation_batch --reviewed data/annotations/relations-pilot-v2/relations-pilot-reviewed.json --output data/annotations/relations-pilot-v2/validation-receipt.json
```

El recibo no sobrescribe archivos existentes. Si ya existe uno, usar otro nombre. El validador comprueba hashes de los originales, dimensiones, particiones, grupos, cajas, referencias y estados. La declaración humana es una afirmación del revisor, no una verificación de su identidad. Los vínculos inciertos/rechazados se excluyen de `confirmed_links`.

**`training_ready` sigue siendo falso incluso tras revisar:** todas estas imágenes pertenecen a validación. El resultado puede servir de referencia para estudiar asociaciones en este pequeño piloto, nunca trasladarse a entrenamiento. Su selección por errores tampoco permite estimar precisión representativa del sistema. Para entrenar relaciones hacen falta anotaciones independientes de imágenes de entrenamiento.

## Reproducir y verificar

```powershell
.venv/Scripts/python.exe training/build_annotation_batch.py
.venv/Scripts/python.exe -m training.validate_annotation_batch
.venv/Scripts/python.exe -m unittest discover -s training -p "test_*.py"
```

El constructor regenera el borrador y el HTML, por lo que las revisiones deben guardarse con nombres distintos de `draft.json`. El HTML generado se excluye de Git por contener imágenes incrustadas; el borrador JSON y el constructor conservan la trazabilidad.

Validación realizada: estructura del lote y hashes originales correctos; 17 pruebas unitarias aprobadas; sintaxis JavaScript del editor comprobada. No se ha realizado una prueba visual del editor en navegador ni se ha confirmado ninguna anotación en nombre del usuario.
