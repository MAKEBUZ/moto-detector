# Revisión de negativos difíciles

Abre **review.html**. Son seis imágenes de entrenamiento. Exporta un único archivo **train-hard-negatives-v2-01-reviewed.json** con «Validar y exportar revisión» y guárdalo en esta carpeta. Introduce tu identificador. Puedes guardar/importar un borrador para continuar; no cierres sin guardarlo.

1. Revisa la caja magenta n0. Si no es una cabeza, elige «No es cabeza» y el tipo: asiento, ropa/mochila, pieza/espejo u otra región. La confianza mostrada pertenece al modelo, no certifica la etiqueta.
2. Revisa todas las cabezas reales con/sin casco, personas, motos y relaciones. La moto encierra solo la moto; la persona tiene caja independiente.
3. No dibujes asientos, ropa ni espejos como cabezas. No hay una quinta clase negativo. Estas regiones no son cabezas, aunque pertenezcan a moto/persona.
4. Si la región magenta contiene una cabeza real o es dudosa, usa la opción correspondiente y deja cobertura completa desmarcada. Si faltan entidades que el editor no permita añadir, también deja esa casilla desmarcada y avísame.
5. Confirma únicamente lo distinguible; marca escena revisada y cobertura completa cuando corresponda. Exporta al terminar las seis imágenes.

Imágenes: Image_00616 (ropa), Image_00655 (espejo), Image_02253 (asiento), Image_00297 (espejo/mano), Image_02208 (ropa), Image_02245 (asiento/pieza trasera).

Todo sigue pendiente de revisión humana. No se ampliaron datasets ni se entrenó. Validación intacta.
