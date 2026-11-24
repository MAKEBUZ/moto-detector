# Ejecución parcial excluida

Se interrumpió la primera ejecución larga para investigar diferencias GPU entre pruebas con y sin observador. No se reanudó ni se usa su checkpoint para comparar resultados. Las ejecuciones completas reiniciadas desde cero y los informes están en ../negative-exposure-02.

Las pruebas cortas de2 pasos sin observador también divergen entre sí. gpu-equivalence-investigation.json registra pérdidas y diagnóstico. Las pruebas unitarias posteriores verifican máscaras RPN/RoI, RNG y pérdidas exactas para el mismo forward en GPU; no se garantiza igualdad de gradientes CUDA. CPU sí conserva pérdidas, gradientes, estado y RNG.
