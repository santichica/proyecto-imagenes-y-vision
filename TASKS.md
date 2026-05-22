# Lista de tareas — entrega final

Última actualización: 2026-05-22

## Completadas ✅

| # | Tarea |
|---|---|
| 1 | Integrar ramas `gabriel_develop` y `feat/-FID-IS-Metrics` en main |
| 2 | Consolidar y reorganizar notebooks |
| 3 | Documentar metodología bajo el marco CRISP-ML(Q) en METHODOLOGY.md |
| 9 | Limpiar experiments/ — archivar runs incompletos en experiments/archive/ |
| 10 | Actualizar CLAUDE.md — Phase 3 completada, Phase 4 con 6 escenarios reales |
| 11 | Agregar generated_by_user/, *.zip, *.Zone.Identifier al .gitignore |

## Pendientes ⬜

| # | Tarea | Prioridad | Bloqueado por |
|---|---|---|---|
| 4 | Organizar resultados en relación con literatura del problema | 🔴 | Resultados Colab (corriendo ahora) |
| 5 | Referenciar los datos en el repositorio (provenance y acceso) | 🟡 | — |
| 6 | Orientar resultados a conclusiones en README/reporte final | 🔴 | #4 |
| 7 | [Gabriel] Serializar modelos de generación para webapp | 🟡 | Decisión SD vs GAN |
| 8 | Crear requirements.txt para el entorno principal | 🔴 | — |
| 12 | Añadir requirements.txt para webapp y ampliar GAN/README.md | 🟢 | — |

## Decisiones pendientes

- **Webapp SD**: decidir si integrar SD via HF Inference API (Opción B recomendada) o mantener solo GAN
  - GAN ya funciona (`webapp/backend/main.py` + `generator_final.h5`)
  - SD requiere base model ~4GB; opción más práctica es HF Inference API con token
  - Gabriel es responsable de implementar

## Bugs resueltos en sesión (para referencia Colab)

1. `RuntimeError: Input type HalfTensor / weight FloatTensor` → eliminada toda lógica de float16, todo float32
2. `ValueError: 0 samples` en val/test → `image_id` en CSV no tiene extensión, fix: `img_id + '.jpg'`
3. ZIP path → `DRIVE_ROOT / 'data' / classification_data.zip` (estructura Drive real)
4. Ruta synthetic → `DRIVE_ROOT / 'data' / 'synthetic'` (NO en raíz de ham10000-augmentation)
