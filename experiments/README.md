# experiments/

Cada subdirectorio corresponde a un run de clasificación con el formato `YYYYMMDD_HHMMSS_<escenario>`.

## Run canónico de referencia

| Run | Escenario | AUC | Recall mel | F1 mel |
|---|---|---|---|---|
| `20260426_204545_real_only` | real_only (baseline) | 0.9259 | 0.8428 | 0.6036 |

Los resultados de los 6 escenarios comparativos (`real_only`, `real_2x_ti`, `real_2x_lora`, `real_2x_gan`, `real_2x_derm`, `synthetic_only_ti`) se generan con `HAM10000_classification_comparative.ipynb` y se consolidan en `comparative_results.csv`.

## archive/

Runs interrumpidos o preliminares sin `test_metrics.json`. Se conservan para trazabilidad pero no forman parte del análisis.
