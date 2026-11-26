# Generación de imágenes con GAN y difusión latente

Actividad R2-A5-S14 Generación (Moodle 73208).

Este repositorio contiene la implementación de una GAN introductoria y cuadernos de Colab para una DCGAN y un ajuste fino LoRA de Stable Diffusion. El informe principal se entrega como PDF en Moodle.

## Archivos esperados

- `Karen_DCGAN_Rostros_LFW.ipynb`: arquitectura y entrenamiento de DCGAN con LFW.
- `Karen_Diffusers_LoRA_Pokemon.ipynb`: ajuste exploratorio con Diffusers/LoRA.
- `gan_core.py` y `experimento_digitos.py`: GAN en NumPy para imágenes de 8 × 8.
- `resultados_gan_basico/`: métricas y figuras de cinco semillas.

## Estado de las pruebas

La GAN introductoria sobre dígitos se ejecutó en cinco semillas. Su MMD² medio bajó de 0,4137 antes del entrenamiento a 0,0265 al final; la comparación real-real fue 0,0086. Esa medida resume distancia entre distribuciones con el kernel usado en el código y no demuestra calidad o diversidad visual.

Las prácticas de DCGAN y LoRA están preparadas, pero sus cuadernos todavía necesitan ejecutarse en Colab. Por tanto, no se atribuyen a esos modelos imágenes o métricas experimentales en el informe. Tras ejecutarlos, se deben añadir sus salidas y actualizar el análisis.

## Ejecución local

Con Python, NumPy, Matplotlib y scikit-learn: `python experimento_digitos.py`. Las semillas 204–208 están fijadas en el script.

No se deben publicar tokens de Hugging Face ni pesos privados. Revisar las licencias del conjunto de datos y del modelo antes de redistribuir imágenes o adaptadores.
